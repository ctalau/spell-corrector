#!/usr/bin/env python3
"""Create a Runpod GPU pod that runs the whole experiment unattended.

The pod's start command is `scripts/runpod/bootstrap.sh`, which clones the
repo, sets itself up, runs the experiment, and serves progress and artifacts
over Runpod's HTTP proxy:

    https://<pod-id>-8000.proxy.runpod.net/run.log
    https://<pod-id>-8000.proxy.runpod.net/STATUS
    https://<pod-id>-8000.proxy.runpod.net/DONE        (appears when finished)
    https://<pod-id>-8000.proxy.runpod.net/artifacts/...

Driving the pod this way rather than over SSH means the run needs nothing but
outbound HTTPS, and the experiment is reproducible from a single command.

Reads the API key from RUNPOD_KEY. Terminate with scripts/runpod/terminate.py
-- pods bill for as long as they exist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

REST = "https://rest.runpod.io/v1"

# Runpod sits behind Cloudflare, which rejects the default urllib user agent
# with a 403 (error 1010). requests also picks up proxy environment variables.
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "spell-corrector-runpod/1.0",
}

#: Preference order. RTX 4090 is the best bf16 throughput per dollar here; the
#: rest are fallbacks for when the community cloud has no 4090 free.
GPU_PREFERENCE = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA A40",
    "NVIDIA L40S",
]

#: Ships CUDA torch, so setup.sh never downloads a torch wheel.
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
#: Frozen selector: Python 3.11 so hunspell==0.5.5 builds (setuptools<60), and
#: transformers 4.48.x so ModernBERT imports on the image's torch 2.4.1.
#: Do not use the cu128 / py3.12 / torch 2.8 image until hunspell works on 3.12.
FROZEN_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
#: LLM-judge / Gemma-4: same py3.11 + CUDA 12.4 host image (hunspell). Image
#: torch is 2.4.1; setup_llm_judge.sh upgrades the venv to torch>=2.5.1+cu124
#: so `torch.distributed.tensor.DTensor` exists. Never the cu128 / py3.12
#: image -- those wheels fall back to CPU on Community CUDA 12.4 hosts.
LLM_JUDGE_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def api(path: str, method: str = "GET", payload: dict | None = None) -> dict | list:
    key = os.environ.get("RUNPOD_KEY")
    if not key:
        raise SystemExit("RUNPOD_KEY is not set")
    response = requests.request(
        method,
        f"{REST}{path}",
        json=payload,
        headers={**HEADERS, "Authorization": f"Bearer {key}"},
        timeout=120,
    )
    if response.status_code >= 400:
        raise SystemExit(f"runpod {method} {path} failed {response.status_code}: {response.text[:800]}")
    return response.json() if response.text.strip() else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--image",
        default=None,
        help=(
            "Pod image. Frozen and LLM-judge default is the cu124/py3.11 "
            "image (hunspell). LLM-judge then pip-installs torch>=2.5.1+cu124 "
            "for Gemma-4 / DTensor. Do not pass the cu128/py3.12 torch 2.8 "
            "image for frozen or LLM-judge."
        ),
    )
    parser.add_argument("--disk-gb", type=int, default=None)
    parser.add_argument("--gpu", action="append", default=None, help="GPU type id (repeatable)")
    parser.add_argument("--max-price", type=float, default=None, help="USD/hr ceiling")
    parser.add_argument("--wait", type=int, default=600, help="seconds to wait for RUNNING")
    parser.add_argument("--repo-url", default="https://github.com/ctalau/spell-corrector")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--commit", default=None, help="exact revision (default: branch head)")
    parser.add_argument("--target-train", type=int, default=None)
    parser.add_argument("--target-valid", type=int, default=None)
    parser.add_argument("--experiment", choices=("byte", "frozen"), default="byte",
                        help="byte-level reranker, or frozen ModernBERT selector")
    parser.add_argument("--config", default=None)
    parser.add_argument("--idle", action="store_true", help="do not auto-run the experiment")
    parser.add_argument(
        "--bootstrap-path",
        default="scripts/runpod/bootstrap.sh",
        help="repo-relative path to the entrypoint script the pod fetches and runs",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra environment variable for the pod, repeatable",
    )
    args = parser.parse_args()

    if args.config is None:
        args.config = (
            "configs/train_frozen_modernbert.yaml"
            if args.experiment == "frozen"
            else "configs/train_full.yaml"
        )
    elif "frozen" in Path(args.config).name:
        args.experiment = "frozen"

    frozen = args.experiment == "frozen"
    llm_judge = Path(args.bootstrap_path).name == "bootstrap_llm_judge.sh"
    if args.image is None:
        if frozen:
            args.image = FROZEN_IMAGE
        elif llm_judge:
            args.image = LLM_JUDGE_IMAGE
        else:
            args.image = DEFAULT_IMAGE
    elif (frozen or llm_judge) and (
        "torch280" in args.image
        or "py3.12" in args.image
        or "ubuntu2404" in args.image
        or "cu128" in args.image
    ):
        needed = FROZEN_IMAGE if frozen else LLM_JUDGE_IMAGE
        print(
            "warning: cu128/py3.12 images cannot build hunspell==0.5.5 "
            "(setuptools<60 / ImpImporter) and cu128 torch falls back to "
            f"CPU on CUDA 12.4 hosts. This run needs {needed}",
            file=sys.stderr,
        )
    if args.name is None:
        args.name = "spell-corrector-frozen" if frozen else "spell-corrector-train"
    if args.disk_gb is None:
        args.disk_gb = 100 if frozen else 80
    if args.max_price is None:
        args.max_price = 0.40 if frozen else 0.80
    if args.target_train is None:
        args.target_train = 400_000 if frozen else 3_000_000
    if args.target_valid is None:
        args.target_valid = 40_000 if frozen else 60_000
    gpu_preference = (
        [
            "NVIDIA RTX A5000",
            "NVIDIA GeForce RTX 4090",
            "NVIDIA GeForce RTX 3090",
            "NVIDIA A40",
            "NVIDIA L40S",
        ]
        if frozen
        else GPU_PREFERENCE
    )

    # Pin to an exact commit rather than the branch name. raw.githubusercontent
    # caches branch paths for minutes, so a freshly pushed fix is not
    # necessarily what the pod would fetch -- and the pod should run a known
    # revision anyway.
    commit = args.commit
    if not commit:
        slug = args.repo_url.rstrip("/").removeprefix("https://github.com/")
        ref = requests.get(
            f"https://api.github.com/repos/{slug}/commits/{args.branch}",
            headers={"Accept": "application/vnd.github.sha", **HEADERS},
            timeout=60,
        )
        if ref.status_code >= 400:
            raise SystemExit(f"cannot resolve {args.branch}: {ref.status_code} {ref.text[:200]}")
        commit = ref.text.strip()
    raw_base = args.repo_url.replace("https://github.com/", "https://raw.githubusercontent.com/")
    bootstrap_url = f"{raw_base}/{commit}/{args.bootstrap_path}"

    extra_env = {}
    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        extra_env[key] = value

    gpus = args.gpu or gpu_preference
    payload = {
        "name": args.name,
        "imageName": args.image,
        "gpuTypeIds": gpus,
        "gpuCount": 1,
        "cloudType": "COMMUNITY",
        "containerDiskInGb": args.disk_gb,
        "volumeInGb": 0,
        "ports": ["8000/http", "22/tcp"],
        "supportPublicIp": True,
        "interruptible": False,
        "env": {
            "REPO_URL": args.repo_url,
            "REPO_BRANCH": args.branch,
            "REPO_COMMIT": commit,
            "TARGET_TRAIN": str(args.target_train),
            "TARGET_VALID": str(args.target_valid),
            "CONFIG": args.config,
            "EXPERIMENT": args.experiment,
            **extra_env,
        },
    }
    if not args.idle:
        # Override the entrypoint, not just the command: the base image wraps
        # CMD in its own init script, which swallowed a start command passed
        # through dockerStartCmd and left the container crash-looping.
        #
        # The command fetches bootstrap.sh from the branch under test, so the
        # pod runs the script that is in the repo rather than a copy embedded
        # in the pod spec.
        #
        # It stays small and defensive: it starts the progress server itself
        # and never exits, so a failure inside the bootstrap shows up as a
        # readable log instead of a crash-looping container with nothing
        # listening.
        # The fetch is retried: on a cold container the entrypoint can run
        # before networking is ready, and a single failed curl would leave the
        # pod idling forever with nothing to do.
        entry = (
            "mkdir -p /workspace/out; "
            "(nohup python3 -m http.server 8000 --directory /workspace/out "
            ">/dev/null 2>&1 &); "
            "for i in $(seq 1 30); do "
            f"curl -fsSL {bootstrap_url} -o /workspace/out/bootstrap.sh && break; "
            "echo \"entrypoint: fetch attempt $i failed\" >> /workspace/out/run.log; "
            "sleep 10; done; "
            "if [ -s /workspace/out/bootstrap.sh ]; then "
            # -x, and stderr captured separately: a bootstrap that dies before
            # its own logging is set up would otherwise leave no trace at all.
            "bash -x /workspace/out/bootstrap.sh 2>>/workspace/out/boot.err; "
            "echo \"entrypoint: bootstrap exited rc=$?\" >> /workspace/out/run.log; "
            "else echo 'entrypoint: could not fetch bootstrap' >> /workspace/out/run.log; fi; "
            "sleep infinity"
        )
        payload["dockerEntrypoint"] = ["/bin/bash", "-lc", entry]
        payload["dockerStartCmd"] = []
        print(f"commit:    {commit}")
        print(f"bootstrap: {bootstrap_url}")
        print(f"experiment:{args.experiment} config={args.config}")
        print(f"image:     {args.image}")
    print(f"creating pod over {gpus} (<= ${args.max_price}/hr)...")
    pod = api("/pods", "POST", payload)
    pod_id = pod.get("id")
    if not pod_id:
        raise SystemExit(f"no pod id in response: {json.dumps(pod)[:800]}")
    print(f"pod id: {pod_id}")

    deadline = time.time() + args.wait
    while time.time() < deadline:
        info = api(f"/pods/{pod_id}")
        status = info.get("desiredStatus") or info.get("lastStatusChange")
        ip = info.get("publicIp")
        ports = info.get("portMappings") or {}
        if ip and ports.get("22"):
            base = f"https://{pod_id}-8000.proxy.runpod.net"
            print(json.dumps({
                "id": pod_id,
                "vcpu": info.get("vcpuCount"),
                "ramGb": info.get("memoryInGb"),
                "costPerHr": info.get("costPerHr"),
                "ssh": f"ssh root@{ip} -p {ports['22']} -i ~/.ssh/id_ed25519",
                "log": f"{base}/run.log",
                "status": f"{base}/STATUS",
                "done": f"{base}/DONE",
                "artifacts": f"{base}/artifacts/",
            }, indent=2))
            return 0
        print(f"  status={status} ip={ip} ...")
        time.sleep(10)
    print(f"pod {pod_id} did not expose SSH within {args.wait}s; check the console", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
