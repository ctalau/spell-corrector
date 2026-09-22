#!/usr/bin/env python3
"""Create the 3090 pod for the 0.8B throughput hill-climb.

Same self-driving-entrypoint shape as the other launchers here (the sandbox has
outbound HTTPS only, and the base images swallow `dockerStartCmd`), with one
difference: this pod does not run an experiment. It prepares a serving stack
and then waits on a control port, because a hill-climb picks step N+1 from step
N's number and so has to stay under interactive control.

    RUNPOD_KEY=... python scripts/runpod/launch_throughput.py

Two ports are exposed over the Runpod proxy:

    https://<pod-id>-8000.proxy.runpod.net/run.log   setup progress (public)
    https://<pod-id>-8001.proxy.runpod.net/status    control plane (token)

The control token is generated here, passed to the pod as an environment
variable and written to `--token-out` -- which must not be inside the
repository, since it authenticates a public URL.

Terminate with scripts/runpod/terminate.py <pod-id>. Never `--all` unless you
know every pod on the account is yours: pods bill per second either way.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
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

#: The card the experiment is about, then the Ampere 24GB siblings that would
#: answer the same question if the community cloud has no 3090 free. Prices are
#: community on-demand as quoted 2026-09-11; the API's actual price is recorded
#: in the launcher's run record.
GPU_CANDIDATES: list[tuple[str, float]] = [
    ("NVIDIA GeForce RTX 3090", 0.22),
    ("NVIDIA GeForce RTX 3090 Ti", 0.27),
    ("NVIDIA RTX A5000", 0.16),
    ("NVIDIA RTX A6000", 0.33),
]

#: vLLM ships torch, CUDA and its kernels already matched in this image;
#: v0.29.0 rather than the v0.30.0 released hours before this run, because a
#: throughput number from a same-day release is a number about that release.
DEFAULT_IMAGE = "vllm/vllm-openai:v0.29.0"
DEFAULT_HF_MODEL = "ctalau/qwen35-08b-spell-m7-distill"


def api(path: str, method: str = "GET", payload: dict | None = None, *, raise_on_error: bool = True):
    key = os.environ.get("RUNPOD_KEY")
    if not key:
        raise SystemExit("RUNPOD_KEY is not set")
    response = requests.request(
        method, f"{REST}{path}", json=payload,
        headers={**HEADERS, "Authorization": f"Bearer {key}"}, timeout=120,
    )
    if response.status_code >= 400:
        message = f"runpod {method} {path} failed {response.status_code}: {response.text[:800]}"
        if raise_on_error:
            raise SystemExit(message)
        return {"_error": message, "_status": response.status_code}
    return response.json() if response.text.strip() else {}


def resolve_commit(repo_url: str, branch: str, commit: str | None) -> str:
    """Pin an exact SHA: raw.githubusercontent caches branch paths for minutes,
    so a freshly pushed fix is not necessarily what the pod would fetch."""
    if commit:
        return commit
    slug = repo_url.rstrip("/").removeprefix("https://github.com/")
    ref = requests.get(
        f"https://api.github.com/repos/{slug}/commits/{branch}",
        headers={"Accept": "application/vnd.github.sha", **HEADERS}, timeout=60,
    )
    if ref.status_code >= 400:
        raise SystemExit(f"cannot resolve {branch}: {ref.status_code} {ref.text[:200]}")
    return ref.text.strip()


def entrypoint_command(bootstrap_url: str) -> list[str]:
    """Small, defensive, and it never exits.

    It starts the progress server itself and retries the fetch: on a cold
    container the entrypoint can run before networking is up, and one failed
    fetch would leave a pod idling with nothing listening. `curl` is not
    assumed -- the vLLM image is not the pytorch image -- so the fetch goes
    through the python that is certainly there.
    """
    fetch = (
        "import sys,urllib.request;"
        f"open('/workspace/out/bootstrap.sh','wb').write(urllib.request.urlopen('{bootstrap_url}',timeout=60).read())"
    )
    entry = (
        "mkdir -p /workspace/out; "
        "(nohup python3 -m http.server 8000 --directory /workspace/out >/dev/null 2>&1 &); "
        "for i in $(seq 1 30); do "
        f"python3 -c \"{fetch}\" && break; "
        'echo "entrypoint: fetch attempt $i failed" >> /workspace/out/run.log; '
        "sleep 10; done; "
        "if [ -s /workspace/out/bootstrap.sh ]; then "
        "bash /workspace/out/bootstrap.sh 2>>/workspace/out/boot.err; "
        'echo "entrypoint: bootstrap exited rc=$?" >> /workspace/out/run.log; '
        "else echo 'entrypoint: could not fetch bootstrap' >> /workspace/out/run.log; fi; "
        "sleep infinity"
    )
    return ["/bin/bash", "-lc", entry]


def create_first_available(candidates, build_payload, create):
    """Try each GPU type in order; return the first pod that is actually created."""
    attempts: list[dict] = []
    for gpu, price in candidates:
        result = create(build_payload(gpu, price))
        if isinstance(result, dict) and result.get("id"):
            attempts.append({"gpu": gpu, "expected_usd_per_hr": price, "result": "created"})
            return result, gpu, price, attempts
        attempts.append({"gpu": gpu, "expected_usd_per_hr": price, "result": "unavailable",
                         "error": (result or {}).get("_error", "no pod id in response")[:300]})
    raise SystemExit("no GPU type had capacity; attempts:\n" +
                     "\n".join(f"  {a['gpu']}: {a.get('error')}" for a in attempts))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="spell-08b-3090-throughput")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--disk-gb", type=int, default=80,
                        help="the vLLM image is ~10GB, plus fp16 (1.7GB) and the quantized copy")
    parser.add_argument("--gpu", action="append", default=None, metavar="GPU_TYPE_ID",
                        help="override the candidate order (repeatable)")
    parser.add_argument("--cloud", default="COMMUNITY", choices=("COMMUNITY", "SECURE"))
    parser.add_argument("--wait", type=int, default=600)
    parser.add_argument("--repo-url", default="https://github.com/ctalau/spell-corrector")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--commit", default=None)
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL)
    parser.add_argument("--quant-scheme", default="W4A16")
    parser.add_argument("--quant-algorithm", default="gptq", choices=("gptq", "awq", "rtn"))
    parser.add_argument("--quant-samples", type=int, default=256)
    parser.add_argument("--bootstrap-path", default="scripts/runpod/bootstrap_throughput.sh")
    parser.add_argument("--token-out", type=Path, required=True,
                        help="where to write the control token + URLs; keep it OUT of the repo")
    args = parser.parse_args()

    commit = resolve_commit(args.repo_url, args.branch, args.commit)
    raw_base = args.repo_url.replace("https://github.com/", "https://raw.githubusercontent.com/")
    bootstrap_url = f"{raw_base}/{commit}/{args.bootstrap_path}"
    token = secrets.token_urlsafe(24)
    candidates = ([(g, dict(GPU_CANDIDATES).get(g, 0.0)) for g in args.gpu]
                  if args.gpu else list(GPU_CANDIDATES))

    print(f"commit:    {commit}")
    print(f"bootstrap: {bootstrap_url}")
    print(f"image:     {args.image}")
    print(f"model:     {args.hf_model} -> {args.quant_algorithm} {args.quant_scheme}")

    def build_payload(gpu: str, price: float) -> dict:
        return {
            "name": args.name,
            "imageName": args.image,
            "gpuTypeIds": [gpu],
            "gpuCount": 1,
            "cloudType": args.cloud,
            "containerDiskInGb": args.disk_gb,
            "volumeInGb": 0,
            "ports": ["8000/http", "8001/http", "22/tcp"],
            "supportPublicIp": True,
            "interruptible": False,
            "env": {
                "REPO_URL": args.repo_url,
                "REPO_BRANCH": args.branch,
                "REPO_COMMIT": commit,
                "CHOSEN_GPU": gpu,
                "GPU_PRICE_USD_HR": str(price),
                "HF_MODEL": args.hf_model,
                "QUANT_SCHEME": args.quant_scheme,
                "QUANT_ALGORITHM": args.quant_algorithm,
                "QUANT_SAMPLES": str(args.quant_samples),
                "CONTROL_TOKEN": token,
                "VLLM_LOGGING_LEVEL": "INFO",
                "HF_HUB_ENABLE_HF_TRANSFER": "0",
            },
            "dockerEntrypoint": entrypoint_command(bootstrap_url),
            "dockerStartCmd": [],
        }

    pod, gpu, price, attempts = create_first_available(
        candidates, build_payload, lambda payload: api("/pods", "POST", payload, raise_on_error=False)
    )
    pod_id = pod["id"]
    print(f"\npod id:    {pod_id}\ngpu:       {gpu} (expected ${price}/hr)")

    record = {
        "id": pod_id, "gpu": gpu, "expectedCostPerHr": price, "commit": commit,
        "image": args.image, "hf_model": args.hf_model, "attempts": attempts,
        "progress": f"https://{pod_id}-8000.proxy.runpod.net/run.log",
        "status_file": f"https://{pod_id}-8000.proxy.runpod.net/STATUS",
        "ready": f"https://{pod_id}-8000.proxy.runpod.net/READY",
        "control": f"https://{pod_id}-8001.proxy.runpod.net",
        "control_token": token,
    }
    deadline = time.time() + args.wait
    while time.time() < deadline:
        info = api(f"/pods/{pod_id}")
        record["costPerHr"] = info.get("costPerHr", price)
        record["vcpu"] = info.get("vcpuCount")
        record["ramGb"] = info.get("memoryInGb")
        if info.get("publicIp") or info.get("desiredStatus") == "RUNNING":
            break
        print(f"  status={info.get('desiredStatus')} ...")
        time.sleep(10)

    args.token_out.parent.mkdir(parents=True, exist_ok=True)
    args.token_out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.chmod(args.token_out, 0o600)
    printable = {k: v for k, v in record.items() if k != "control_token"}
    print(json.dumps(printable, indent=2))
    print(f"\ncontrol token written to {args.token_out} (not printed, not committed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
