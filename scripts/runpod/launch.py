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
    parser.add_argument("--name", default="spell-corrector-train")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--disk-gb", type=int, default=80)
    parser.add_argument("--gpu", action="append", default=None, help="GPU type id (repeatable)")
    parser.add_argument("--max-price", type=float, default=0.80, help="USD/hr ceiling")
    parser.add_argument("--wait", type=int, default=600, help="seconds to wait for RUNNING")
    parser.add_argument("--repo-url", default="https://github.com/ctalau/spell-corrector")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--target-train", type=int, default=3_000_000)
    parser.add_argument("--target-valid", type=int, default=60_000)
    parser.add_argument("--config", default="configs/train_full.yaml")
    parser.add_argument("--idle", action="store_true", help="do not auto-run the experiment")
    args = parser.parse_args()

    raw_base = args.repo_url.replace("https://github.com/", "https://raw.githubusercontent.com/")
    bootstrap_url = f"{raw_base}/{args.branch}/scripts/runpod/bootstrap.sh"

    gpus = args.gpu or GPU_PREFERENCE
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
            "TARGET_TRAIN": str(args.target_train),
            "TARGET_VALID": str(args.target_valid),
            "CONFIG": args.config,
        },
    }
    if not args.idle:
        # Override the entrypoint, not just the command: the base image wraps
        # CMD in its own init script, which swallowed a start command passed
        # through dockerStartCmd and left the container crash-looping.
        #
        # The command itself stays short and fetches bootstrap.sh from the
        # branch under test, so the pod runs the same script that is in the
        # repo rather than a copy embedded in the pod spec.
        payload["dockerEntrypoint"] = [
            "/bin/bash",
            "-c",
            f"curl -fsSL {bootstrap_url} -o /bootstrap.sh && exec bash /bootstrap.sh",
        ]
        payload["dockerStartCmd"] = []
    if not args.idle:
        print(f"bootstrap: {bootstrap_url}")
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
