#!/usr/bin/env python3
"""Create a Runpod GPU pod for the training run.

Reads the API key from the RUNPOD_KEY environment variable. Prints the pod id
and the SSH command. Terminate with scripts/runpod/terminate.py -- pods bill
for as long as they exist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
    args = parser.parse_args()

    gpus = args.gpu or GPU_PREFERENCE
    payload = {
        "name": args.name,
        "imageName": args.image,
        "gpuTypeIds": gpus,
        "gpuCount": 1,
        "cloudType": "COMMUNITY",
        "containerDiskInGb": args.disk_gb,
        "volumeInGb": 0,
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "interruptible": False,
    }
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
            print(json.dumps({
                "id": pod_id,
                "gpu": info.get("machine", {}).get("gpuTypeId") or info.get("gpuTypeId"),
                "costPerHr": info.get("costPerHr"),
                "ssh": f"ssh root@{ip} -p {ports['22']} -i ~/.ssh/id_ed25519",
            }, indent=2))
            return 0
        print(f"  status={status} ip={ip} ...")
        time.sleep(10)
    print(f"pod {pod_id} did not expose SSH within {args.wait}s; check the console", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
