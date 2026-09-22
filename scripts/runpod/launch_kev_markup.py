#!/usr/bin/env python3
"""Create a GPU pod that runs scripts/runpod/bootstrap_kev_markup.sh (kev-4b over the unmarked-markup candidates).

kev-4b in bf16 is ~9GB, so any 16GB+ card holds it; the walk goes cheapest-first like launch_throughput.py, whose
helpers this reuses. Reads RUNPOD_KEY, never prints it. Terminate with scripts/runpod/terminate.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_throughput import api, create_first_available, entrypoint_command, resolve_commit  # noqa: E402

IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
GPUS = [("NVIDIA RTX A5000", 0.16), ("NVIDIA RTX A4500", 0.19), ("NVIDIA RTX 4000 Ada Generation", 0.20),
        ("NVIDIA GeForce RTX 3090", 0.22), ("NVIDIA GeForce RTX 4090", 0.34), ("NVIDIA RTX A6000", 0.33),
        ("NVIDIA L4", 0.39), ("NVIDIA A40", 0.35)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-url", default="https://github.com/ctalau/spell-corrector")
    ap.add_argument("--branch", required=True)
    ap.add_argument("--commit", default=None)
    ap.add_argument("--kev-commit", default="08ab0b87d27cb5577a3b371ad7ed4e4686b0502b")
    ap.add_argument("--kev-run", default="jaredpalmer/kev-4b")
    ap.add_argument("--cloud", default="SECURE", choices=("COMMUNITY", "SECURE"),
                    help="SECURE by default: community hosts with a CUDA 12.4 driver cannot run the cu128 image")
    a = ap.parse_args()

    commit = resolve_commit(a.repo_url, a.branch, a.commit)
    raw = a.repo_url.replace("https://github.com/", "https://raw.githubusercontent.com/")
    boot = f"{raw}/{commit}/scripts/runpod/bootstrap_kev_markup.sh"
    print(f"commit: {commit}\nbootstrap: {boot}")

    def payload(gpu, price):
        return {"name": "kev-markup-audit", "imageName": IMAGE, "gpuTypeIds": [gpu], "gpuCount": 1,
                "cloudType": a.cloud, "containerDiskInGb": 60, "volumeInGb": 0, "ports": ["8000/http"],
                "interruptible": False, "dockerEntrypoint": entrypoint_command(boot), "dockerStartCmd": [],
                "env": {"REPO_URL": a.repo_url, "REPO_BRANCH": a.branch, "REPO_COMMIT": commit,
                        "KEV_COMMIT": a.kev_commit, "KEV_RUN": a.kev_run, "CHOSEN_GPU": gpu,
                        "GPU_PRICE_USD_HR": str(price)}}

    pod, gpu, price, attempts = create_first_available(
        GPUS, payload, lambda p: api("/pods", "POST", p, raise_on_error=False))
    pid = pod["id"]
    for _ in range(60):
        info = api(f"/pods/{pid}")
        if info.get("desiredStatus") == "RUNNING": break
        time.sleep(10)
    print(json.dumps({"id": pid, "gpu": gpu, "usd_per_hr": price, "attempts": attempts,
                      "progress": f"https://{pid}-8000.proxy.runpod.net/run.log"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
