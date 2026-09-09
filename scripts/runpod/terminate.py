#!/usr/bin/env python3
"""Terminate Runpod pods. Run this at the end of every experiment.

Pods bill per second for as long as they exist, whether or not anything is
running on them, so this is not optional cleanup.

  scripts/runpod/terminate.py --all          # every pod on the account
  scripts/runpod/terminate.py <pod-id> ...   # specific pods
  scripts/runpod/terminate.py --list         # show without deleting
"""

from __future__ import annotations

import argparse
import json
import os
import requests

REST = "https://rest.runpod.io/v1"

# Runpod sits behind Cloudflare, which rejects the default urllib user agent
# with a 403 (error 1010). requests also picks up proxy environment variables.
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "spell-corrector-runpod/1.0",
}


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
    parser.add_argument("pod_ids", nargs="*")
    parser.add_argument("--all", action="store_true", help="terminate every pod on the account")
    parser.add_argument("--list", action="store_true", help="list pods and exit")
    args = parser.parse_args()

    pods = api("/pods")
    if not isinstance(pods, list):
        pods = pods.get("pods", [])  # type: ignore[union-attr]

    if args.list or (not args.pod_ids and not args.all):
        if not pods:
            print("no pods")
        for pod in pods:
            print(f"{pod.get('id')}  {pod.get('name')}  {pod.get('desiredStatus')}  ${pod.get('costPerHr')}/hr")
        return 0

    targets = [p.get("id") for p in pods] if args.all else args.pod_ids
    if not targets:
        print("no pods to terminate")
        return 0
    for pod_id in targets:
        api(f"/pods/{pod_id}", "DELETE")
        print(f"terminated {pod_id}")

    remaining = api("/pods")
    if isinstance(remaining, list) and remaining:
        print(f"WARNING: {len(remaining)} pod(s) still present: {[p.get('id') for p in remaining]}")
    else:
        print("no pods remaining")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
