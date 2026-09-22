#!/usr/bin/env python3
"""Client for the pod's throughput control plane. Runs in the sandbox.

    python scripts/runpod/throughput_driver.py status  --token-file T
    python scripts/runpod/throughput_driver.py job     --token-file T --job step.json
    python scripts/runpod/throughput_driver.py results --token-file T
    python scripts/runpod/throughput_driver.py file    --token-file T --path server_s01.log --tail 4000
    python scripts/runpod/throughput_driver.py wait    --token-file T --timeout 900

The token file is the launcher's `--token-out` record: it holds the control URL
and the token, and lives outside the repository.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

HEADERS = {"User-Agent": "spell-corrector-runpod/1.0"}


def load(token_file: Path) -> tuple[str, str]:
    record = json.loads(token_file.read_text(encoding="utf-8"))
    return record["control"].rstrip("/"), record["control_token"]


def call(base: str, token: str, path: str, method: str = "GET", payload=None, timeout: int = 90):
    response = requests.request(
        method, f"{base}{path}", json=payload,
        headers={**HEADERS, "X-Control-Token": token}, timeout=timeout,
    )
    return response.status_code, response.text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("status", "job", "results", "file", "wait", "shutdown"))
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--job", type=Path, help="JSON file describing one step")
    parser.add_argument("--path", default="", help="file to fetch, relative to the pod's output dir")
    parser.add_argument("--tail", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1800, help="for `wait`")
    parser.add_argument("--poll", type=int, default=30, help="for `wait`")
    args = parser.parse_args()

    base, token = load(args.token_file)

    if args.action == "status":
        code, text = call(base, token, "/status")
        print(text if code == 200 else f"{code}: {text[:400]}")
        return 0 if code == 200 else 1
    if args.action == "job":
        job = json.loads(args.job.read_text(encoding="utf-8"))
        code, text = call(base, token, "/job", "POST", job)
        print(f"{code}: {text}")
        return 0 if code < 300 else 1
    if args.action == "results":
        code, text = call(base, token, "/results")
        print(text if code == 200 else f"{code}: {text[:400]}")
        return 0 if code == 200 else 1
    if args.action == "file":
        query = f"/file?p={args.path}" + (f"&tail={args.tail}" if args.tail else "")
        code, text = call(base, token, query)
        print(text if code == 200 else f"{code}: {text[:400]}")
        return 0 if code == 200 else 1
    if args.action == "shutdown":
        code, text = call(base, token, "/shutdown", "POST", {})
        print(f"{code}: {text}")
        return 0

    # wait: block until the control plane is idle (or failed) with nothing queued
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        try:
            code, text = call(base, token, "/status", timeout=30)
            state = json.loads(text) if code == 200 else {}
        except Exception as exc:  # noqa: BLE001 - a pod that is busy can refuse a connection
            print(f"  (poll failed: {str(exc)[:120]})")
            time.sleep(args.poll)
            continue
        label = f"{state.get('state')} current={state.get('current')} queued={state.get('queued')}"
        last = state.get("last")
        if last:
            label += f" last={last.get('name')}@c{last.get('concurrency')} {last.get('rps')}rps"
        print(f"  {time.strftime('%H:%M:%S')} {label}")
        if state.get("state") in ("idle", "failed") and not state.get("queued") and not state.get("current"):
            print(json.dumps(state, indent=2))
            return 0 if state.get("state") == "idle" else 2
        time.sleep(args.poll)
    print("timed out waiting", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
