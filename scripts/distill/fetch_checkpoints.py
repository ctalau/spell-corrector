#!/usr/bin/env python3
"""Pull the student's checkpoints off a running pod, so a dead pod costs nothing.

The pod publishes `artifacts/student/best/` (rewritten whenever dev accuracy
improves) and `metrics.jsonl` over its HTTP proxy. This polls both, and keeps a
local copy of every distinct checkpoint step under `--dest`, newest also copied
to `latest/`. It is a plain download loop: no pod-side cooperation beyond the
progress server that is already there.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

FILES = ("adapter_model.safetensors", "adapter_config.json", "best_meta.json")


#: Runpod's proxy sits behind Cloudflare, which answers urllib's default user
#: agent with a 403 (error 1010) -- the same trap scripts/runpod/launch.py
#: documents. Without this header every download here silently returns nothing.
HEADERS = {"User-Agent": "spell-corrector-runpod/1.0"}


def get(url: str, timeout: float = 120.0, quiet: bool = True) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:  # noqa: BLE001
        if not quiet:
            print(f"fetch failed {url}: {type(exc).__name__}: {exc}", flush=True)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pod", required=True, help="pod id")
    ap.add_argument("--dest", type=Path, default=Path("artifacts/distill_checkpoints"))
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--max-hours", type=float, default=8.0)
    args = ap.parse_args()

    base = f"https://{args.pod}-8000.proxy.runpod.net"
    args.dest.mkdir(parents=True, exist_ok=True)
    seen_step = None
    deadline = time.time() + args.max_hours * 3600

    while time.time() < deadline:
        meta_raw = get(f"{base}/artifacts/student/best/best_meta.json", timeout=60, quiet=False)
        metrics = get(f"{base}/PROGRESS", timeout=120)
        if metrics:
            (args.dest / "metrics.jsonl").write_bytes(metrics)
        for name in ("run_meta.json", "train_summary.json"):
            blob = get(f"{base}/artifacts/student/{name}", timeout=60)
            if blob:
                (args.dest / name).write_bytes(blob)

        if meta_raw:
            try:
                meta = json.loads(meta_raw)
            except json.JSONDecodeError:
                meta = None
            if meta and meta.get("step") != seen_step:
                step = meta["step"]
                out = args.dest / f"step_{step:06d}"
                out.mkdir(parents=True, exist_ok=True)
                ok = True
                for name in FILES:
                    blob = get(f"{base}/artifacts/student/best/{name}", timeout=600)
                    if blob is None:
                        ok = False
                        break
                    (out / name).write_bytes(blob)
                if ok:
                    seen_step = step
                    latest = args.dest / "latest"
                    if latest.exists():
                        shutil.rmtree(latest)
                    shutil.copytree(out, latest)
                    print(
                        f"[{time.strftime('%H:%M:%S')}] saved checkpoint step={step} "
                        f"dev_casefold={meta.get('dev_acc_casefold')} -> {out}",
                        flush=True,
                    )
                else:
                    shutil.rmtree(out, ignore_errors=True)
                    print(f"[{time.strftime('%H:%M:%S')}] step {step} download incomplete", flush=True)

        if get(f"{base}/DONE", timeout=30) is not None:
            print("pod reports DONE; final fetch complete", flush=True)
            return 0
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
