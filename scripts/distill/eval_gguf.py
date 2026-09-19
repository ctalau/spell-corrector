#!/usr/bin/env python3
"""Score a Q4_K_M GGUF corrector against a running llama.cpp `llama-server`.

Same decode settings as the M4-M6 CPU serving path: greedy, temperature 0,
`max_tokens` 5, one request at a time, so latency is a single-request number.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_jsonl, normalize_prediction, write_jsonl  # noqa: E402


def post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_for_server(base_url: str, timeout: float = 600.0) -> float:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                if resp.status == 200:
                    return time.time() - start
        except Exception:  # noqa: BLE001
            time.sleep(2)
    raise SystemExit(f"llama-server at {base_url} never became healthy")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--out-metrics", type=Path, required=True)
    ap.add_argument("--out-predictions", type=Path, default=None)
    ap.add_argument("--max-tokens", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--model-path", default="")
    args = ap.parse_args()

    rows = load_jsonl(args.split)
    if args.limit:
        rows = rows[: args.limit]
    startup = wait_for_server(args.base_url)

    url = f"{args.base_url}/v1/chat/completions"
    preds, latencies = [], []
    exact = casefold = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        payload = {
            "messages": [{"role": "user", "content": r["user_text"]}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        t1 = time.time()
        try:
            out = post(url, payload)
            raw = out["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raw = ""
            print(f"request {i} failed: {exc}", flush=True)
        latencies.append(time.time() - t1)
        pred = normalize_prediction(raw)
        is_exact = pred == r["gold"]
        is_cf = pred.casefold() == r["gold"].casefold()
        exact += int(is_exact)
        casefold += int(is_cf)
        preds.append(
            {
                "error_index": r["error_index"],
                "typo": r["typo"],
                "gold": r["gold"],
                "sentence": r["sentence"],
                "raw_output": raw,
                "pred": pred,
                "exact": is_exact,
                "casefold": is_cf,
                "latency_s": latencies[-1],
            }
        )
        if i % 100 == 0:
            print(f"gguf eval {i}/{len(rows)} acc={casefold / max(1, i + 1):.3f}", flush=True)

    n = max(1, len(rows))
    ordered = sorted(latencies)
    metrics = {
        "label": args.label or args.split.stem,
        "split": str(args.split),
        "n": len(rows),
        "acc@1_exact": exact / n,
        "acc@1_casefold": casefold / n,
        "backend": "llama.cpp llama-server",
        "model_path": args.model_path,
        "quantization": "Q4_K_M",
        "decode": {"greedy": True, "temperature": 0, "max_tokens": args.max_tokens},
        "server_startup_wait_s": round(startup, 2),
        "wall_seconds": round(time.time() - t0, 2),
        "latency_s": {
            "p50": statistics.median(ordered),
            "p90": ordered[int(0.9 * (len(ordered) - 1))],
            "p99": ordered[int(0.99 * (len(ordered) - 1))],
            "mean": sum(ordered) / len(ordered),
        },
    }
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.out_predictions:
        write_jsonl(args.out_predictions, preds)
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
