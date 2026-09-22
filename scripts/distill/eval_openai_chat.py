#!/usr/bin/env python3
"""Score a corrector checkpoint on a held-out BEA-60K split via an
OpenAI-compatible server (vLLM), concurrently.

The sibling `eval_gguf.py` scores one request at a time because its subject is
a CPU llama.cpp server whose latency is the point. Here the subject is a GPU
engine that serves hundreds of corrections a second, and 2,000 sequential
requests would spend twenty minutes measuring the round trip instead of the
model. Decode settings are identical to `eval_gguf.py` -- greedy, temperature
0, `max_tokens` 5, the same `normalize_prediction`, the same exact/casefold
pair -- so the accuracy numbers are comparable; the latency numbers are not,
and are labelled under-load.

Splits: held-out only. `data/distill/test.jsonl` and `frozen_100.jsonl` are the
M7 student's held-out sets; `train`/`val` were trained on and scoring them
would measure memorisation. The caller is expected to pass a held-out split,
and the control plane that drives this on a pod refuses the others.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_jsonl, normalize_prediction, write_jsonl  # noqa: E402


def wait_for_server(base_url: str, timeout: float = 900.0) -> float:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                if resp.status == 200:
                    return time.time() - start
        except Exception:  # noqa: BLE001
            time.sleep(2)
    raise SystemExit(f"server at {base_url} never became healthy")


def post(url: str, payload: dict, timeout: float = 300.0) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="spell")
    parser.add_argument("--out-metrics", type=Path, required=True)
    parser.add_argument("--out-predictions", type=Path, default=None)
    parser.add_argument("--max-tokens", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--label", default="")
    parser.add_argument("--quantization", default="")
    parser.add_argument("--backend", default="vLLM OpenAI server")
    parser.add_argument("--thinking", action="store_true", help="leave the model's <think> block on")
    args = parser.parse_args()

    rows = load_jsonl(args.split)
    if args.limit:
        rows = rows[: args.limit]
    startup = wait_for_server(args.base_url)
    url = f"{args.base_url}/v1/chat/completions"

    results: list[dict | None] = [None] * len(rows)
    latencies: list[float] = []
    failures: list[str] = []
    cursor = [0]
    cursor_lock = threading.Lock()
    collect_lock = threading.Lock()

    def worker() -> None:
        local_latencies: list[float] = []
        local_failures: list[str] = []
        while True:
            with cursor_lock:
                i = cursor[0]
                cursor[0] += 1
            if i >= len(rows):
                break
            row = rows[i]
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": row["user_text"]}],
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "stream": False,
            }
            if not args.thinking:
                # The Qwen3.5 template opens a <think> block; with a 5-token
                # budget the whole answer would be reasoning scaffolding. This
                # is llama.cpp's `--reasoning off` by another name.
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            started = time.time()
            try:
                out = post(url, payload)
                raw = out["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001 - counted, not swallowed
                raw = ""
                local_failures.append(f"row {row.get('error_index')}: {str(exc)[:160]}")
            elapsed = time.time() - started
            local_latencies.append(elapsed)
            pred = normalize_prediction(raw)
            results[i] = {
                "error_index": row["error_index"],
                "typo": row["typo"],
                "gold": row["gold"],
                "sentence": row["sentence"],
                "raw_output": raw,
                "pred": pred,
                "exact": pred == row["gold"],
                "casefold": pred.casefold() == row["gold"].casefold(),
                "latency_s": elapsed,
            }
        with collect_lock:
            latencies.extend(local_latencies)
            failures.extend(local_failures)

    t0 = time.time()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, args.concurrency))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.time() - t0

    preds = [r for r in results if r is not None]
    n = max(1, len(preds))
    ordered = sorted(latencies)
    metrics = {
        "label": args.label or args.split.stem,
        "split": str(args.split),
        "n": len(preds),
        "acc@1_exact": sum(r["exact"] for r in preds) / n,
        "acc@1_casefold": sum(r["casefold"] for r in preds) / n,
        "empty_predictions": sum(1 for r in preds if not r["pred"]),
        "request_failures": len(failures),
        "failure_samples": failures[:5],
        "backend": args.backend,
        "quantization": args.quantization,
        "decode": {"greedy": True, "temperature": 0, "max_tokens": args.max_tokens,
                   "enable_thinking": bool(args.thinking)},
        "concurrency": args.concurrency,
        "server_startup_wait_s": round(startup, 2),
        "wall_seconds": round(wall, 2),
        "throughput_rps": round(len(preds) / wall, 2) if wall else 0.0,
        # Under concurrency these are queueing-inclusive, not single-request
        # latencies. Kept for completeness, never comparable to eval_gguf.py's.
        "latency_s_under_load": {
            "p50": statistics.median(ordered) if ordered else 0.0,
            "p90": ordered[int(0.9 * (len(ordered) - 1))] if ordered else 0.0,
            "mean": (sum(ordered) / len(ordered)) if ordered else 0.0,
        },
    }
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if args.out_predictions:
        write_jsonl(args.out_predictions, preds)
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
