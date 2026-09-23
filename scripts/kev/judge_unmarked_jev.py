#!/usr/bin/env python3
"""Stage 2 of the unmarked-markup audit, with hosted Jev as the judge instead of kev.

Jev is TypeSafe's closed decision model; kev is the open-weights reconstruction judge_unmarked.py runs. OpenRouter
serves Jev at /api/v1/systemone with the same request and answer shapes kev's server uses, so this script asks the
exact questions judge_unmarked.py asks (imported from it, not copied) on the same clipped context, and writes the same
row schema. The answer fields keep kev's names (`kev`, `kev_probs`, `p_*`) so report_unmarked.py scores both runs
unchanged; `judge` records which model answered.

    OPENROUTER_API_KEY=... judge_unmarked_jev.py reports/markup_audit/gold_sample.jsonl --out judged_gold_jev.jsonl

Reads OPENROUTER_API_KEY from the environment and never prints it. Resumable: rows already in --out are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_unmarked import INSTRUCTIONS, NOUL, OPTIONS, clip  # noqa: E402

URL = "https://openrouter.ai/api/v1/systemone"


def ask(key: str, model: str, r: dict) -> dict:
    qs = {"tag": {"type": "choice", "instructions": INSTRUCTIONS.format(match=r["match"]), "criteria": OPTIONS}}
    qs.update({f"is_{el}": {"type": "noul", "instructions": q.format(match=r["match"])} for el, q in NOUL.items()})
    body = json.dumps({"model": model, "state": clip(r["context"]), "questions": qs}).encode()
    for attempt in range(5):
        req = urllib.request.Request(URL, body, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        s = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                res = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            # retry rate limits and provider-side failures only; any other 4xx is a real error
            if attempt == 4 or (e.code < 500 and e.code != 429):
                raise RuntimeError(f"Jev HTTP {e.code}: {e.read()[:300]!r}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt == 4: raise
        time.sleep(2 ** attempt)
    ms = 1000 * (time.perf_counter() - s)
    ans = res["answers"]
    return {**r, "kev": ans["tag"]["choice"], "kev_confidence": ans["tag"]["confidence"],
            "kev_probs": ans["tag"]["probabilities"], **{f"p_{el}": ans[f"is_{el}"]["noul"] for el in NOUL},
            "latency_ms": round(ms, 1), "judge": res["model"], "cost_usd": res["usage"]["cost"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="typesafe/jev-1.13")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget", type=float, default=2.0, help="stop once this many USD have been spent")
    a = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY") or ap.error("set OPENROUTER_API_KEY")

    rows = [json.loads(l) for l in Path(a.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit: rows = rows[: a.limit]
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {(r["file"], r["line"], r["context"]) for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}
    todo = [r for r in rows if (r["file"], r["line"], r["context"]) not in done]

    lock, spent, n = threading.Lock(), [0.0], [0]
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as pool:
        def one(r):
            if spent[0] >= a.budget: return
            j = ask(key, a.model, r)
            with lock:
                f.write(json.dumps(j, ensure_ascii=False) + "\n"); f.flush()
                spent[0] += j["cost_usd"]; n[0] += 1
                if n[0] % 250 == 0: print(f"{n[0]}/{len(todo)} ${spent[0]:.3f}", file=sys.stderr, flush=True)
        for fut in [pool.submit(one, r) for r in todo]: fut.result()
    print(f"judged {n[0]} rows, ${spent[0]:.4f}", file=sys.stderr)
    return 0 if spent[0] < a.budget else 1


if __name__ == "__main__":
    raise SystemExit(main())
