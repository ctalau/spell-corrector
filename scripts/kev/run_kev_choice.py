#!/usr/bin/env python3
"""kev as the candidate chooser, in the seat "Jev Choice" occupied.

The milestone-3/4 reports quote a prior, external number -- "Jev Choice
(approx, prior) ~91%, API chooser over lists" -- measured on the frozen 100
typos with classic candidate lists. Jev is hosted and closed; kev
(github.com/jaredpalmer/kev) is an open-weights reconstruction of the same
architecture that serves the same /v1/systemone contract, so it can be put in
the same seat and actually run here.

Protocol: same frozen 100 typos, Hunspell candidates as the list (the only
candidate generator this repository owns), one `choice` question per typo,
argmax over the returned probabilities. Scoring is casefold, as in milestone 3.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEV = Path("/home/user/jaredpalmer/kev")
for p in (str(ROOT), str(KEV)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from spelling_reranker.candidates import build_pool  # noqa: E402
from spelling_reranker.hunspell import default_engine  # noqa: E402

FROZEN = ROOT / "artifacts/spell_slm_m7/results/predictions_teacher_nf4_frozen_100.jsonl"


def load_frozen() -> list[dict]:
    rows = []
    for line in FROZEN.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows.append({k: r[k] for k in ("error_index", "typo", "gold", "sentence")})
    return rows


def build_request(sentence: str, typo: str, candidates: list[str], instructions: str) -> dict:
    """The /v1/systemone request kev's API shapes accept: one choice question."""
    return {
        "state": sentence,
        "model": "kev-latest",
        "questions": {
            "correction": {
                "type": "choice",
                "instructions": instructions.format(typo=typo),
                # value None -> the option is rendered as the bare word, no description
                "criteria": {c: None for c in candidates},
            }
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-0.5b")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--limit-candidates", type=int, default=8,
                    help="cap on the Hunspell list; 0 = uncapped")
    ap.add_argument("--instructions", default=
                    "The word <TYPO>{typo}</TYPO> in the text is misspelled. "
                    "Which option is the word the writer meant?")
    ap.add_argument("--max-items", type=int, default=0, help="0 = all 100")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import os
    if a.dtype == "bf16":
        os.environ["KEV_DTYPE"] = "bf16"
    from kev.api import SystemOneRequest, to_answers, to_record
    from kev.evaluate import load
    from kev.model import MAX_BRANCH, MAX_STATE

    hunspell = default_engine()
    rows = load_frozen()
    if a.max_items:
        rows = rows[: a.max_items]

    t0 = time.perf_counter()
    tok, model = load(a.run, a.device)
    load_seconds = time.perf_counter() - t0

    limit = a.limit_candidates or 10_000
    records, latencies, truncated = [], [], 0
    for i, row in enumerate(rows):
        pool = build_pool(hunspell.suggest_raw(row["typo"]), limit=limit)
        gold_in_list = any(c.casefold() == row["gold"].casefold() for c in pool)
        if not pool:
            records.append({**row, "candidates": [], "pred": None, "correct": False,
                            "gold_in_list": False, "hunspell_top1": None,
                            "hunspell_top1_correct": False, "latency_ms": 0.0})
            continue

        req = SystemOneRequest(**build_request(row["sentence"], row["typo"], pool, a.instructions))
        record, meta = to_record(req)
        record["questions"][0]["label"] = 0  # unused; encode() wants the key
        enc = model.encode(tok, record, max_state=MAX_STATE, max_branch=MAX_BRANCH)
        if enc.get("state_truncated"):
            truncated += 1
        start = time.perf_counter()
        with torch.no_grad():
            probs = model.probs(enc)
        latency_ms = 1000 * (time.perf_counter() - start)
        latencies.append(latency_ms)

        answers = to_answers([p.tolist() for p in probs], meta)
        pred = answers["correction"]["choice"]
        top1 = pool[0]
        records.append({
            **row,
            "candidates": pool,
            "n_candidates": len(pool),
            "pred": pred,
            "confidence": answers["correction"]["confidence"],
            "correct": pred.casefold() == row["gold"].casefold(),
            "gold_in_list": gold_in_list,
            "hunspell_top1": top1,
            "hunspell_top1_correct": top1.casefold() == row["gold"].casefold(),
            "latency_ms": latency_ms,
        })
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)

    n = len(records)
    correct = sum(r["correct"] for r in records)
    covered = [r for r in records if r["gold_in_list"]]
    hs_correct = sum(r["hunspell_top1_correct"] for r in records)
    lat = sorted(latencies)

    def q(p: float) -> float:
        return lat[min(len(lat) - 1, int(p * len(lat)))] if lat else 0.0

    report = {
        "run": a.run,
        "device": a.device,
        "dtype": a.dtype,
        "instructions": a.instructions,
        "limit_candidates": a.limit_candidates,
        "n": n,
        "overall_accuracy": correct / n,
        "coverage_gold_in_list": len(covered) / n,
        "conditional_accuracy": (sum(r["correct"] for r in covered) / len(covered)) if covered else None,
        "hunspell_top1_accuracy": hs_correct / n,
        "mean_candidates": sum(r.get("n_candidates", 0) for r in records) / n,
        "truncated_records": truncated,
        "load_seconds": load_seconds,
        "latency_ms": {"p50": q(0.5), "p90": q(0.9), "mean": sum(lat) / len(lat) if lat else 0.0},
    }

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
