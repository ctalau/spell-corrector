#!/usr/bin/env python3
"""Locked BEA-60K benchmark for Hunspell + reranker vs Aspell top-1."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spelling_reranker.aspell import AspellEngine, aspell_metadata
from spelling_reranker.bea60k import extract_word_errors, load_bea_pairs
from spelling_reranker.byte_encoding import N_CANDIDATE_SLOTS, nfc
from spelling_reranker.candidates import build_pool, pad_pool
from spelling_reranker.hunspell import default_engine
from spelling_reranker.inference import load_model_dir, predict_indices
from spelling_reranker.serialization import MAX_CANDIDATE_BYTES

HIST_KEYS = [
    *[str(i) for i in range(N_CANDIDATE_SLOTS)],
    "present_later",
    "not_present",
    "hunspell_did_not_flag",
    "hunspell_no_suggestions",
]


def classify_gold_position(flagged: bool, suggestions: list[str], gold: str) -> str:
    gold_n = nfc(gold)
    if not flagged:
        return "hunspell_did_not_flag"
    if not suggestions:
        return "hunspell_no_suggestions"
    for i, cand in enumerate(suggestions):
        if nfc(cand) == gold_n:
            return str(i) if i < N_CANDIDATE_SLOTS else "present_later"
    return "not_present"


def write_histogram(counts: Counter[str], n: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    payload = {"n": n, "counts": {}, "percentages": {}}
    for key in HIST_KEYS:
        c = int(counts.get(key, 0))
        pct = 100.0 * c / n if n else 0.0
        payload["counts"][key] = c
        payload["percentages"][key] = pct
        rows.append({"bin": key, "count": c, "percent": f"{pct:.4f}"})
    (out_dir / "hunspell_gold_index_histogram.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "hunspell_gold_index_histogram.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bin", "count", "percent"])
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [r["bin"] for r in rows]
    values = [r["count"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("count")
    ax.set_title("BEA-60K gold position in Hunspell suggestions")
    fig.tight_layout()
    fig.savefig(out_dir / "hunspell_gold_index_histogram.png", dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bea-dir", type=Path, default=ROOT / "data" / "bea60k")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "bea60k")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-aspell", action="store_true")
    args = parser.parse_args()

    pairs = load_bea_pairs(args.bea_dir)
    errors = extract_word_errors(pairs)
    if args.max_examples is not None:
        errors = errors[: args.max_examples]
    n = len(errors)
    print(f"extracted {n} word-level errors from {len(pairs)} sentence pairs", flush=True)

    hunspell = default_engine()
    # BEA repeats many misspellings and suggest() is the slowest step in the
    # sweep, so memoise it on the typo string.
    sugg_cache: dict[str, tuple[bool, tuple[str, ...]]] = {}

    def hunspell_lookup(word: str) -> tuple[bool, list[str]]:
        hit = sugg_cache.get(word)
        if hit is None:
            flagged = not hunspell.spell(word)
            suggestions = tuple(hunspell.suggest(word)) if flagged else ()
            hit = (flagged, suggestions)
            sugg_cache[word] = hit
        return hit[0], list(hit[1])

    model = None
    device = None
    if args.model is not None and Path(args.model).exists():
        import torch

        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = load_model_dir(args.model, device=device)

    # ---- phase 1: Hunspell over every error ----------------------------------
    hist: Counter[str] = Counter()
    hunspell_detect = 0
    hunspell_top1 = 0
    hunspell_oracle_slots = 0
    hunspell_oracle_at_10_legacy = 0
    records: list[dict] = []
    scored: list[int] = []

    print("phase 1: Hunspell", flush=True)
    for err in errors:
        typo, gold = err["typo"], err["gold"]
        flagged, suggestions = hunspell_lookup(typo)
        pool = build_pool(suggestions, limit=N_CANDIDATE_SLOTS, max_bytes=MAX_CANDIDATE_BYTES)
        hist[classify_gold_position(flagged, suggestions, gold)] += 1
        if flagged:
            hunspell_detect += 1
        gold_n = nfc(gold)
        in_pool = any(nfc(c) == gold_n for c in pool)
        hunspell0_ok = bool(pool) and nfc(pool[0]) == gold_n
        if hunspell0_ok:
            hunspell_top1 += 1
        if in_pool:
            hunspell_oracle_slots += 1
        if any(nfc(c) == gold_n for c in suggestions[:10]):
            hunspell_oracle_at_10_legacy += 1
        record = {
            **err,
            "hunspell_flagged": flagged,
            "hunspell_suggestions": suggestions,
            "gold_bucket": classify_gold_position(flagged, suggestions, gold),
            "hunspell_top1_ok": hunspell0_ok,
            "pool": pool,
            "in_top10": in_pool,
            "gold_n": gold_n,
        }
        if model is not None and in_pool:
            scored.append(len(records))
        records.append(record)

    # ---- phase 2: Aspell baseline -------------------------------------------
    aspell_top1 = 0
    aspell_meta = None
    if not args.skip_aspell:
        print("phase 2: Aspell baseline", flush=True)
        aspell = AspellEngine()
        aspell_meta = aspell_metadata()
        aspell_cache: dict[str, str | None] = {}
        try:
            for record in records:
                typo = record["typo"]
                if typo not in aspell_cache:
                    aspell_cache[typo] = aspell.top1(typo)
                top1 = aspell_cache[typo]
                ok = top1 is not None and nfc(top1) == record["gold_n"]
                record["aspell_top1_ok"] = ok
                if ok:
                    aspell_top1 += 1
        finally:
            aspell.close()
    else:
        for record in records:
            record["aspell_top1_ok"] = False

    # ---- phase 3: model, batched --------------------------------------------
    if model is not None and scored:
        print(f"phase 3: model over {len(scored)} solvable errors", flush=True)
        items = [
            (
                records[i]["context_before"],
                records[i]["typo"],
                records[i]["context_after"],
                pad_pool(records[i]["pool"], N_CANDIDATE_SLOTS),
            )
            for i in scored
        ]
        indices = predict_indices(model, items, device=device, batch_size=args.batch_size)
        for i, idx in zip(scored, indices):
            record = records[i]
            record["model_index"] = idx
            if 0 <= idx < len(record["pool"]):
                record["model_word"] = record["pool"][idx]
                record["model_ok"] = nfc(record["model_word"]) == record["gold_n"]

    # ---- tally ---------------------------------------------------------------
    model_overall = 0
    model_conditional_n = 0
    model_conditional_ok = 0
    hunspell_cond_top1 = 0
    movement: Counter[tuple[int, int]] = Counter()
    samples: dict[str, list[dict]] = {
        k: [] for k in ("fixed_top1", "damaged_top1", "both_failed", "gold_outside_top10")
    }

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as pred_f:
        for record in records:
            record.setdefault("model_index", None)
            record.setdefault("model_word", None)
            record.setdefault("model_ok", False)
            pool = record["pool"]
            gold_n = record["gold_n"]
            if model is not None:
                if record["in_top10"]:
                    model_conditional_n += 1
                    if record["model_ok"]:
                        model_conditional_ok += 1
                        model_overall += 1
                    if record["hunspell_top1_ok"]:
                        hunspell_cond_top1 += 1
                    gold_idx = next(i for i, c in enumerate(pool) if nfc(c) == gold_n)
                    if record["model_index"] is not None:
                        movement[(gold_idx, int(record["model_index"]))] += 1
                    if record["model_ok"] and not record["hunspell_top1_ok"] and len(samples["fixed_top1"]) < 50:
                        samples["fixed_top1"].append(record)
                    elif not record["model_ok"] and record["hunspell_top1_ok"] and len(samples["damaged_top1"]) < 50:
                        samples["damaged_top1"].append(record)
                    elif not record["model_ok"] and not record["hunspell_top1_ok"] and len(samples["both_failed"]) < 50:
                        samples["both_failed"].append(record)
                elif len(samples["gold_outside_top10"]) < 50:
                    samples["gold_outside_top10"].append(record)
            out = {k: v for k, v in record.items() if k != "gold_n"}
            pred_f.write(json.dumps(out, ensure_ascii=False) + "\n")

    write_histogram(hist, n, args.output)

    results = {
        "n_sentence_pairs": len(pairs),
        "n_word_errors": n,
        "n_candidate_slots": N_CANDIDATE_SLOTS,
        "hunspell_detection_rate": hunspell_detect / n if n else 0.0,
        "hunspell_top1": hunspell_top1 / n if n else 0.0,
        "hunspell_oracle_at_slots": hunspell_oracle_slots / n if n else 0.0,
        "hunspell_oracle_at_10": hunspell_oracle_at_10_legacy / n if n else 0.0,
        "model_overall_success": (model_overall / n) if model is not None and n else None,
        "model_conditional_accuracy": (
            model_conditional_ok / model_conditional_n
            if model is not None and model_conditional_n
            else None
        ),
        "hunspell_top1_conditional": (
            hunspell_cond_top1 / model_conditional_n
            if model is not None and model_conditional_n
            else None
        ),
        "aspell_top1": (aspell_top1 / n) if n and not args.skip_aspell else None,
        "aspell_metadata": aspell_meta,
        "hunspell_metadata": hunspell.metadata(),
        "model_path": str(args.model) if args.model else None,
        "gold_index_histogram": dict(hist),
        "movement_matrix": {f"{a}->{b}": c for (a, b), c in movement.items()},
    }
    if results["aspell_top1"] is not None and results["model_overall_success"] is not None:
        results["beat_aspell"] = results["model_overall_success"] > results["aspell_top1"]
        results["aspell_delta_pp"] = 100.0 * (
            results["model_overall_success"] - results["aspell_top1"]
        )
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for name, rows in samples.items():
        (args.output / f"examples_{name}.jsonl").write_text(
            "".join(json.dumps({k: v for k, v in r.items() if k != "gold_n"}, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
    print(json.dumps({k: v for k, v in results.items() if k != "movement_matrix"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
