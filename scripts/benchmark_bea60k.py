#!/usr/bin/env python3
"""Locked BEA-60K benchmark for Hunspell + reranker vs Aspell top-1."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spelling_reranker.aspell import AspellEngine, aspell_metadata
from spelling_reranker.byte_encoding import N_CANDIDATE_SLOTS, nfc
from spelling_reranker.candidates import build_pool, pad_pool
from spelling_reranker.hunspell import default_engine
from spelling_reranker.inference import load_model_dir, predict_index
from spelling_reranker.serialization import MAX_CANDIDATE_BYTES


def align_errors(noisy: str, clean: str) -> list[dict]:
    n_toks = noisy.split()
    c_toks = clean.split()
    matcher = SequenceMatcher(a=n_toks, b=c_toks, autojunk=False)
    errors: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        n_span = n_toks[i1:i2]
        c_span = c_toks[j1:j2]
        if len(n_span) == 1 and len(c_span) == 1:
            left = " ".join(n_toks[:i1])
            right = " ".join(n_toks[i2:])
            if left:
                left += " "
            if right:
                right = " " + right
            errors.append(
                {
                    "typo": n_span[0],
                    "gold": c_span[0],
                    "context_before": left,
                    "context_after": right,
                    "noisy_sentence": noisy,
                    "clean_sentence": clean,
                }
            )
    return errors


def load_bea_pairs(bea_dir: Path) -> list[tuple[str, str]]:
    clean_path = bea_dir / "test.bea60k"
    noise_path = bea_dir / "test.bea60k.noise"
    if not clean_path.is_file() or not noise_path.is_file():
        raise FileNotFoundError(
            f"BEA files missing in {bea_dir}. Run scripts/download_bea60k.py"
        )
    cleans = clean_path.read_text(encoding="utf-8", errors="replace").splitlines()
    noises = noise_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(cleans) != len(noises):
        raise ValueError(f"line count mismatch: clean={len(cleans)} noise={len(noises)}")
    return list(zip(noises, cleans))


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
    parser.add_argument("--skip-aspell", action="store_true")
    args = parser.parse_args()

    pairs = load_bea_pairs(args.bea_dir)
    errors: list[dict] = []
    for noisy, clean in pairs:
        errors.extend(align_errors(noisy, clean))
    if args.max_examples is not None:
        errors = errors[: args.max_examples]
    n = len(errors)
    print(f"extracted {n} word-level errors from {len(pairs)} sentence pairs")

    hunspell = default_engine()
    aspell = None if args.skip_aspell else AspellEngine()
    model = None
    device = None
    if args.model is not None and Path(args.model).exists():
        import torch

        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = load_model_dir(args.model, device=device)

    hist: Counter[str] = Counter()
    hunspell_detect = 0
    hunspell_top1 = 0
    hunspell_oracle10 = 0
    hunspell_oracle_at_10_legacy = 0
    model_overall = 0
    model_conditional_n = 0
    model_conditional_ok = 0
    hunspell_cond_top1 = 0
    aspell_top1 = 0
    movement: Counter[tuple[int, int]] = Counter()
    samples = {k: [] for k in ("fixed_top1", "damaged_top1", "both_failed", "gold_outside_top10", "context_sensitive")}

    pred_path = args.output
    pred_path.mkdir(parents=True, exist_ok=True)
    pred_f = (pred_path / "predictions.jsonl").open("w", encoding="utf-8")

    try:
        for err in errors:
            typo, gold = err["typo"], err["gold"]
            flagged = not hunspell.spell(typo)
            suggestions = hunspell.suggest(typo)
            top10 = build_pool(
                suggestions, limit=N_CANDIDATE_SLOTS, max_bytes=MAX_CANDIDATE_BYTES
            )
            bucket = classify_gold_position(flagged, suggestions, gold)
            hist[bucket] += 1
            if flagged:
                hunspell_detect += 1
            gold_n = nfc(gold)
            in_top10 = any(nfc(c) == gold_n for c in top10)
            hunspell0_ok = bool(top10) and nfc(top10[0]) == gold_n
            if hunspell0_ok:
                hunspell_top1 += 1
            if in_top10:
                hunspell_oracle10 += 1
            if any(nfc(c) == gold_n for c in suggestions[:10]):
                hunspell_oracle_at_10_legacy += 1

            aspell_ok = False
            if aspell is not None:
                a0 = aspell.top1(typo)
                aspell_ok = a0 is not None and nfc(a0) == gold_n
                if aspell_ok:
                    aspell_top1 += 1

            model_idx = None
            model_word = None
            model_ok = False
            if model is not None and in_top10:
                padded: list[str | None] = pad_pool(top10, N_CANDIDATE_SLOTS)
                model_idx = predict_index(
                    model,
                    err["context_before"],
                    typo,
                    err["context_after"],
                    padded,
                    device=device,
                )
                if 0 <= model_idx < len(top10):
                    model_word = top10[model_idx]
                    model_ok = nfc(model_word) == gold_n
            if model is not None:
                if in_top10:
                    model_conditional_n += 1
                    if model_ok:
                        model_conditional_ok += 1
                        model_overall += 1
                    gold_idx = next(i for i, c in enumerate(top10) if nfc(c) == gold_n)
                    if hunspell0_ok:
                        hunspell_cond_top1 += 1
                    if model_idx is not None:
                        movement[(gold_idx, model_idx)] += 1
                    if model_ok and not hunspell0_ok and len(samples["fixed_top1"]) < 50:
                        samples["fixed_top1"].append(err | {"model": model_word, "hunspell0": top10[0] if top10 else None})
                    if (not model_ok) and hunspell0_ok and len(samples["damaged_top1"]) < 50:
                        samples["damaged_top1"].append(err | {"model": model_word, "hunspell0": top10[0]})
                    if (not model_ok) and (not hunspell0_ok) and len(samples["both_failed"]) < 50:
                        samples["both_failed"].append(err | {"top10": top10, "model": model_word})
                else:
                    if len(samples["gold_outside_top10"]) < 50:
                        samples["gold_outside_top10"].append(err | {"suggestions": suggestions[:15]})

            rec = {
                **err,
                "hunspell_flagged": flagged,
                "hunspell_suggestions": suggestions,
                "gold_bucket": bucket,
                "hunspell_top1_ok": hunspell0_ok,
                "in_top10": in_top10,
                "aspell_top1_ok": aspell_ok,
                "model_index": model_idx,
                "model_word": model_word,
                "model_ok": model_ok,
            }
            pred_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        pred_f.close()
        if aspell is not None:
            aspell.close()

    write_histogram(hist, n, args.output)

    overall_model = model_overall / n if n else 0.0
    results = {
        "n_sentence_pairs": len(pairs),
        "n_word_errors": n,
        "hunspell_detection_rate": hunspell_detect / n if n else 0.0,
        "hunspell_top1": hunspell_top1 / n if n else 0.0,
        "n_candidate_slots": N_CANDIDATE_SLOTS,
        "hunspell_oracle_at_slots": hunspell_oracle10 / n if n else 0.0,
        "hunspell_oracle_at_10": hunspell_oracle_at_10_legacy / n if n else 0.0,
        "model_overall_success": overall_model if model is not None else None,
        "model_conditional_accuracy": (
            model_conditional_ok / model_conditional_n if model is not None and model_conditional_n else None
        ),
        "hunspell_top1_conditional": (
            hunspell_cond_top1 / model_conditional_n if model is not None and model_conditional_n else None
        ),
        "aspell_top1": aspell_top1 / n if n and aspell is not None else None,
        "aspell_metadata": None if args.skip_aspell else aspell_metadata(),
        "hunspell_metadata": hunspell.metadata(),
        "model_path": str(args.model) if args.model else None,
        "gold_index_histogram": dict(hist),
        "movement_matrix": {f"{a}->{b}": c for (a, b), c in movement.items()},
    }
    if results["aspell_top1"] is not None and results["model_overall_success"] is not None:
        delta = 100.0 * (results["model_overall_success"] - results["aspell_top1"])
        results["beat_aspell"] = results["model_overall_success"] > results["aspell_top1"]
        results["aspell_delta_pp"] = delta
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for name, rows in samples.items():
        (args.output / f"examples_{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
