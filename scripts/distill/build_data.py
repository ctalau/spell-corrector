#!/usr/bin/env python3
"""Build the M7 distillation dataset from BEA-60K.

Enumeration is `spelling_reranker.bea60k.extract_word_errors`, the same order
the M4/M5/M6 milestones used, so `error_index` is comparable across them.

Splits, all disjoint at the level of the *source sentence* (a BEA line can hold
several word errors; putting two of them on opposite sides of a split would
leak context):

    frozen_100  the M4-M6 holdout, reconstructed from the committed M6
                predictions, with the same two gold overrides
    test        a larger held-out sample, scored once at the end
    dev         progress tracking and checkpoint selection during training
    val         teacher-forced loss only
    train       everything left

Nothing outside `train`/`val` is ever trained on, and the leak check at the end
fails the build if a sentence or error index appears in two splits.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_bea_module():
    spec = importlib.util.spec_from_file_location(
        "bea60k", ROOT / "spelling_reranker" / "bea60k.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def render_sentence(err: dict) -> str:
    return f"{err['context_before']}<TYPO>{err['typo']}</TYPO>{err['context_after']}"


def render_user_text(prompt_template: str, sentence: str) -> str:
    return prompt_template.replace("{{SENTENCE}}", sentence)


def row(idx: int, err: dict, sent_id: int, prompt_template: str, gold: str) -> dict:
    sentence = render_sentence(err)
    return {
        "error_index": idx,
        "sentence_index": sent_id,
        "typo": err["typo"],
        "gold": gold,
        "sentence": sentence,
        "user_text": render_user_text(prompt_template, sentence),
        "target": gold,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bea-dir", type=Path, default=ROOT / "data" / "bea60k")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "distill")
    ap.add_argument(
        "--m6-predictions",
        type=Path,
        default=ROOT / "artifacts" / "spell_slm_m6" / "predictions_m6_qlora_2b.jsonl",
        help="source of the frozen seed-1337 BEA-100 error indices and gold overrides",
    )
    ap.add_argument(
        "--prompt",
        type=Path,
        default=ROOT / "artifacts" / "spell_slm_m6" / "direct_correct_v1.txt",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--test-size", type=int, default=2000)
    ap.add_argument("--dev-size", type=int, default=1000)
    ap.add_argument("--val-size", type=int, default=1000)
    ap.add_argument("--max-train", type=int, default=0, help="0 = every remaining error")
    args = ap.parse_args()

    bea = _load_bea_module()
    pairs = bea.load_bea_pairs(args.bea_dir)
    errors = bea.extract_word_errors(pairs)

    # sentence id: every error carries its noisy line, which identifies the pair
    sent_ids: dict[str, int] = {}
    err_sent: list[int] = []
    for err in errors:
        key = err["noisy_sentence"]
        if key not in sent_ids:
            sent_ids[key] = len(sent_ids)
        err_sent.append(sent_ids[key])

    prompt_template = args.prompt.read_text(encoding="utf-8").strip()

    # --- frozen 100 ---------------------------------------------------------
    frozen_meta = [json.loads(l) for l in args.m6_predictions.read_text(encoding="utf-8").splitlines() if l.strip()]
    frozen_idx = [int(p["error_index"]) for p in frozen_meta]
    overrides: dict[int, str] = {}
    mismatch = []
    for p in frozen_meta:
        i = int(p["error_index"])
        if errors[i]["typo"] != p["typo"] or render_sentence(errors[i]) != p["sentence"]:
            mismatch.append(i)
        if errors[i]["gold"] != p["gold"]:
            overrides[i] = p["gold"]
    if mismatch:
        raise SystemExit(f"frozen-100 reconstruction mismatch on {mismatch[:5]} (n={len(mismatch)})")

    frozen_set = set(frozen_idx)
    blocked_sentences = {err_sent[i] for i in frozen_idx}

    # --- remaining pool, split by sentence ----------------------------------
    rng = random.Random(args.seed)
    free_sentences = sorted(set(range(len(sent_ids))) - blocked_sentences)
    rng.shuffle(free_sentences)

    by_sentence: dict[int, list[int]] = {}
    for i, s in enumerate(err_sent):
        if i in frozen_set:
            continue
        by_sentence.setdefault(s, []).append(i)

    def take(n_errors: int, cursor: int) -> tuple[list[int], int]:
        picked: list[int] = []
        while cursor < len(free_sentences) and len(picked) < n_errors:
            picked.extend(by_sentence.get(free_sentences[cursor], []))
            cursor += 1
        return picked, cursor

    cursor = 0
    test_idx, cursor = take(args.test_size, cursor)
    dev_idx, cursor = take(args.dev_size, cursor)
    val_idx, cursor = take(args.val_size, cursor)
    train_idx: list[int] = []
    while cursor < len(free_sentences):
        train_idx.extend(by_sentence.get(free_sentences[cursor], []))
        cursor += 1
    if args.max_train:
        train_idx = train_idx[: args.max_train]

    splits = {
        "frozen_100": frozen_idx,
        "test": test_idx,
        "dev": dev_idx,
        "val": val_idx,
        "train": train_idx,
    }

    # --- leak checks --------------------------------------------------------
    seen_err: dict[int, str] = {}
    seen_sent: dict[int, str] = {}
    for name, idxs in splits.items():
        for i in idxs:
            if i in seen_err:
                raise SystemExit(f"error {i} in both {seen_err[i]} and {name}")
            seen_err[i] = name
            s = err_sent[i]
            if seen_sent.setdefault(s, name) != name:
                raise SystemExit(f"sentence {s} shared by {seen_sent[s]} and {name}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, idxs in splits.items():
        rows = [
            row(i, errors[i], err_sent[i], prompt_template, overrides.get(i, errors[i]["gold"]))
            for i in idxs
        ]
        write_jsonl(args.out_dir / f"{name}.jsonl", rows)
        stats[name] = {"n_errors": len(rows), "n_sentences": len({r["sentence_index"] for r in rows})}

    meta = {
        "seed": args.seed,
        "bea_dir": str(args.bea_dir),
        "n_pairs": len(pairs),
        "n_word_errors": len(errors),
        "prompt_sha256": hashlib.sha256(prompt_template.encode()).hexdigest()[:16],
        "prompt_file": str(args.prompt),
        "frozen_100_source": str(args.m6_predictions),
        "gold_overrides": {str(k): {"bea_clean": errors[k]["gold"], "used": v} for k, v in overrides.items()},
        "splits": stats,
        "leak_check": "passed: no error index or source sentence is shared by two splits",
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
