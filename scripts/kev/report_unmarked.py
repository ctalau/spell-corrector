#!/usr/bin/env python3
"""Stage 3 of the unmarked-markup audit: score kev on the validation rows and write the findings.

    report_unmarked.py judged_gold_sample.jsonl judged_candidates.jsonl --out reports/markup_audit

Writes summary.json (confusion matrix on the gold rows, candidate counts per rule and verdict) and findings.csv:
every candidate kev assigns to a markup element, most confident first, with file, line and the sentence.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

LABELS = ["uicontrol", "filepath", "codeph", "plain"]


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("candidates")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-confidence", type=float, default=0.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    gold = load(a.gold)
    conf = {g: collections.Counter(r["kev"] for r in gold if r["element"] == g) for g in LABELS[:3]}
    recall = {g: conf[g][g] / max(1, sum(conf[g].values())) for g in conf}
    # Recall at confidence thresholds: how much of the real markup kev still names if we only trust confident calls.
    by_thr = {}
    for t in (0.0, 0.3, 0.5, 0.7):
        by_thr[t] = {g: sum(r["kev"] == g and r["kev_confidence"] >= t for r in gold if r["element"] == g)
                     / max(1, sum(r["element"] == g for r in gold)) for g in conf}

    cands = load(a.candidates)
    table = collections.Counter((r["element"], r["rule"], r["kev"]) for r in cands)
    findings = [r for r in cands if r["kev"] != "plain" and r["kev_confidence"] >= a.min_confidence]
    findings.sort(key=lambda r: (-r["kev_confidence"], r["file"], r["line"]))
    agree = sum(r["kev"] == r["element"] for r in findings)

    summary = {
        "gold_n": len(gold),
        "gold_confusion": {g: dict(c) for g, c in conf.items()},
        "gold_recall": recall,
        "gold_recall_by_confidence": by_thr,
        "candidates_n": len(cands),
        "candidates_by_verdict": dict(collections.Counter(r["kev"] for r in cands)),
        "candidates_by_rule": {f"{e}/{rule}": {k: table[(e, rule, k)] for k in LABELS if table[(e, rule, k)]}
                               for e, rule in sorted({(r["element"], r["rule"]) for r in cands})},
        "findings_n": len(findings),
        "findings_by_element": dict(collections.Counter(r["kev"] for r in findings)),
        "findings_where_kev_agrees_with_regex": agree,
        "latency_ms_p50": sorted(r["latency_ms"] for r in cands)[len(cands) // 2] if cands else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with open(out / "findings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["suggested_element", "confidence", "text", "file", "line", "regex_rule", "regex_element", "context"])
        for r in findings:
            w.writerow([r["kev"], f"{r['kev_confidence']:.3f}", r["match"], r["file"], r["line"], r["rule"],
                        r["element"], r["context"][:400]])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
