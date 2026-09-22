#!/usr/bin/env python3
"""Stage 3 of the unmarked-markup audit: measure kev, then write the findings.

    report_unmarked.py judged_gold_sample.jsonl judged_candidates.jsonl \
        --labels reports/markup_audit/hand_labels.jsonl --out reports/markup_audit

Two decision rules are scored:
  choice     kev's 4-way choice (uicontrol / filepath / codeph / plain); flag when the argmax is not plain.
  noul@t     three yes/no questions; flag the element with the highest p(yes) when it is >= t.
  hybrid@t   the element comes from the choice (best of the three markup options, ignoring plain); flag it when its
             yes/no p(yes) is >= t. The choice separates filepath from codeph better; the yes/no finds more UI labels.
Recall comes from the gold rows (spans the writers did mark up, shown to kev as plain text). Precision comes from the
hand-labelled sample of candidates. findings.csv uses --rule / --threshold.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

ELEMENTS = ["uicontrol", "filepath", "codeph"]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def key(r):
    return r["file"], r["line"], r["match"], r["context"]


def verdict(r, rule: str, t: float) -> str:
    if rule == "choice":
        return r["kev"]
    if rule == "hybrid":
        el = max(ELEMENTS, key=lambda e: r["kev_probs"][e])
        return el if r[f"p_{el}"] >= t else "plain"
    el = max(ELEMENTS, key=lambda e: r[f"p_{e}"])
    return el if r[f"p_{el}"] >= t else "plain"


def group(r) -> str:
    return {"emphasis": "emphasis", "dt-convention": "dt"}.get(r["rule"], r["element"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("candidates")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rule", default="hybrid", choices=["hybrid", "noul", "choice"])
    ap.add_argument("--threshold", type=float, default=0.7)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    gold, cands = load(a.gold), load(a.candidates)
    by_key = {key(r): r for r in cands}
    labelled = [(by_key[key(l)], l["label"]) for l in load(a.labels) if key(l) in by_key]
    rules = [("choice", 0.0)] + [(k, t) for k in ("noul", "hybrid") for t in THRESHOLDS]

    evaluation = {}
    for rule, t in rules:
        name = rule if rule == "choice" else f"{rule}@{t}"
        rec = {e: sum(verdict(r, rule, t) == e for r in gold if r["element"] == e) /
                  max(1, sum(r["element"] == e for r in gold)) for e in ELEMENTS}
        flagged = [(r, lab) for r, lab in labelled if verdict(r, rule, t) != "plain"]
        # A flag is right when the span needs markup *and* kev named the right element.
        exact = sum(verdict(r, rule, t) == lab for r, lab in flagged)
        needs = sum(lab != "plain" for _, lab in labelled)
        per_group = {}
        for g in ("uicontrol", "filepath", "codeph", "emphasis", "dt"):
            rows = [(r, lab) for r, lab in labelled if group(r) == g]
            fl = [(r, lab) for r, lab in rows if verdict(r, rule, t) != "plain"]
            per_group[g] = {"n": len(rows), "need_markup": sum(lab != "plain" for _, lab in rows), "flagged": len(fl),
                            "flagged_correct": sum(verdict(r, rule, t) == lab for r, lab in fl),
                            "missed": sum(lab != "plain" and verdict(r, rule, t) == "plain" for r, lab in rows)}
        evaluation[name] = {
            "gold_recall": rec,
            "sample_flagged": len(flagged),
            "sample_precision": exact / len(flagged) if flagged else None,
            "sample_recall": sum(verdict(r, rule, t) == lab for r, lab in labelled if lab != "plain") / max(1, needs),
            "sample_by_group": per_group,
            "candidates_flagged": dict(collections.Counter(v for v in (verdict(r, rule, t) for r in cands) if v != "plain")),
        }

    chosen = a.rule if a.rule == "choice" else f"{a.rule}@{a.threshold}"
    findings = [r for r in cands if verdict(r, a.rule, a.threshold) != "plain"]
    score = (lambda r: r["kev_confidence"]) if a.rule == "choice" else (lambda r: r[f"p_{verdict(r, a.rule, a.threshold)}"])
    findings.sort(key=lambda r: (verdict(r, a.rule, a.threshold), -score(r)))
    summary = {
        "gold_n": len(gold), "candidates_n": len(cands), "hand_labelled_n": len(labelled),
        "candidates_by_rule": dict(collections.Counter(f"{r['element']}/{r['rule']}" for r in cands)),
        "gold_confusion_choice": {e: dict(collections.Counter(r["kev"] for r in gold if r["element"] == e)) for e in ELEMENTS},
        "evaluation": evaluation,
        "findings_rule": chosen,
        "findings_n": len(findings),
        "findings_by_element": dict(collections.Counter(verdict(r, a.rule, a.threshold) for r in findings)),
        "findings_by_source": dict(collections.Counter(group(r) for r in findings)),
        "latency_ms_p50": sorted(r["latency_ms"] for r in cands)[len(cands) // 2],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with open(out / "findings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["suggested_element", "score", "text", "file", "line", "source", "p_uicontrol", "p_filepath",
                    "p_codeph", "choice", "context"])
        for r in findings:
            w.writerow([verdict(r, a.rule, a.threshold), f"{score(r):.3f}", r["match"], r["file"], r["line"],
                        f"{r['element']}/{r['rule']}", *(f"{r[f'p_{e}']:.3f}" for e in ELEMENTS), r["kev"],
                        r["context"][:400]])
    print(json.dumps({k: v for k, v in summary.items() if k != "evaluation"}, indent=2))
    for name, ev in evaluation.items():
        print(f"{name:10} gold_recall={ {e: round(v, 2) for e, v in ev['gold_recall'].items()} } "
              f"sample_flagged={ev['sample_flagged']} precision={ev['sample_precision'] and round(ev['sample_precision'], 2)} "
              f"sample_recall={ev['sample_recall']:.2f} flagged={ev['candidates_flagged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
