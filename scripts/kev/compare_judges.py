#!/usr/bin/env python3
"""Compare two judges of the unmarked-markup audit (kev-4b vs hosted Jev) on the same spans.

    compare_judges.py --gold-a judged_gold_sample.jsonl --cands-a judged_candidates.jsonl \
        --gold-b judged_gold_sample_jev.jsonl --cands-b judged_candidates_jev.jsonl \
        --labels reports/markup_audit/hand_labels.jsonl --out reports/markup_audit/judge_comparison.json

Every rule and threshold report_unmarked.py knows is scored for both judges, with the same gold rows and hand labels.
It also reports how often the two judges agree, per rule, on the gold rows and on every candidate, and the
hand-labelled spans where exactly one of them is right.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from report_unmarked import ELEMENTS, THRESHOLDS, group, key, load, verdict

GROUPS = ("uicontrol", "filepath", "codeph", "emphasis", "dt")


def score(gold, cands, labelled, rule, t):
    by_key = {key(r): r for r in cands}
    lab = [(by_key[k], y) for k, y in labelled]
    flagged = [(r, y) for r, y in lab if verdict(r, rule, t) != "plain"]
    needs = sum(y != "plain" for _, y in lab)
    by_group = {}
    for g in GROUPS:
        fl = [(r, y) for r, y in flagged if group(r) == g]
        by_group[g] = [sum(verdict(r, rule, t) == y for r, y in fl), len(fl)]
    return {
        "gold_recall": {e: sum(verdict(r, rule, t) == e for r in gold if r["element"] == e) /
                           max(1, sum(r["element"] == e for r in gold)) for e in ELEMENTS},
        "sample_flagged": len(flagged),
        "sample_precision": sum(verdict(r, rule, t) == y for r, y in flagged) / len(flagged) if flagged else None,
        "sample_recall": sum(verdict(r, rule, t) == y for r, y in lab if y != "plain") / max(1, needs),
        "sample_right_flagged_by_source": by_group,
        "candidates_flagged": sum(verdict(r, rule, t) != "plain" for r in {key(r): r for r in cands}.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    for s in ("a", "b"):
        ap.add_argument(f"--gold-{s}", required=True)
        ap.add_argument(f"--cands-{s}", required=True)
    ap.add_argument("--names", default="kev-4b,jev-1.13")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    na, nb = a.names.split(",")
    labelled = [(key(l), l["label"]) for l in load(a.labels)]
    judges = {na: (load(a.gold_a), load(a.cands_a)), nb: (load(a.gold_b), load(a.cands_b))}
    rules = [("choice", 0.0)] + [(k, t) for k in ("noul", "hybrid", "combined") for t in THRESHOLDS]

    result = {"rules": {}, "agreement": {}, "gold_confusion_choice": {}, "disagreements_on_labels": {}}
    for rule, t in rules:
        name = rule if rule == "choice" else f"{rule}@{t}"
        result["rules"][name] = {j: score(g, c, labelled, rule, t) for j, (g, c) in judges.items()}
        # agreement of the two judges' verdicts, over the gold rows and over the unique candidates
        ga, gb = ({key(r): verdict(r, rule, t) for r in judges[j][0]} for j in (na, nb))
        ca, cb = ({key(r): verdict(r, rule, t) for r in judges[j][1]} for j in (na, nb))
        result["agreement"][name] = {"gold": sum(ga[k] == gb[k] for k in ga) / len(ga),
                                     "candidates": sum(ca[k] == cb[k] for k in ca) / len(ca),
                                     "flagged_by_both": sum(ca[k] != "plain" and cb[k] != "plain" for k in ca),
                                     "flagged_by_only_" + na: sum(ca[k] != "plain" and cb[k] == "plain" for k in ca),
                                     "flagged_by_only_" + nb: sum(ca[k] == "plain" and cb[k] != "plain" for k in ca)}
    for j, (g, _) in judges.items():
        result["gold_confusion_choice"][j] = {e: dict(collections.Counter(r["kev"] for r in g if r["element"] == e))
                                              for e in ELEMENTS}
    # hand-labelled spans on which the two judges' plain choices differ in correctness
    ca, cb = ({key(r): r for r in judges[j][1]} for j in (na, nb))
    for k, y in labelled:
        va, vb = ca[k]["kev"], cb[k]["kev"]
        if (va == y) != (vb == y):
            result["disagreements_on_labels"].setdefault(f"only_{na if va == y else nb}_right", []).append(
                {"match": ca[k]["match"], "source": group(ca[k]), "label": y, na: va, nb: vb})
    Path(a.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'rule':14}{'judge':10}{'gold ui/fp/code':>20}{'flags':>7}{'prec':>6}{'recall':>7}{'all flags':>10}")
    for name, per in result["rules"].items():
        for j, s in per.items():
            gr = "/".join(f"{s['gold_recall'][e]:.2f}" for e in ELEMENTS)
            p = f"{s['sample_precision']:.2f}" if s["sample_precision"] is not None else "-"
            print(f"{name:14}{j:10}{gr:>20}{s['sample_flagged']:>7}{p:>6}{s['sample_recall']:>7.2f}{s['candidates_flagged']:>10}")
        ag = result["agreement"][name]
        print(f"{'':14}agree gold={ag['gold']:.2f} candidates={ag['candidates']:.2f}")
    for k, v in result["disagreements_on_labels"].items():
        print(k, len(v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
