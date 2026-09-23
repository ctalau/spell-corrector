# Experiment 12 — hosted **Jev** instead of kev as the judge for missing DITA markup

| | |
|---|---|
| **Status** | **completed** |
| **When** | 2026-09-23, branch `claude/tender-albattani-mrsm09`. Same candidates, gold rows and hand labels as [experiment 11](../11-kev-unmarked-markup/README.md). |
| **Question** | Does TypeSafe's hosted Jev do better than open-weights `kev-4b` at deciding which spans in the user guide need `<uicontrol>`, `<filepath>` or `<codeph>`? |
| **Method** | [`judge_unmarked_jev.py`](../../../scripts/kev/judge_unmarked_jev.py) sends exactly the requests `kev-4b` answered: the same four questions, imported from `judge_unmarked.py`, on the same clipped context. They go to `typesafe/jev-1.13` through OpenRouter's `/api/v1/systemone`, which accepts kev's request shape. [`compare_judges.py`](../../../scripts/kev/compare_judges.py) scores both judges with every rule and threshold from experiment 11. |
| **Headline** | **Jev is better on every axis that matters.** Its plain 4-way choice finds **78%** of the real `<uicontrol>` spans, where kev's finds 33%. With the simplest rule, `noul@0.7`, Jev's flags are **84% right** and it catches 79% of the spans that need markup, with 4,588 flags. kev's best rule, `combined@0.8`, is 74% right and catches 78%. |
| **Cost** | **$0.44** for all 15,334 spans (900 gold, 14,434 candidates). No GPU. The p50 latency was 500 ms per request (four questions), run 12 at a time. The whole run took about 17 minutes. |
| **Output** | [`reports/markup_audit/jev/findings.csv`](../../markup_audit/jev/findings.csv), in the same format as kev's findings. Side-by-side numbers for every rule are in [`judge_comparison.json`](../../markup_audit/judge_comparison.json). |

## Results

Both judges are scored on the same spans. The provider refused 5 of the 14,439 candidates with HTTP 403 on every attempt ([see below](#refused-spans)), so 13,853 unique candidates, 900 gold rows and all 200 hand labels are compared. As in experiment 11, a flag is right only if the span needs markup *and* the judge named the right element.

| rule | judge | gold recall ui / fp / code | sample flags | **precision** | **sample recall** | unique flags |
|---|---|---:|---:|---:|---:|---:|
| `choice` | kev-4b | 0.33 / **0.91** / 0.65 | 99 | 0.75 | 0.66 | 4,644 |
| `choice` | Jev | **0.78** / 0.85 / **0.89** | 139 | 0.73 | **0.90** | 7,709 |
| `combined@0.8` (exp 11's pick) | kev-4b | 0.60 / 0.91 / 0.64 | 117 | 0.74 | 0.78 | 5,309 |
| `combined@0.8` | Jev | 0.79 / 0.85 / 0.88 | 139 | 0.73 | 0.90 | 7,714 |
| `noul@0.6` | kev-4b | 0.65 / 0.75 / 0.84 | 166 | 0.55 | 0.81 | 10,311 |
| `noul@0.6` | Jev | 0.71 / 0.89 / 0.73 | 116 | 0.82 | 0.85 | 5,642 |
| **`noul@0.7`** (chosen for Jev) | kev-4b | 0.61 / 0.73 / 0.74 | 147 | 0.59 | 0.77 | 8,320 |
| **`noul@0.7`** | **Jev** | 0.67 / 0.84 / 0.63 | 105 | **0.84** | **0.79** | **4,543** |
| `hybrid@0.7` | kev-4b | 0.48 / 0.85 / 0.72 | 129 | 0.63 | 0.72 | 7,398 |
| `hybrid@0.7` | Jev | 0.66 / 0.78 / 0.64 | 102 | 0.85 | 0.78 | 4,333 |
| `noul@0.8` | kev-4b | 0.57 / 0.69 / 0.58 | 99 | 0.66 | 0.58 | 5,551 |
| `noul@0.8` | Jev | 0.56 / 0.75 / 0.48 | 75 | **0.95** | 0.63 | 3,099 |

The table shows the most useful rows. `compare_judges.py` prints all 16 rule and threshold pairs.

### What changed

- **The choice question works with Jev.** kev's choice sent 160 of 300 real `<uicontrol>` spans to "plain", which is why experiment 11 needed the `combined` patch. Jev sends 25 there. Its choice alone already covers what `combined` was built to fix.
- **Jev's yes/no answers can be trusted.** kev's `codeph` yes/no said yes to almost anything short and technical: 11.8k flags at `noul@0.5`, 51% right. Jev's yes/no answers flag fewer spans *and* catch more of the right ones. At every threshold from 0.5 to 0.8, Jev's `noul` rule is 18–29 points more precise than kev's.
- **The two judges want different rules.** For kev, the choice is precise and the yes/no answers are noisy. For Jev it is the other way round: its choice over-calls UI and code (73% precision), while its yes/no answers at 0.7 are clean. So the best rule for Jev is simply "flag the element with the highest yes/no probability, if it is ≥ 0.7".
- **Jev has one weakness: it takes headings for UI labels.** Of the 21 hand-labelled spans where only kev's choice was right, 9 are `<dt>` terms that name a section ("Drag and Drop Mechanism", "Contextual Menu Actions"). Jev's choice calls them `uicontrol`. Five more are emphasised prose ("functions", "rowspans", "AND/OR") that it calls code. Of the 35 spans where only Jev's choice was right, 16 are real UI labels that kev called plain ("Author", "Help > Install", "Copy as Markdown") and 9 are command-line flags and editor variables that kev missed or mis-typed (`-gp`, `type="surround"`, `${ask('Hello world!')}`).
- **The two judges often disagree.** With each judge's own recommended rule, both flag 3,933 candidates. Jev flags 610 that kev does not, and kev flags 4,387 that Jev does not, mostly `codeph` from the emphasis source.

### Jev's findings (`noul@0.7`)

- **4,588 flags**: 1,181 `uicontrol`, 400 `filepath` and 3,007 `codeph`.
- By source: 2,518 emphasis, 1,531 codeph regexes, 287 UI regexes, 150 filepath regexes and 102 `<dt>`.
- Precision by source on the sample (right / flagged): codeph 26/28, filepath 29/31, `<dt>` 18/24, UI regexes 9/14, emphasis 6/8. kev's `combined@0.8` scores 29/32, 28/36, 17/24, 7/14 and 6/11 on the same sources.
- The files with the most flags are the same reference tables as in experiment 11: the WebHelp parameter reusables (194), `dcpp_parameters.dita` (166), `reusables-editor-variables.dita` (156) and `ch_css_extensions.dita` (155).

## Refused spans

The provider's firewall returns an HTML 403 page for any request containing `c:\boot.ini` (3 candidates in `preferences-editor-spell-check.dita`) or one long Java stack trace (2 in `zendesk-transformation-output.dita`). The page looks like a Cloudflare WAF block. The error is the same every time, so the judge logs these spans and skips them. They are missing from Jev's findings, and both judges are compared without them.

## Honest caveats

- **The rule was picked on the labels it is scored on**, the same issue as in experiment 11, and the choice was among the same 16 pairs. `noul@0.7` is not the single best Jev row. `noul@0.6` and `hybrid@0.7` are within noise of it. It was picked as the simplest rule with both precision and recall at or above 0.79. With n=200 and 105 flags, the 84% carries about ±7 pp.
- **The hand labels were written for experiment 11, before any judge output, by one person.** Jev saw none of them, so they still count as blind.
- **This compares a closed service with a 4B open model.** Jev's size and training data are unknown. `typesafe/jev-1.13-20260917` is the dated revision that answered. `~typesafe/jev-latest` will move.
- **Jev's probabilities come back rounded to two decimals**, so its thresholds are coarser than kev's four-decimal ones.

## Reproduce

```bash
M=reports/markup_audit
python scripts/kev/judge_unmarked_jev.py $M/gold_sample.jsonl --out $M/jev/judged_gold_sample_jev.jsonl --workers 12
python scripts/kev/judge_unmarked_jev.py $M/candidates.jsonl  --out /tmp/judged_candidates_jev.jsonl --workers 12
zcat $M/judged_candidates.jsonl.gz > /tmp/judged_candidates_kev.jsonl
cd scripts/kev && python compare_judges.py \
    --gold-a ../../$M/judged_gold_sample.jsonl --cands-a /tmp/judged_candidates_kev.jsonl \
    --gold-b ../../$M/jev/judged_gold_sample_jev.jsonl --cands-b /tmp/judged_candidates_jev.jsonl \
    --labels ../../$M/hand_labels.jsonl --out ../../$M/judge_comparison.json && cd ../..
python scripts/kev/report_unmarked.py $M/jev/judged_gold_sample_jev.jsonl /tmp/judged_candidates_jev.jsonl \
    --labels $M/hand_labels.jsonl --rule noul --threshold 0.7 --out $M/jev
```

`judge_unmarked_jev.py` reads `OPENROUTER_API_KEY` from the environment. The judged candidates are committed compressed as `jev/judged_candidates_jev.jsonl.gz`.
