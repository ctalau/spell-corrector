# Experiment 9 — plan: put open-weights **kev** in the seat "Jev Choice" occupied

## Why

Three milestone reports quote a number this repository cannot reproduce:

| Where | Quote |
|---|---|
| `artifacts/spell_slm_m4/milestone4_direct_report.md` | "Jev Choice (approx, prior) \| ~91% \| — \| API chooser over lists" |
| `artifacts/spell_slm_m3/milestone3_finetune_report.md` | "bigger backbone (Jev already at 91% as an API chooser)" |
| `artifacts/spell_slm_m{4,5,6}/metrics_milestone*.json` | `"jev_choice_approx": 0.91` |

Jev is TypeSafe's hosted decision model: typed questions in, calibrated
probabilities out. It is closed, it was measured in a session this repository
does not contain, and the figure is carried forward marked *approx, prior*. It
is the one row in the comparison table that has never been re-run here.

[`jaredpalmer/kev`](https://github.com/jaredpalmer/kev) is an open-weights
reconstruction of that architecture — LoRA + pointer head on a Qwen backbone,
block-causal branch mask, softmax over option spans — serving the same
`/v1/systemone` contract. Weights are on the Hub (0.5B, 0.6B, 4B, 8B). So the
Jev row can be replaced by a row that actually runs on this box.

## Question

Does an open-weights typed-decision model, used as a candidate chooser, beat
Hunspell's top-1 on the frozen 100 — and how far is it from the ~91% the
hosted Jev is credited with?

## Protocol

- **Items**: the frozen 100 typos, recovered from
  `artifacts/spell_slm_m7/results/predictions_teacher_nf4_frozen_100.jsonl`
  (`error_index`, `typo`, `gold`, `sentence`). Same 100 items M3–M7 were scored on.
- **Candidates**: Hunspell suggestions via `spelling_reranker.candidates.build_pool`.
  Two conditions, both reported, neither selected on its score:
  `--limit-candidates 8` (the exp 5/6 index-mode width) and uncapped.
  Dictionary hashes match `artifacts/hunspell_metadata.json` exactly, so the
  pools are the ones earlier experiments saw.
- **The ask**: one `choice` question per typo. `state` = the sentence with the
  typo marked `<TYPO>…</TYPO>`; `criteria` = the candidate words, no
  descriptions; answer = argmax of the returned probabilities.
- **Instruction string**: one, fixed before the first run, never varied.
  BEA-60K is locked: no prompt search, no condition picked after seeing a score.
- **Scoring**: casefold, as in milestone 3. Overall = correct / 100.
  Conditional = correct / rows where gold was in the list. Hunspell top-1 on the
  same rows is the do-nothing baseline.
- **Models**: `kev-0.5b`, `kev-0.6b` (fp32), `kev-4b` (bf16), all CPU, this box.

## What would count as an answer

- kev **above** Hunspell top-1 (60%) → a typed-decision reranker is worth a row.
- kev **at or below** 60% → reranking with it is worse than not reranking, the
  verdict experiment 6 already returned for MiniCPM5-1B, and the Jev row stays
  unreproduced rather than replaced.

## Code

`scripts/kev/run_kev_choice.py`. kev is cloned read-only next to the repo and
imported; nothing is vendored.
