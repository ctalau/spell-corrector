# Experiment 7 — gemma-4-E2B q4_0 on GPU via llama.cpp + DSPy prompt optimization

| | |
|---|---|
| **Status** | **running** — in flight as of 2026-09-11. No numbers yet. |
| **When** | Started 2026-09-11. Pod id, GPU type and commit: **not yet recorded** — fill in below. |
| **Headline result** | *pending* |
| **Cost** | *pending* — record pod hours × hourly rate, and remember `scripts/runpod/terminate.py --all` on every exit path. |
| **What it settles** | *pending* — the two open questions it inherits are (a) whether q4_0's 2-3 point accuracy cost measured on CPU ([experiment 6](../06-llm-judge-cpu-llamacpp/README.md)) holds on GPU, and (b) whether DSPy-optimized prompts beat the hand-written ones, in particular on the punctuation-fidelity failure that dominates sentence mode's strict score. |
| **What it leaves open** | *pending* |
| **Code** | `scripts/dspy_prompt_search.py`, `spelling_reranker/dspy_program.py`, `spelling_reranker/dev_set.py`, `scripts/benchmark_llama_server.py`, `scripts/runpod/bootstrap_gpu_llama.sh` (owned by the concurrent runs — not described here). |
| **Artifacts** | *pending* — state the output directory under `reports/` once the run writes one. |

---

## Where to write the numbers when the run finishes

1. **This file.** Replace the *pending* cells in the table above (status →
   `completed` / `abandoned`, date, pod id, GPU, commit, headline result, cost,
   what it settled, what it left open, artifact path). Then add the detail
   sections below the table, in the shape used by
   [experiment 6](../06-llm-judge-cpu-llamacpp/README.md): setup, results table,
   reading the results, reproducing.

2. **[`reports/README.md`](../../README.md)** — two places:
   - the **"Every measured system on BEA-60K"** table: add one row per measured
     configuration, with the sample size `n` stated in its own column;
   - the **experiment index** table: change this experiment's status from
     `running` to its final status and replace the headline cell.

Keep the sample size visible in every row. A run on n=100 and a run on n=68,429
do not belong in the same sentence without it.

## Benchmark rules that still apply

BEA-60K is a **locked** benchmark: never train, validate, tune or prompt-search
on it. DSPy prompt optimization must run on a development set that is not
BEA-60K — see [`spelling_reranker/dev_set.py`](../../../spelling_reranker/dev_set.py).
BEA-60K is for the final measurement only, and the number reported is whatever
it is.
