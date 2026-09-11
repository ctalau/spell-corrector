# Experiment 1 — 28M byte-level Hunspell reranker

| | |
|---|---|
| **Status** | **completed** — superseded by [experiment 2](../02-byte-reranker-87m/README.md) |
| **When** | Date not recorded. Commit range not recorded. The founding spec is [PLAN.md](PLAN.md). |
| **Headline result** | **62.56% overall** top-1 on BEA-60K, against Aspell's 60.55%. Objective ("beat Aspell's top-1 on BEA-60K") met. Decomposition: `62.56% = 80.38% pool coverage × 77.84% conditional`. |
| **Cost** | Not recorded. |
| **What it settled** | The architecture works: a ~28M byte-level bidirectional encoder that scores all Hunspell candidates in one forward pass beats Aspell top-1. It also fixed the ceiling in place — with Hunspell as the only candidate source, pool coverage (~80.4% at 10 slots) is a hard cap on overall accuracy. |
| **What it left open** | Two defects, both diagnosed afterwards and both addressed in experiment 2: (a) the typo generator emitted **exclusively edit-distance-1 typos**, while authentic misspellings are ~73% ED1 / ~25% ED2 / ~2% ED3+ — a quarter of real errors was a shape the model had never seen, and precisely the quarter where Hunspell's own top-1 collapses (60.2% → 46.6% → 4.9% on BEA by edit distance); (b) training used clean context only, 235k examples, 3 epochs. A candidate-pooling bug also materialised a `[batch, cands, seq, dim]` tensor (~1 GB per batch), which is what drove the 22 GiB peak and capped throughput. |
| **Artifacts** | None retained under `reports/` — the committed BEA artifacts in [`reports/bea60k/`](../../bea60k/) are experiment 2's. The numbers above are the ones carried forward in [experiment 2's plan](../02-byte-reranker-87m/PLAN.md) and the root README. |

> This directory has no results file of its own. It exists so the lineage starts
> where it actually started, and so the original project charter lives next to
> the experiment it specified.

## The original charter

[PLAN.md](PLAN.md) is the original project specification (moved here from the
repository root): a ~28M-parameter contextual reranker over Hunspell's **top 10**
suggestions, with the objective "beat Aspell's top-1 spelling correction accuracy
on BEA-60K". It is **historical** — kept for provenance and because it is still
the clearest statement of the design constraints the whole line inherits (byte
vocabulary, preserved Hunspell order, one forward pass, no generative fallback,
seed 1337, no benchmark leakage, never leave a pod running). Sections 25 and 26
(two cheap ablations, and the final-report checklist) were written for
experiment 1 and were not re-run later.

Where PLAN.md and later documents disagree, the later documents win:

| PLAN.md says | Superseded by |
|---|---|
| Top 10 candidates, label 0..9 | Experiment 2 uses 16 slots (untruncated Hunspell list); the [experiment 3 plan](../03-scaling-campaign-75/PLAN.md) proposes going back to a strict raw top-10 as the primary policy |
| ~28M parameters | 87M in experiment 2 |
| Beat Aspell | Target raised to 75% overall from experiment 2 onwards |
