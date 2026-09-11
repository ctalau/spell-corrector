# Experiment 3 — the 75% campaign (E0-E6)

| | |
|---|---|
| **Status** | **planned, never run — superseded** |
| **When** | Written 2026-09-09 against commit `1047c53bb3c1ae27c48d96c17b22ab3516b969e8`. Superseded the same day by the [frozen-encoder pilot](../04-frozen-encoder/PLAN.md), which took the immediate budget. |
| **Headline result** | None. No training was performed and no accuracy is claimed. |
| **Cost** | $0 spent. The plan *budgets* $15 for the first stage and ~$60-115 for the full gated campaign on Community Cloud (~$85-155 on Secure Cloud); 60-110 pod hours total. |
| **What it settled** | Two things that outlived the plan itself. (1) **The arithmetic correction to experiment 2's takeaway**: an 80.89% pool ceiling does *not* put 75% out of reach — the 10.18-point gap can be closed inside the existing pool; the 19.11% outside-pool errors prevent perfection, not 75%. (2) **The evaluation contract** — slice raw suggestions before filtering, keep every eligible word error in the overall denominator, rename the misleading `in_top10` field, report retention *and* rescue. Exp2's movement matrix: retention 33,912/36,725 = 92.34%, rescue 10,445/18,627 = 56.07%. |
| **What it left open** | Everything it proposed: strict-top-10 policy plumbing, the D-in / D-pair / D-real / D-stress splits, rank balancing against produced rows, worker-independent typo sharding, dedicated byte-MLM pretraining. None of it is implemented. The D-real set (≥2,000 authentic contextual errors) is named as the largest unpriced dependency and does not exist. |
| **Documents** | [PLAN.md](PLAN.md) — the execution plan, evaluation contract, E0-E6 queue, pod spec and cost model. [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) — the concepts primer (coverage vs conditional vs overall, retention/rescue, how to read a comparison statistically). Read the guide first if the metric vocabulary in this repo is unfamiliar. |

> **Why it is filed as an experiment rather than deleted.** Its evaluation
> contract and its correction of experiment 2's conclusion are load-bearing for
> everything after it, and its cost model is the only worked budget in the repo.
> The E0-E6 queue itself is dormant, not cancelled: nothing has replaced it as
> the path to 75% with a *trained* reranker.
