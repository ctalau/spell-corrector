# Experiment 3: reach 75% overall correction accuracy

> Update 2026-09-09: the next run is the [frozen ModernBERT + selector pilot on a 24 GB pod](FROZEN_ENCODER_PLAN.md). It supersedes this plan's E1–E6 execution queue and L40S budget for the immediate experiment. The evaluation safeguards below remain applicable.

Status: execution plan; no new training or accuracy result is claimed here.
Based on repository commit `1047c53bb3c1ae27c48d96c17b22ab3516b969e8`.

## 1. Target and current evidence

Primary target: **at least 75.00% overall top-1 accuracy on the existing BEA-60K word-error task, using only Hunspell's first 10 suggestions**. Also report the existing 16-slot system as a secondary comparison. Preserve byte inputs, candidate order, NFC exact matching, and one model forward pass for all candidates. No Aspell candidates, gold insertion, candidate reordering, global lowercasing, or generative fallback.

The latest completed experiment is exp2, not the unfinished run described in HANDOFF.md. Sources: [results.json](bea60k/results.json), [SUMMARY.txt](SUMMARY.txt), [experiment report](EXPERIMENT.md).

| Quantity | Exp2 |
|---|---:|
| Word errors / correct predictions | 68,429 / 44,357 |
| Overall accuracy, 16 slots | 64.8219% |
| Conditional accuracy, 16 slots | 80.1362% |
| Raw Hunspell oracle@10 | 80.3446% |
| Effective oracle@16 | 80.8897% |
| Synthetic validation top-1 | 92.42% |
| Training examples / parameters | 2,081,171 / 87.35M |

For a predictor constrained to its candidate pool:

`overall = effective pool coverage × conditional top-1`

At the published raw top-10 coverage, 75% requires **93.3479% conditional accuracy**. At the effective 16-slot coverage it requires **92.7189%**. These are demanding targets, not forecasts. The strict top-10 effective ceiling must be measured after the same filtering used in inference; 93.35% is optimistic if filtering removes any gold candidates.

At the old denominator, success requires at least **51,322 correct predictions**, 6,965 more than exp2. That is about **63.3% of exp2's remaining in-pool mistakes**. Exp2's 16-slot movement matrix gives:

- Retention when Hunspell is already correct: 33,912 / 36,725 = **92.34%**.
- Rescue when gold is elsewhere in the pool: 10,445 / 18,627 = **56.07%**.

Thus blindly favouring candidate zero cannot solve the problem. Both retention and rescue matter. Illustratively, at the current 16-slot group proportions, 97% retention would still require about 84.3% rescue to hit 75%.

**Correction to the final Takeaway in EXPERIMENT.md:** an 80.89% ceiling does not put 75% out of reach. The 10.18-point target gap can be closed inside the existing pool. The 19.11% outside-pool errors prevent perfection, not 75%. Going from 10 to 16 suggestions adds only about 0.55 points of coverage; it cannot explain away the gap.

The 92.42% synthetic validation score is not evidence that deployment conditional accuracy is already near target: exp2 is roughly 12.3 points worse on the benchmark's solvable population. Prioritize generalization and measurement before another model-size increase.

## 2. Evaluation contract: implement before selecting experiments

BEA remains a final-only benchmark. Use the already published aggregate results above to define the target; do not inspect its example JSONLs for rule design, sample hard examples from it, tune on it, or rerun it after each arm. Historical results have already been observed, so describe future BEA evaluation as a frozen benchmark evaluation, not a wholly unseen test.

### Pool semantics and accounting

Implementation locations: `candidates.py`, `data_build.py`, `benchmark_bea60k.py`, and a shared evaluation helper.

- Introduce an explicit candidate-limit setting propagated through build, training manifests and inference. Retain the 16-slot tensor/vocabulary layout for checkpoint compatibility; pad unused slots.
- For the primary track, **slice raw suggestions first**: `build_pool(suggestions[:10], limit=10, max_bytes=32)`. Calling `build_pool(suggestions, limit=10)` can backfill with raw ranks beyond 10 after deduplication or length filtering and is a different experiment.
- Record raw oracle@10, effective oracle@10, effective oracle@16 and losses from detection, empty lists, length filtering, and serialization fallback. No gold-dependent inference filtering.
- Keep all eligible word errors in the overall denominator, including gold-absent, undetected and empty-pool errors. Conditional accuracy uses only effective-pool-solvable errors.
- Rename the misleading `in_top10` field in the existing benchmark: it currently means membership in the 16-slot pool. Keep explicit raw-top10 and effective-pool fields.
- Log example ID, group/document ID, gold index, prediction, correctness, candidate count, truncation and fallback flags. Retain full predictions locally; commit aggregate reports only.
- Evaluate existing 16-slot weights with a top-10-only input as a diagnostic, then train the primary models with the same top-10 policy. Masking output logits alone is insufficient: extra candidates must not remain in the encoded input.
- Keep the current word-error extraction and NFC exact-match denominator unchanged. Do not claim sentence correction, detection accuracy, or multiword editing success.

Acceptance tests: gold at raw rank 11 cannot enter top10 after filtering; padded slots cannot win; raw/effective ceilings differ correctly on long/duplicate suggestions; overall equals coverage × conditional; undetected errors remain in the denominator; single and batched inference agree.

### Development splits

The current builder uses the **same typo table** for official WikiText train and validation. Article separation is useful but does not test unseen typo-to-gold mappings. This is a generalization risk to measure, not proof of leakage from BEA.

Create and hash the following splits before the sweep:

1. **D-in:** preserve the existing article-separated synthetic validation for historical comparability.
2. **D-pair:** new synthetic validation from held-out articles with disjoint NFC `(typo, gold)` pairs. Deterministically assign pair hashes to 90% train / 10% held-out pools, then generate each split only from its permitted pairs. Report overlap assertions and actual sizes; do not silently reuse pairs to hit a target.
3. **D-real:** independently collected authentic contextual spelling errors, with no BEA or source-document overlap. Collect at least 2,000 usable word errors as an initial target; split by author/document into development and sealed test partitions before any tuning. Record provenance, licenses, extraction rules and overlap checks. Human transcription/drafting errors with original context are acceptable. The vendored Wikipedia spelling list alone has no authentic context and is already used for calibration, so it is not an independent test.
4. **D-stress:** fixed D-pair variants with clean/noisy context, no context and mismatched context, plus slices by edit distance, gold rank (0 / 1–3 / 4–9), word frequency, capitalization, candidate count and truncation. These are diagnostics, not additional independent evidence.

If D-real cannot be assembled, continue low-cost D-pair experiments but explicitly mark real-world transfer unvalidated. Do not substitute BEA as development data or infer 75% readiness from D-in alone.

Training may discard unsolvable examples for ranking loss, but D-pair and D-real must also retain an unfiltered error inventory for coverage and overall scoring. Current validation parquet contains only solvable examples; `evaluate_validation.py` reports conditional accuracy even though its output is simply named `acc_top1`.

Use micro overall accuracy on D-real as the primary selection metric once available, with D-pair conditional accuracy as a guardrail. Before D-real exists, selection is provisional on D-pair conditional accuracy. Always report retention, rescue and sample counts. Bootstrap paired differences by source document (2,000 resamples, fixed analysis seed); repeated corruptions from one document are not independent observations.

## 3. Ordered experiment queue

Run one change at a time against the indicated control. All thresholds below are predeclared engineering decisions, not measured improvements. Keep the data seed at 1337. First-screen training uses seed 1337; confirm only finalists with training seeds 1338 and 1339, documenting this extension to the repository's single-seed convention.

| ID | Hypothesis / change | Control and fixed budget | Advance / stop |
|---|---|---|---|
| E0 | Reproduce exp2 validation and establish strict-top10 baseline | Existing weights if available; otherwise retrain current 87M configuration. Separate 16-slot reproduction from top10 training | Resolve metric/pool mismatches before comparing models |
| E1 | Actual rank balancing improves rescue without excessive damage | E0 top10; replace target-count cap with deterministic stratified sampling from produced rows. Compare natural distribution, 65% and 50% slot-0 training share; identical fixed validation | Advance best arm with >=1 pp primary development gain and <=1 pp D-pair regression |
| E2 | Generalization needs diverse errors, not repetition of easy pairs | E1 winner; mix 25% fresh hard examples with 75% ordinary examples, keeping total updates fixed | Require >=1 pp primary gain and improvement on D-pair nonzero-rank accuracy |
| E3 | Stronger context representation closes the transfer gap | Best data arm; compare current auxiliary MLM against dedicated clean-corpus byte MLM initialization followed by identical ranking training | Require >=1 pp primary gain; report total compute, including pretraining |
| E4 | Typo/candidate spelling evidence complements context | E3 or best earlier winner; add train-only edit-channel log probability and log word-frequency features to the scoring head | Keep only if >=0.5 pp primary gain; ablate each feature |
| E5 | More optimization helps only after data fixes | Best arm at 2 vs 4 epochs with retuned schedule; no architecture change | Stop scaling if added compute yields <0.5 pp gain |
| E6 | Confirm a single deployable winner | Best arm and its immediate control, three training seeds total | Require positive mean primary gain and paired 95% interval above zero for the preselected seed-1337 models; report all seeds |

**E1 implementation.** In `iter_examples`, `gold0_budget = target * gold0_fraction` fails to enforce the requested share when production stops short. Build eligible strata first, then sample/weight against actual counts. Do not balance validation to the training ratio. Log realized shares and unique pairs; use each available pair broadly before increasing repetition. If hard rows run out, reduce the dataset size and disclose it rather than silently filling it with easy rows. Compare equal optimizer updates and report unique examples seen.

**E2 implementation.** Expand corruption diversity and context coverage using train-side evidence. Preserve existing ED1/ED2/ED3 and phonetic mechanisms; avoid merely regenerating the same table. Mine model mistakes only from a held-out portion of training data, generate fresh contexts/typos for those error families, and verify gold naturally occurs in the allowed Hunspell pool. Cap repeated use per pair. Compare independent augmentation seeds; retain the existing 0.25 context-noise probability initially. A second, separate ablation may compare 0 and 0.50; do not combine it with the first hardness test. Keep natural easy cases to protect retention. Use D-stress to test whether gains survive unseen pairs and noisy context.

**E3 implementation.** Keep the custom byte encoder and single-pass deployment model. Add a clean-text pretraining task and a model-weight initialization option; neither a dedicated pretraining command nor a resume/init flag exists today. Pretrain on the allowed training corpus only, using contiguous masked byte spans, then initialize the reranker backbone and finetune on the identical ranking dataset. Start with 25M training bytes; expand to 100M only if the first arm passes the gate. Compare (a) current concurrent MLM, (b) no MLM, and (c) dedicated pretraining, with ranking updates matched and total GPU time disclosed. Validate weight-loading coverage and distinguish fresh finetuning optimizer state from true resumption. No tokenized pretrained encoder is assumed.

**E4 implementation.** Estimate edit-channel statistics and frequencies strictly from training sources, smooth unseen edits, and provide finite missing-value defaults. Concatenate these cheap features into `ScoringHead`, preserving candidate order. Evaluate context-only/typo-only diagnostic ablations on development data to determine which evidence the model uses. These ablations are separate experiments and must not expose the gold word in the context.

Do not run a combinatorial grid. Carry only the best passing arm forward. If nothing through E3 materially improves D-real, stop and revisit data/domain mismatch; a larger randomly initialized encoder is not the default next step. A miss is useful evidence, not a reason to change the task.

## 4. First execution: commands supported by the current repo

This is a plan commit, not an implementation of E1–E6. The following baseline commands use existing interfaces. The strict-top10 switch, new splits, richer evaluator and pretraining task described above must be implemented before the primary sweep.

Use the installation instructions in [README](../README.md), including the Python-version-specific Hunspell setup. Run from the repository root on your GPU machine. Fetch LFS artifacts if available; pointer files are not model weights or parquet data.

```bash
git pull --ff-only
git lfs pull
python -m pytest tests/ -q
python scripts/preflight.py configs/train_full.yaml --device cuda
```

Check `artifacts/model/model.safetensors`, `config.json`, and dataset manifests. If the exp2 checkpoint and exact hashed validation data exist:

```bash
mkdir -p reports/exp3/baseline
python scripts/evaluate_validation.py \
  --config configs/train_full.yaml --model artifacts/model \
  --device cuda --batch-size 32 \
  > reports/exp3/baseline/validation.json
```

If data are absent, generate a separately named baseline dataset. This is a reconstruction, not guaranteed byte-identical reproduction: typo-table sharding currently depends on worker count. Record the chosen count, corpus and dictionary hashes.

```bash
python scripts/download_sources.py
python scripts/build_training_data.py \
  --out-dir data/processed/exp3-baseline \
  --target-train 3000000 --target-valid 60000 \
  --workers 8 --seed 1337
```

Generate an isolated baseline training config only when retraining is needed:

```bash
python - <<'PY'
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("configs/train_full.yaml").read_text())
cfg["data"]["train"] = "data/processed/exp3-baseline/train.parquet"
cfg["data"]["validation"] = "data/processed/exp3-baseline/validation.parquet"
run = Path("artifacts/exp3/e0")
if run.exists():
    raise SystemExit("Refusing to overwrite an existing e0 run; choose a new run ID.")
cfg["output"] = {
    "dir": str(run),
    "metrics_jsonl": str(run / "train_metrics.jsonl"),
    "summary_json": str(run / "train_summary.json"),
    "model_dir": str(run / "model"),
}
Path("configs/train_exp3_e0.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
python scripts/preflight.py configs/train_exp3_e0.yaml --device cuda
python scripts/train.py --config configs/train_exp3_e0.yaml --device cuda
python scripts/evaluate_validation.py \
  --config configs/train_exp3_e0.yaml --model artifacts/exp3/e0/model \
  --device cuda --batch-size 32
```

Current trainer caveats to fix before the sweep:

- It reads dataset hashes from the default `data/processed/manifest.json` even with custom data paths. Record actual input-file hashes separately for E0; make manifest lookup follow configured data paths for later arms.
- It writes `reports/training_loss.png` globally. Archive the existing plot before training; add a per-run plot path before multiple runs.
- Ensure the final optimizer step is evaluated: current code can reuse the last periodic validation metrics and miss the final checkpoint's validation. Test with a run length not divisible by `eval_every`.
- Add full optimizer/scheduler/RNG state only if true resume is needed. Current weight files alone are not resumable training checkpoints.

Do **not** run `scripts/runpod/run_experiment.sh` or the existing self-driving launcher for each sweep arm unchanged: their normal flow downloads and evaluates BEA automatically. For the sweep, add a development-only entrypoint that stops after development evaluation, or execute the explicit commands above on an already provisioned GPU. This plan does not launch or purchase compute.

## 5. Implementation checklist and resource gates

Before starting E1, commit these small implementation units in order:

- [ ] Shared candidate policy and correct raw/effective metrics, with the edge-case tests from section 2.
- [ ] Pair-disjoint split manifests and development prediction exports; assert document/pair isolation.
- [ ] Rank balancing against actual rows, with an undersupply test.
- [ ] Worker-independent typo generation seeds/shards; test workers=1 and workers=2 produce identical content.
- [ ] Per-run output paths, correct input hashes, final-step checkpoint evaluation, and development-only GPU entrypoint.
- [ ] D-real provenance and frozen development/test IDs; benchmark paths excluded from all training/mining code.
- [ ] Configs for each admitted experiment; unsupported pretraining/features added only when their turn arrives.

Keep 87M, 448 bytes, bf16, effective batch 512, and current optimizer settings fixed for the initial comparisons. Preflight every changed config on the actual GPU at maximum sequence length. If necessary use microbatch 64 / accumulation 8; do not change effective batch to hide OOM.

Screen E1/E2 arms at 1,000 optimizer updates on a fixed training subset, with `epochs` large enough to supply all requested steps (`max_steps` is only a cap). Compare fixed examples/updates and log throughput. Run no more than two full finalists per stage. Small pilot gains are admission signals, not final results.

Exp2's recorded reference is 3.11 training hours / 3.74 total pod hours on an L40S; historical price was $0.79/hour. These are past observations, not a current quote or a guaranteed cost. Start with a **12 GPU-hour cap** for baseline and data pilots; review results before pretraining or seed confirmation, which need a separately estimated budget. Measure pilot throughput to estimate remaining time. Stop failed/nonfinite runs promptly. Fetch checkpoints, manifests and logs before terminating the specific experiment pod; never terminate unrelated pods.

## 5a. Pod specification and cost estimate

Estimate dated **2026-09-09**, in USD. **Budget $15 for the first stage; approximately $60–$115 for the full gated campaign on Community Cloud. Set aside $120 if proceeding through all stages.** This buys experiments, not a guarantee of reaching 75%. Stop early if the development results fail the gates.

### Use this pod

| Setting | Recommendation |
|---|---|
| Provider / product | Runpod GPU Pod, Community Cloud, on-demand/non-interruptible |
| GPU | **1 × NVIDIA L40S, 48 GB VRAM**; exact GPU ID `NVIDIA L40S` |
| CPU / host RAM | At least 16 allocated vCPUs and 64 GB RAM; prefer 24–28 vCPUs and 94 GB+ RAM at the same GPU price |
| Disk | 100 GB container disk; no attached persistent/network volume initially |
| Runtime | Python 3.11, PyTorch 2.4.1 + CUDA 12.4 (`cu124`), matching the successful exp2 runtime; use a compatible image or install per README |
| Training | 87M model, bf16, 448-byte maximum; microbatch 128 × accumulation 4 = effective batch 512 |
| Memory fallback | Microbatch 64 × accumulation 8, only after measuring preflight; keep effective batch unchanged |
| CPU workers | Start data generation at 8 workers; retain that count until worker-independent seeding is implemented |
| Lifecycle | Provision for a prepared batch of runs, copy results out, then terminate; do implementation and analysis with the GPU off |

This choice uses the same GPU family on which exp2 completed in 3.11 training hours with about 32.08 GiB observed peak GPU memory. A 24 GB 4090 does not fit that measured configuration unchanged. There is no measured reason yet to pay for multiple GPUs or an H100. Host CPU allocation varies, so slower data preparation is possible even with the same GPU.

If Community L40S is unavailable, select **1 × L40S on Secure Cloud**, keeping the rest of the settings. Budget approximately **$85–$155** for the same campaign there, or set aside $160. Do not silently substitute another GPU and assume the timing estimate remains valid.

Runpod's [L40S page](https://www.runpod.io/gpu-models/l40s) lists $0.79/hour Community and $1.09/hour Secure as checked on the estimate date. These are published reference prices, not reserved capacity or a live pod quote; check the offered CPU/RAM and total rate in the console before deploying.

### Provisioning details that matter

For now, prefer the Runpod console: select the settings above and open its terminal/SSH connection, clone the repo at the intended experiment commit, install the documented runtime, and execute section 4. Do not configure an automatic full-benchmark startup command.

The current `scripts/runpod/launch.py` has three traps:

- Its default GPU preference starts with 24 GB cards. An explicit `--gpu "NVIDIA L40S"` is required when using it.
- Its `--max-price` argument is **not enforced**: it is printed but is neither sent as a price restriction nor checked before/after creation. Treat it as ineffective until implemented; verify the console quote instead.
- `--idle` already exists and suppresses the automatic experiment entrypoint, but then cloning, setup and running commands are manual. Without it the launcher runs the old full experiment, including BEA. It also hardcodes Community Cloud and does not constrain CPU/RAM.

The launcher's default CUDA 12.8/PyTorch 2.8 image differs from exp2's successful CUDA 12.4/PyTorch 2.4.1 runtime. Pin and record the selected image tag/digest and installed versions. Check `torch.cuda.is_available()` and run GPU preflight before building data. Do not rely solely on `nvidia-smi`: a driver/runtime mismatch can leave PyTorch unable to use CUDA.

### How the estimate is calculated

Observed exp2 throughput: 8,130 updates / 3.11 training hours. At similar throughput, 1,000 updates take about **23 minutes**; allow **0.5–0.75 pod hours** per pilot for validation and overhead. One full two-epoch run takes about **3–4 pod hours**, costing roughly **$2.40–$3.25** on Community before contingency. Four epochs roughly double the training portion. These estimates assume reused data and comparable validation frequency.

The following is an allocation for a bounded, sequential campaign, not a requirement to spend each allowance. It includes setup, data builds and evaluation while the pod is running.

| Stage | Assumed scope | Pod hours |
|---|---|---:|
| E0 + first E1/E2 screen | Baseline recovery/retraining as needed, initial short pilots; stop at this checkpoint | 12 |
| Remaining E1/E2 | Additional pilots and admitted full data finalists | 12–22 |
| E3 | Masked-byte pretraining pilot(s), ranking finetunes and controls | 12–24 |
| E4/E5 | Feature ablations and one longer-training comparison | 8–20 |
| E6 | Four additional seed runs: winner/control × two new seeds; 2–4 epochs each | 13–26 |
| Freeze and final evaluation | Independent test, final BEA, artifact transfer | 3–6 |
| **Total if all stages proceed** | Reuse already trained controls; no exhaustive grid | **60–110** |

Pretraining throughput has not been measured; the E3 row is a time allowance, not a prediction derived from corpus bytes. After its first pilot, estimate runtime from measured bytes/second and reduce scope or revise the budget before exceeding the allowance. Likewise, the existing 12-hour first-stage cap may defer extra baseline reruns to the next stage.

[Runpod storage pricing](https://docs.runpod.io/pods/pricing) lists container storage at $0.10/GB/month, billed per second. Using a 720-hour month for estimation, 100 GB adds about $0.014/hour. Confirm whether the console's total already includes storage to avoid double-counting.

`estimated cost = pod hours × (GPU hourly rate + storage hourly rate) × 1.25`

The 25% contingency covers modest failures, slower CPU setup and idle time:

| Scope | Community at $0.79/h | Secure at $1.09/h |
|---|---:|---:|
| First 12 hours, including storage + contingency | about $12.06; budget **$15** | about $16.56; budget **$20** |
| Full 60–110 hours, including storage + contingency | about $60.29–$110.53; budget **$60–$115** | about $82.79–$151.78; budget **$85–$155** |

Taxes, paid dataset licenses/annotation, developer time, external backup storage and any external transfer charges are excluded. The authentic contextual development set is the largest unpriced dependency: collecting/reviewing it and implementing the plan may cost substantially more than GPU rental. No paid labeling or model API is assumed.

These are manual spending gates, not an implemented automatic billing stop. Record actual pod rate and creation/termination timestamps in the run ledger. Do not leave the GPU running between workdays. Download and verify artifacts before stopping or terminating: the selected container disk is temporary.

For terminology and the reasoning behind each experiment, see [the developer explainer](EXPERIMENT3_DEVELOPER_GUIDE.md).

## 6. Freeze, benchmark once, and decide

Before accessing BEA again, commit the winning config, checkpoint hash, candidate policy, selection rule, data manifests and completed development table. Use seed 1337 as the primary reporting checkpoint; the other seeds measure variance rather than provide three attempts at the benchmark. Evaluate the sealed independent test only after selection.

Readiness signal: approximately **94% conditional accuracy on D-real** with retention/rescue improvements and no material D-pair regression. This is a planning signal, not a guarantee of 75% on BEA. If the signal is missed, report that and decide whether to end the experiment before paying for further scaling.

After the candidate-policy implementation, the existing final benchmark command shape remains:

```bash
python scripts/download_bea60k.py
python scripts/benchmark_bea60k.py \
  --model artifacts/exp3/final/model \
  --output reports/exp3/final-bea60k --device cuda --batch-size 32
```

The finalized checkpoint directory must include the saved candidate policy, and the benchmark must enforce it. The unmodified benchmark is **16-slot only**; do not run it unchanged and label the result top10. Predeclare a secondary 16-slot comparison before the final evaluation, if desired, and never select between models using the final numbers.

Record this table for every development arm:

| Run / code SHA / checkpoint hash | Data hashes / seed | Pool | Updates / GPU h | D-in conditional | D-pair conditional | D-real overall / conditional / coverage | Retention / rescue | Paired CI vs control | Decision |
|---|---|---|---|---|---|---|---|---|---|
| E0 | pending | 10 primary; 16 reference | pending | pending | pending | pending | pending | — | baseline |

Final report must state raw correct/total counts, overall accuracy, effective coverage, conditional accuracy, retention/rescue, confidence interval, runtime, fallback counts and the residual split (outside pool versus wrong ranking). Success means **overall >= 0.75 without rounding up** under the predeclared primary policy. If short, report the measured number and the exact remaining in-pool gap; do not widen the pool, change the denominator, or tune on the final errors.
