# Developer guide to the 75% spelling experiment

This guide is for a developer who knows basic probability, statistics and introductory machine learning. Read it alongside [the execution plan](EXPERIMENT3_75_PLAN.md). The plan specifies the experiments and commands; this document explains what they mean and how to judge the results.

**Current situation:** the latest model corrects 64.82% of benchmark word errors. We want 75%. Hunspell already proposes the correct answer in roughly 80% of cases, so the main job is to make the model choose much better among those suggestions.

## 1. What the program learns

Hunspell supplies a list of possible spellings. The model reads the misspelled word, its surrounding text and those suggestions, then chooses one list entry. It cannot invent a new correction.

As an illustrative example, suppose the text is “Please put the book over ther” and the candidates include “there”, “their” and “the”. Context should favour “there”. This is an invented example, not a claim about the installed dictionary's exact suggestion list.

In ordinary supervised-learning terms:

| ML term | Meaning in this project |
|---|---|
| Input features | Typo, left/right context, candidate strings and their order |
| Label / gold | Index of the known correct candidate; “gold” means the reference answer |
| Model output | One numerical score for every candidate |
| Prediction | The valid candidate with the highest score |
| Training example | One typo in context with candidates and a known label |
| Parameter | One learned number in the model; there are about 87 million |
| Reranking | Choosing among candidates supplied by another system |

The encoder is a bidirectional Transformer: each position can use text on both sides. It operates on UTF-8 bytes rather than a vocabulary of word pieces, which lets it represent unfamiliar spellings directly. A small final network, the scoring head, converts the encoder's representations into candidate scores. All candidates are scored together in one forward pass.

Training uses softmax probabilities and cross-entropy. For the correct candidate with predicted probability p, the example's loss is `-log(p)`. Increasing p reduces the loss. Accuracy only checks whether the correct candidate has the largest score; a lower average loss need not imply higher accuracy. Softmax outputs are not automatically reliable confidence estimates.

## 2. Three percentages with different denominators

The most important distinction is between **coverage**, **conditional accuracy**, and **overall accuracy**.

Imagine exactly 1,000 spelling errors:

- Hunspell includes the correct answer for 800: coverage is 80%.
- The model chooses correctly for 750 of those 800: conditional accuracy is 93.75%.
- It therefore fixes 750 of all 1,000 errors: overall accuracy is 75%.

This is the probability identity:

`P(correct) = P(gold available) × P(correct | gold available)`

It holds here because the model cannot be correct when the answer is unavailable. An “oracle” is an imaginary perfect chooser; oracle@10 measures what perfect choice among the allowed ten suggestions could achieve. It does not describe the trained model.

At this repo's published top-10 coverage, the calculation is:

`0.75 / 0.8034459 ≈ 0.93348`

So 75% overall needs approximately 93.35% conditional accuracy. The number could be slightly higher after removing unusable candidates. Reaching it is possible under the measured ceiling, but not assured.

“Top-1” means the first chosen answer. “Top-10 coverage” means the answer occurs anywhere in ten suggestions. A 75% top-10 score is not a 75% top-1 score.

The current validation parquet keeps only examples whose gold answer is available. Its reported accuracy is therefore conditional. Comparing its 92.42% directly with benchmark overall accuracy of 64.82% would mix denominators. The comparable benchmark conditional result is 80.14%, still roughly 12.3 percentage points worse.

A gain from 64% to 65% is **one percentage point**, or about 1.56% relative improvement. The plan's “pp” thresholds mean percentage points.

## 3. Why 10 versus 16 needs an explicit setting

The current model has 16 candidate slots. The new primary experiment uses Hunspell's first ten raw suggestions, while keeping the tensor layout compatible by padding unused slots.

Filtering order matters. Suppose one of the first ten suggestions is too long. Removing it and taking suggestion eleven as a replacement no longer tests the first ten raw suggestions. The plan therefore slices first, filters second, and separately reports raw versus usable coverage.

Similarly, removing candidates only from the final score vector is not equivalent to removing them from the input: the Transformer could still have read the extra candidate strings.

These are ordinary software-contract issues. A result is interpretable only if training, inference and evaluation implement the same contract.

## 4. Why a high validation score can disappoint

Training examples are synthetic: take correct WikiText prose and corrupt a word. Real writers make different errors, use different language, and may have additional mistakes nearby. A model can get good at the synthetic task without transferring equally well.

There is another possible shortcut. If training sees “recieve → receive” repeatedly, recognizing that pair in a new article may be easier than correcting an entirely unfamiliar typo. Article separation alone does not test the latter.

The plan creates several views:

| Dataset | Question it answers |
|---|---|
| D-in | Does the result compare with our historical synthetic validation? |
| D-pair | Does the model handle typo–gold pairs never used for training? |
| D-real development | Do improvements transfer to independent authentic errors? |
| D-real sealed test | Does the chosen design generalize beyond the real examples used for selection? |
| D-stress | What happens when context is absent, noisy or misleading? |
| BEA final benchmark | Did the frozen experiment meet the original target? |

A development set is used repeatedly to make choices. A test set is used after those choices are frozen. Looking at test errors and then changing the model makes those examples part of development in practice, even if they never enter the training file.

The existing BEA result is already known. We can acknowledge that history while avoiding further tuning on its individual examples. Its final score should not become a feedback loop for every experiment.

Synthetic validation improvement is useful evidence. It is not a guarantee of the same-sized improvement on real text.

## 5. Retention and rescue explain what changed

A reranker has two jobs:

1. Keep Hunspell's first answer when it is right: **retention**.
2. Select a later candidate when the first is wrong but gold is available: **rescue**.

Always taking candidate zero gives perfect retention but zero rescue. Aggressively switching away can improve rescue while damaging retention. Report both alongside overall accuracy.

Exp2 retained about 92.34% of already-correct first suggestions and rescued about 56.07% of the remaining solvable cases. These refer to the historical 16-slot run. They explain why both conservative and aggressive mistakes matter.

Training data contains many easy slot-zero examples. “Rank balancing” changes how often training sees each group. It does not change candidate order and does not change the evaluation population. Balanced training can improve attention to hard cases, but can also make the model switch too often; retention tells us whether that happened.

The builder's current cap is based on the requested dataset size. If it produces fewer rows, the intended proportion may not hold. Fixing this is mostly a counting and sampling task, not advanced neural-network work.

## 6. What each experiment is trying to establish

| Experiment | Plain-language question | Why this comes now |
|---|---|---|
| E0: baseline | Can we reproduce a trustworthy starting score? | Otherwise improvements may just be measurement differences |
| E1: balancing | Does practicing more nonzero-rank cases improve decisions? | Directly addresses easy-example dominance |
| E2: harder, more varied data | Does broader practice generalize beyond memorized pairs? | More repeated examples are not necessarily more information |
| E3: context pretraining | Would learning English context first help later correction? | Tests representation quality after data issues are addressed |
| E4: spelling/frequency features | Do cheap explicit clues supplement the learned representation? | Some useful evidence may be easier to supply directly |
| E5: longer training | Is the improved model still learning after two epochs? | More compute makes sense only after earlier problems are controlled |
| E6: repeat seeds | Is the gain reproducible across random initializations? | A lucky run should not decide the conclusion |

“Hard-example mining” means finding training-side cases the current model gets wrong and creating more useful practice around them. Mining validation or benchmark mistakes would compromise the evaluation.

“MLM” means masked language modeling: hide part of the input and ask the model to reconstruct it. Here the hidden units are bytes. Concurrent MLM adds that task while learning candidate selection. Dedicated pretraining learns it first, then finetunes the encoder for ranking. These are different ways to allocate learning and compute; the dedicated version is not implemented yet.

An “edit channel” estimates how plausible a typo is given a candidate spelling. A frequency feature estimates how common a candidate is in training text. Both should use training sources only, and neither should overwhelm context.

An “ablation” removes one ingredient while keeping other settings fixed. If removing an ingredient has no adverse effect, it may not be worth its complexity. Do not change data, architecture, learning rate and training duration together and then attribute a gain to one of them.

## 7. Training settings a developer needs to understand

| Setting | Practical interpretation |
|---|---|
| Epoch | One pass through the training dataset |
| Microbatch | Examples processed together in GPU memory |
| Gradient accumulation | Combine gradients from several microbatches before updating weights |
| Effective batch | Microbatch × accumulation; 128 × 4 = 512 here |
| Optimizer update / step | One actual change to the model weights |
| Learning rate | Controls the size of that change |
| bf16 | Reduced-precision arithmetic used to reduce memory and speed training |
| Seed | Controls pseudo-random initialization/sampling; record it for comparisons |
| Checkpoint | Saved model state at a particular point in training |
| OOM | Out of GPU memory; reduce microbatch and compensate with accumulation |
| Preflight | A worst-case forward/backward check before an expensive run |

An 87M-parameter model can use far more memory than its saved weight file. Training also holds intermediate activations, gradients and optimizer state. Exp2 peaked around 32 GiB, which is why the recommended pod has a 48 GB L40S.

Reducing microbatch from 128 to 64 and increasing accumulation from 4 to 8 preserves an effective batch of 512. It is a reasonable memory adjustment, though runtime and numerical details can still differ.

A weights-only checkpoint lets you run predictions or start finetuning. Exact continuation requires optimizer, scheduler and random-number state as well. Do not call a fresh optimizer run a resumed run.

## 8. Reading a comparison statistically

Compare candidate models on **the same examples**. For each example, record whether the new model fixed a previous mistake or introduced a new one. That paired comparison is more informative than comparing two percentages from unrelated samples.

Errors from one article or writer may be correlated. The plan therefore resamples whole documents in its bootstrap calculation. Each resample gives a new-model-minus-control accuracy difference; their distribution estimates uncertainty around the measured difference.

A paired 95% interval entirely above zero is evidence for a positive difference under the sampling assumptions. It is not a 95% probability that the model is better everywhere, and it does not account for every bias introduced by trying many experiments.

Sample size matters. At 75% accuracy and 1,000 independent errors, a rough binomial standard error is about 1.37 percentage points, giving a roughly ±2.7-point 95% interval for a single accuracy. Correlation can widen it. A paired difference can be more precise, depending on disagreements. Thus a 0.5-point observed gain may be worth a pilot follow-up without being conclusive.

Seeds address another uncertainty: the training process itself. Bootstrap intervals across documents do not measure variability across training initializations. Report the seed results separately rather than combining them into a misleadingly large sample.

This is why the plan uses short screens, full confirmation, and a final frozen test. A small pilot win is permission to investigate, not proof of success.

## 9. How to execute without wasting the budget

Use the [pod and cost section](EXPERIMENT3_75_PLAN.md#5a-pod-specification-and-cost-estimate) as the source of truth for current estimates. Start with the first-stage budget, not every possible experiment at once.

Prepare code, configs and data decisions before renting the GPU. Each run needs a unique output directory, code commit, dataset hashes, configuration, seed and prediction report. Copy those artifacts out before terminating the temporary pod.

The baseline commands exist today. The strict-top10 setting, new split/evaluation tooling, actual-count balancing and dedicated pretraining still require implementation. The plan clearly marks that boundary; the whole campaign is not yet a one-command script.

Operational decisions:

| Observation | Next action |
|---|---|
| CUDA unavailable or preflight fails | Fix the environment or batch size before building data |
| Nonfinite loss | Stop; investigate input validity and optimization |
| D-in improves but D-pair does not | Investigate pair memorization or synthetic shortcuts |
| Rescue improves while retention collapses | Revisit training balance and compare total accuracy |
| D-pair improves but D-real does not | Treat domain transfer as unresolved |
| Short-run gain disappears in full/seed runs | Do not advance it as a robust improvement |
| Final accuracy is 74.9% | Report 74.9%; the 75% target was not met |

The final deliverable is a reproducible model and an honest result. We are testing whether better data and learning can close the in-pool gap. The cost estimate covers that investigation; it does not establish in advance that the target will be reached.
