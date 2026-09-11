PROJECT: Tiny contextual spelling reranker — ~28M params — BEA-60K experiment

GOAL
====
Build, train, and evaluate a very small contextual spelling model that reranks
the top 10 suggestions produced by Hunspell.

The objective of this first experiment is simple:

    Beat Aspell's top-1 spelling correction accuracy on BEA-60K.

This is an experiment, not yet a production implementation. Prefer:
- simple code
- reproducibility
- fast iteration
- explicit metrics
- no benchmark leakage
over sophisticated architecture work.

You have access to Runpod through the configured Runpod MCP / agent tooling.

IMPORTANT INFRA RULE:
- Never leave a Runpod Pod running after a sanity check or training run.
- Always download/sync required artifacts before deleting/stopping the Pod.
- Use ephemeral Pod storage, not a persistent network volume, unless absolutely
  necessary.
- Do not commit any API keys, SSH keys, tokens, or credentials.
- Use git-lfs for large training-data/model artifacts.


======================================================================
1. EXPERIMENT DEFINITION
======================================================================

We are NOT training a generative spelling model.

The pipeline is:

    sentence containing typo
            |
            v
        Hunspell
            |
            v
     top 10 suggestions
            |
            v
    ~28M neural contextual reranker
            |
            v
       choose exactly one

At training time, an example is usable only when:

1. Hunspell identifies the token as misspelled.
2. Hunspell returns suggestions.
3. The known correct spelling occurs somewhere in the first 10 suggestions.

The model's classification label is therefore an integer 0..9.

Preserve Hunspell suggestion order.

Do NOT randomly reorder candidates. Candidate rank is useful information and
the production system will have access to it.

The model should still examine:
- the typo
- surrounding context
- every candidate's spelling
- Hunspell candidate rank


======================================================================
2. HUNSPELL CONFIGURATION
======================================================================

Use Hunspell with a fixed English US dictionary for this experiment.

Record exactly:
- hunspell version
- dictionary package/version
- dictionary file hashes
- OS/package versions

Save this metadata in:

    artifacts/hunspell_metadata.json

Candidate generation must be deterministic.

For each typo:

    suggestions = hunspell.suggest(typo)[:10]

Keep the original suggestion order.

If fewer than 10 candidates exist:
- retain the available candidates
- pad the remainder in the neural input
- mask padded candidates from the softmax

Do not invent filler words.

Normalize Unicode using NFC before comparisons.

Use exact string equality after NFC normalization when determining whether
a candidate matches the gold correction.

Do not lowercase everything globally.

For this first English benchmark, preserve casing.


======================================================================
3. MODEL ARCHITECTURE
======================================================================

Implement a custom BYTE-LEVEL bidirectional Transformer encoder.

Do NOT use BPE, SentencePiece, WordPiece, etc.

Vocabulary:
- byte values 0..255
- PAD
- CLS
- LANG_EN
- CTX_START
- CTX_END
- TYPO_START
- TYPO_END
- CAND_0 through CAND_9
- CAND_END

Approximately 274 tokens total.

Architecture:

    layers             = 8
    d_model            = 512
    attention_heads    = 8
    head_dim           = 64
    FFN hidden         = 1536
    FFN                = SwiGLU
    normalization      = RMSNorm
    positional encoding= RoPE
    attention          = bidirectional
    dropout            = 0.10
    attention dropout  = 0.00 or 0.05
    max sequence bytes = 384 initially

Use PyTorch scaled_dot_product_attention where possible so CUDA can select an
efficient attention implementation.

No causal mask.

Use pre-norm Transformer blocks.

Expected Transformer parameter count:

    self-attention/layer ~= 4 * 512^2
    SwiGLU/layer         ~= 3 * 512 * 1536

    total blocks ~= 27.3M params

Then add the small scoring head described below.

Target total model size:

    approximately 27.5M - 28.2M parameters

Add a unit test/assertion that prints and verifies total trainable parameter
count.

Do not exceed ~29M without documenting why.


======================================================================
4. INPUT SERIALIZATION
======================================================================

One forward pass must score ALL Hunspell candidates.

Do NOT run the Transformer separately 10 times.

Conceptually serialize:

    [CLS]
    [LANG_EN]
    [CTX_START]
        left context
        [TYPO_START]
            typo bytes
        [TYPO_END]
        right context
    [CTX_END]

    [CAND_0] candidate_0 bytes [CAND_END]
    [CAND_1] candidate_1 bytes [CAND_END]
    ...
    [CAND_9] candidate_9 bytes [CAND_END]

Use raw UTF-8 bytes for ordinary text.

The candidate marker itself communicates the Hunspell rank.

If the full sequence exceeds max length:
- NEVER truncate candidates
- NEVER truncate the typo
- truncate context symmetrically around the typo
- prefer keeping approximately equal left/right contextual bytes
- ensure all candidate strings remain intact

Reject pathological examples where the 10 candidates themselves cannot fit
within max length.

Candidate strings longer than a reasonable limit (e.g. 40 UTF-8 bytes) can be
discarded from DATA CONSTRUCTION if needed, but do not silently truncate a
candidate into a different word.


======================================================================
5. RANKING HEAD
======================================================================

For each candidate i, obtain:

    candidate_repr_i =
        mean hidden state over candidate bytes

Obtain:

    typo_repr =
        mean hidden state over typo bytes

Obtain:

    context_repr =
        hidden state at CLS

For candidate i concatenate:

    [
        candidate_repr_i,   # 512
        typo_repr,          # 512
        context_repr        # 512
    ]

Total = 1536 dimensions.

Scoring MLP:

    Linear(1536, 320)
    SiLU or GELU
    Dropout(0.10)
    Linear(320, 1)

Apply the same scoring MLP independently to each candidate.

Result:

    logits shape = [batch, 10]

Mask nonexistent candidate slots with -inf before cross entropy.

Training loss:

    CrossEntropyLoss(logits, gold_candidate_index)

Start WITHOUT label smoothing.

The total MLP should add roughly 0.5M parameters, bringing the complete model
very close to 28M.


======================================================================
6. DATASET STRATEGY
======================================================================

BEA-60K MUST NOT BE USED FOR:
- training
- validation
- architecture decisions based on labels
- hyperparameter selection
- corruption statistics

Treat BEA-60K as a locked final benchmark.

For the first training experiment prepare roughly:

    TRAIN: 240,000 usable examples
    VALID:  20,000 usable examples

Total:

    ~260,000 examples

If constructing 260k examples is unexpectedly expensive, minimum acceptable
first full run:

    TRAIN >= 150,000
    VALID >= 10,000

But aim for 240k/20k.


----------------------------------------------------------------------
6A. CLEAN CONTEXT SOURCE
----------------------------------------------------------------------

Use a reasonably clean public English corpus.

Preferred simple choice:

    WikiText-103 raw

Record:
- exact dataset/config/version
- source URL or dataset identifier
- license
- source checksum/version if practical

Do NOT allow exact BEA benchmark text into the training corpus if an obvious
duplicate check can detect it.

Since BEA must remain locked, perform generic deduplication by normalized
sentence hash rather than examining BEA labels during model development.


----------------------------------------------------------------------
6B. SYNTHETIC TYPO GENERATION
----------------------------------------------------------------------

Most training examples can be synthetic.

Generate ONE corrupted target token per context window.

Only select source words that:
- contain alphabetic characters
- are roughly length 3..25
- Hunspell considers correctly spelled BEFORE corruption
- aren't URLs/emails/etc.

Initial corruption mixture:

    25% adjacent-key substitution
    15% random character substitution
    15% character deletion
    12% character insertion
    13% adjacent transposition
    10% duplicate character
    10% common English spelling-pattern corruption

Examples of common-pattern corruption:
    ie <-> ei
    remove one letter from doubled consonant
    duplicate a consonant
    silent-e mistakes
    common vowel substitution

Use QWERTY adjacency for keyboard substitutions.

After corruption:

    typo = corrupt(clean_word)

Run:

    hunspell.spell(typo)

Require:
    Hunspell marks it misspelled.

Then:

    candidates = hunspell.suggest(typo)[:10]

Require:
    original clean_word appears in candidates.

The gold label is its exact candidate index.

If requirements fail:
    discard example and generate another.

IMPORTANT:
This conditioning means the dataset measures the RERANKING problem rather
than candidate-generation recall.


----------------------------------------------------------------------
6C. OPTIONAL REAL-TYPO DATA
----------------------------------------------------------------------

If convenient and licensing permits redistribution in this repository, add
authentic English typo corrections from a source such as the GitHub Typo
Corpus.

Only use very high-confidence edits:
- exactly one changed lexical token
- no large rewrite
- no URL/code identifier changes
- Damerau-Levenshtein reasonably small
- typo is rejected by Hunspell
- gold correction exists in Hunspell top 10

Split authentic data by repository/source document BEFORE train/validation
splitting.

Do not let variants from the same source document appear in both.

Target mixture if enough clean real examples are available:

    70-80% synthetic
    20-30% authentic

If licensing is unclear:
    DO NOT commit redistributed real text.
    Use synthetic WikiText-derived data for this first experiment instead.

Document this decision.

Do not delay the experiment because authentic-data licensing is complicated.


======================================================================
7. PREPARED DATA FORMAT
======================================================================

Prepare data BEFORE training.

Use Parquet.

Schema approximately:

    example_id: string
    source: string
    context_before: string
    typo: string
    context_after: string
    gold: string

    cand_0: string|null
    cand_1: string|null
    ...
    cand_9: string|null

    gold_index: int8

    corruption_type: string|null
    source_document_id: string|null

Also store optionally:

    original_sentence_hash

Do not store tensors/tokenized representation.
Tokenize into bytes in the training DataLoader.

Outputs:

    data/processed/train.parquet
    data/processed/validation.parquet
    data/processed/data_stats.json
    data/processed/manifest.json

data_stats.json must include:
- number of examples
- distribution of gold_index 0..9
- corruption-type counts
- typo length distribution
- candidate count distribution
- source mixture
- average serialized byte length
- p50/p95/max serialized byte length
- discarded-example counts/reasons

Set and record RNG seed.

Suggested main seed:

    1337


======================================================================
8. TRAIN / VALID SPLIT
======================================================================

Split BEFORE generating multiple corruptions from the same sentence/document
where possible.

Avoid sentence leakage.

For WikiText:
- split by article/document where possible
- otherwise hash normalized clean sentence and make split deterministic

For any authentic corpus:
- split by repository/document/source unit

Never randomly split duplicated corruption variants of the same clean
sentence across train and validation.


======================================================================
9. TRAINING CONFIG
======================================================================

Start with:

    dtype: BF16
    epochs: 3

    optimizer: AdamW

    learning_rate: 3e-4
    betas: (0.9, 0.95)
    eps: 1e-8
    weight_decay: 0.10

    warmup: 5% total steps
    scheduler: cosine decay

    max_grad_norm: 1.0

    effective batch size: 256 examples

Try:

    microbatch: 128
    grad accumulation: 2

If OOM:
    microbatch: 64
    grad accumulation: 4

If still OOM:
    microbatch: 32
    grad accumulation: 8

Keep effective batch approximately 256.

Use:
- BF16 autocast
- torch.compile only AFTER baseline correctness is established
- efficient SDPA
- pinned-memory DataLoader
- several workers if useful

Do not spend significant time on low-level kernel optimization for this run.

Save:
- last checkpoint
- best validation-loss checkpoint
- best validation-accuracy checkpoint if different

Validation metrics:
- loss
- top-1 candidate accuracy
- top-3 candidate accuracy
- accuracy stratified by gold Hunspell index

Training metrics:
- loss by step
- learning rate
- examples/sec
- sequences/sec
- approximate bytes/sec
- GPU memory allocated/reserved
- epoch duration

Write machine-readable metrics:

    artifacts/train_metrics.jsonl
    artifacts/train_summary.json


======================================================================
10. CHECKPOINT FORMAT
======================================================================

Final portable model artifacts should include:

    artifacts/model/model.safetensors
    artifacts/model/config.json
    artifacts/model/special_tokens.json
    artifacts/model/hunspell_metadata.json
    artifacts/model/training_manifest.json

Prefer safetensors rather than pickle.

training_manifest.json should include:
- git commit
- RNG seed
- dataset hashes
- train/validation counts
- PyTorch version
- CUDA version
- GPU model
- total trainable parameters
- batch size
- number of optimizer steps
- training duration
- best validation metrics


======================================================================
11. REQUIRED UNIT / PIPELINE TESTS
======================================================================

Before touching Runpod, implement local CPU tests.

At minimum:

1. BYTE ROUNDTRIP TEST
   UTF-8 text -> byte IDs -> text is lossless.

2. INPUT SERIALIZATION TEST
   typo and all candidate spans are correctly located.

3. CANDIDATE LABEL TEST
   gold_index points to the exact target candidate.

4. PADDING TEST
   fewer than 10 candidates results in properly masked logits.

5. MODEL SHAPE TEST
   batch input produces logits [B, 10].

6. PARAMETER COUNT TEST
   model is approximately 28M parameters.

7. LOSS TEST
   cross entropy can run forward/backward.

8. OVERFIT TEST
   model can substantially overfit a tiny fixed set, e.g. 128-512 examples.

9. DETERMINISM TEST
   prepared examples generated with same seed are reproducible.

10. NO-BEA-TRAINING TEST
   BEA files are not referenced by the training-data construction code.


======================================================================
12. CHEAP RUNPOD SANITY CHECK
======================================================================

Once local tests pass:

Use Runpod MCP to find the CHEAPEST suitable currently available CUDA Pod with:

    VRAM >= 16 GB

Prefer inexpensive options such as:
- A5000 24GB
- RTX 3090 24GB
- equivalent

No need for a 4090 for sanity testing.

Use a recent PyTorch + CUDA base image.

Do not create a persistent volume.

Container disk:
    ~20-30 GB is sufficient.

Procedure:

1. Create Pod.
2. Record:
   - Pod ID
   - GPU
   - advertised hourly price
   - start timestamp
3. SSH into it.
4. Clone repo.
5. git lfs pull if required.
6. Verify:

       nvidia-smi
       python -c "import torch; print(torch.__version__); ..."
       torch.cuda.is_available()

7. Install project dependencies.
8. Run unit tests.
9. Run a tiny GPU overfit test.
10. Run a sanity training job using approximately:

       2,000 train examples
       500 validation examples
       max ~100-200 optimizer steps

11. Confirm:
    - loss decreases materially
    - validation code executes
    - no NaNs
    - checkpoint saves
    - checkpoint reloads
    - inference produces candidate indices
    - model fits in memory comfortably

12. Save sanity logs locally into:

       reports/sanity/

13. Document:
    - exact commands used
    - GPU
    - runtime
    - cost estimate
    - peak GPU memory
    - bugs found/fixed

14. STOP/DELETE THE POD.

This last step is mandatory.

Verify through Runpod tooling that the Pod no longer exists/runs.


======================================================================
13. SANITY ACCEPTANCE CRITERIA
======================================================================

Do NOT launch the full experiment until:

- all tests pass
- GPU training works
- checkpoint save/reload works
- training loss clearly decreases
- tiny-set overfit works
- validation accuracy is above random
- no NaNs/infs
- Runpod Pod lifecycle has been verified
- data pipeline has no obvious label/candidate bugs


======================================================================
14. FULL RUNPOD TRAINING RUN
======================================================================

For the real experiment choose a fast 24GB GPU.

Preferred:

    RTX 4090 24GB

if currently available at a reasonable prepaid Runpod rate.

Otherwise use:
- 3090 24GB
- A5000 24GB
- another well-supported >=24GB CUDA GPU

Since this experiment should be relatively short, optimize for completion time
rather than saving a few cents.

Procedure:

1. Ensure prepared training/validation data exists locally.
2. Commit code and data through git/git-lfs.
3. Push repo if remote access is configured.
4. Create Runpod.
5. Clone repo on Pod.
6. Pull Git LFS files.
7. Validate hashes of prepared datasets.
8. Run training using the fixed config.
9. Stream/log metrics.
10. Save best checkpoint.
11. Run final validation.
12. Save:
       metrics
       config
       model
       environment info
       GPU memory stats
       timing
13. Download all artifacts back to local machine.
14. Verify downloaded model checksum.
15. Load downloaded model locally and run one inference smoke test.
16. STOP/DELETE THE POD.
17. Verify no active Runpod GPU resources remain.

Even if training fails:
- collect logs
- download useful artifacts
- delete Pod


======================================================================
15. BEA-60K BENCHMARK
======================================================================

Run BEA-60K LOCALLY AFTER training is complete.

Do not use it on Runpod for model selection.

Pin the exact benchmark version and record its checksum.

If using the BEA-60K distribution published through NeuSpell or another public
repository, document exactly where it came from.

Inspect the dataset format carefully and make sure the evaluation corresponds
to the intended spelling-correction benchmark.

For each BEA typo example:

    context + typo + gold correction

Run Hunspell:

    flagged = not hunspell.spell(typo)
    suggestions = hunspell.suggest(typo)

Preserve full suggestion list for analysis.

Take top 10 for the neural model.


======================================================================
16. BEA-60K METRICS
======================================================================

Report ALL of the following.

A. TOTAL NUMBER OF BENCHMARK ERRORS

    N


B. HUNSPELL DETECTION RATE

Percentage where Hunspell identifies the erroneous token as misspelled.


C. HUNSPELL TOP-1 CORRECTION ACCURACY

    correct if:
        suggestions[0] == gold


D. HUNSPELL ORACLE@10

Percentage where:

    gold in suggestions[:10]

This is the theoretical upper bound of our reranker.


E. MODEL + HUNSPELL OVERALL SUCCESS RATE

Primary metric:

    number where model-selected candidate == gold
    ------------------------------------------------
              total BEA benchmark errors

If:
- Hunspell doesn't flag typo
- Hunspell produces no suggestions
- gold isn't in first 10

count it as a FAILURE for overall success rate.

This metric measures the complete system.


F. RERANKER CONDITIONAL ACCURACY

Among examples where:

    gold in Hunspell top 10

report:

    model-selected candidate == gold

This isolates neural reranking quality from candidate generation.


G. HUNSPELL TOP-1 CONDITIONAL ACCURACY

Among the same oracle@10-covered examples, report how often candidate index 0
was already correct.


H. ASPELL BASELINE

Install/use a fixed Aspell English dictionary.

Record:
- aspell version
- dictionary version

For every BEA typo obtain Aspell's ordered suggestions.

Primary Aspell metric:

    gold == Aspell first suggestion

over ALL benchmark examples.

Treat no suggestion as failure.

This should be the apples-to-apples baseline that we need to beat.

Do not rely only on a number copied from a paper/repository; run the baseline
against exactly the benchmark copy used here.


======================================================================
17. REQUIRED HUNSPELL GOLD-INDEX HISTOGRAM
======================================================================

Compute histogram of the GOLD correction's position in Hunspell's suggestions.

Bins:

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    >=10 / present later
    not present
    Hunspell did not flag typo
    Hunspell returned no suggestions

At minimum save:

    reports/bea60k/hunspell_gold_index_histogram.json
    reports/bea60k/hunspell_gold_index_histogram.csv
    reports/bea60k/hunspell_gold_index_histogram.png

Report both:
- raw counts
- percentages

This histogram is extremely important because it reveals how much of our
remaining error is due to candidate generation versus reranking.


======================================================================
18. OPTIONAL BUT HIGH-VALUE ANALYSIS
======================================================================

Also generate:

MODEL ACCURACY BY HUNSPELL GOLD INDEX

Example:

    gold at index 0:  accuracy ...
    gold at index 1:  accuracy ...
    ...
    gold at index 9:  accuracy ...

This tells us whether the model can genuinely move lower-ranked suggestions
upward.

Generate a confusion/movement matrix:

    rows    = original Hunspell gold index
    columns = model selected index

Also sample ~50 examples in categories:

1. Model fixed Hunspell top-1 mistake.
2. Model damaged a correct Hunspell top-1.
3. Both failed despite gold being in top 10.
4. Gold wasn't in top 10.
5. Context-sensitive examples where multiple candidates are plausible.

Save examples as JSONL and include representative examples in the report.


======================================================================
19. TRAINING REPORT
======================================================================

Create:

    reports/EXPERIMENT.md

It should be understandable without reading the code.

Include:

1. Goal
2. Data sources
3. Dataset construction
4. Hunspell version/dictionary
5. Number of examples
6. Model architecture
7. Exact parameter count
8. Training hyperparameters
9. GPU used
10. Runpod runtime and approximate compute cost
11. Peak VRAM
12. Training loss curve/table
13. Validation loss
14. Validation candidate accuracy
15. BEA-60K results
16. Aspell baseline
17. Hunspell top-1
18. Hunspell oracle@10
19. Model overall success rate
20. Model conditional reranker accuracy
21. Gold-index histogram
22. Example improvements/failures
23. Main conclusions
24. Recommended next experiment

Also produce:

    reports/training_loss.png
    reports/bea60k/hunspell_gold_index_histogram.png


======================================================================
20. RESULT TABLE FORMAT
======================================================================

Include something like:

| System                         | Overall correction accuracy |
|--------------------------------|-----------------------------|
| Aspell top-1                   | XX.XX%                      |
| Hunspell top-1                 | XX.XX%                      |
| Hunspell oracle@10             | XX.XX%                      |
| 28M reranker + Hunspell        | XX.XX%                      |

And:

| Metric                                  | Value |
|-----------------------------------------|-------|
| BEA errors                              | ...   |
| Hunspell detected                       | ...%  |
| Gold in Hunspell top 10                 | ...%  |
| Model accuracy when gold in top 10      | ...%  |
| Model overall accuracy                  | ...%  |
| Aspell overall top-1 accuracy           | ...%  |

Explicitly state:

    DID WE BEAT ASPELL? YES / NO

and the absolute percentage-point difference.


======================================================================
21. REPOSITORY STRUCTURE
======================================================================

Target roughly:

    spelling-reranker/
    ├── README.md
    ├── pyproject.toml
    ├── requirements.lock               # or uv.lock
    ├── .gitignore
    ├── .gitattributes                   # Git LFS
    │
    ├── configs/
    │   ├── model_28m.yaml
    │   ├── train_full.yaml
    │   └── train_sanity.yaml
    │
    ├── spelling_reranker/
    │   ├── __init__.py
    │   ├── model.py
    │   ├── byte_encoding.py
    │   ├── hunspell.py
    │   ├── serialization.py
    │   ├── dataset.py
    │   └── inference.py
    │
    ├── scripts/
    │   ├── download_sources.py
    │   ├── build_training_data.py
    │   ├── train.py
    │   ├── evaluate_validation.py
    │   ├── download_bea60k.py
    │   ├── benchmark_bea60k.py
    │   ├── benchmark_aspell.py
    │   └── run_sanity.sh
    │
    ├── tests/
    │   ├── test_byte_encoding.py
    │   ├── test_serialization.py
    │   ├── test_dataset.py
    │   ├── test_model.py
    │   └── test_tiny_overfit.py
    │
    ├── data/
    │   ├── processed/
    │   │   ├── train.parquet
    │   │   ├── validation.parquet
    │   │   ├── data_stats.json
    │   │   └── manifest.json
    │   └── README.md
    │
    ├── artifacts/
    │   └── model/
    │       ├── model.safetensors
    │       ├── config.json
    │       ├── special_tokens.json
    │       ├── hunspell_metadata.json
    │       └── training_manifest.json
    │
    └── reports/
        ├── EXPERIMENT.md
        ├── training_loss.png
        ├── sanity/
        └── bea60k/
            ├── results.json
            ├── predictions.jsonl
            ├── hunspell_gold_index_histogram.csv
            ├── hunspell_gold_index_histogram.json
            └── hunspell_gold_index_histogram.png


======================================================================
22. GIT / GIT-LFS
======================================================================

Training data and trained model must be committed to the repository as
requested.

Use Git LFS for at least:

    *.parquet
    *.safetensors
    *.pt
    *.bin

Example:

    git lfs track "*.parquet"
    git lfs track "*.safetensors"

Commit .gitattributes.

Before committing training data:
- verify redistribution terms of the source dataset
- include source/license information in data/README.md

If any optional authentic-data source cannot legally be redistributed:
- do not commit copyrighted/restricted derived text
- omit that source from this first experiment
- use the redistributable synthetic-data route instead

Prefer a legally clean experiment over squeezing in one more corpus.


Suggested commits:

1.
    "Implement 28M byte-level spelling reranker"

2.
    "Add reproducible spelling training dataset"

3.
    "Document Runpod sanity run"

4.
    "Add trained 28M spelling model"

5.
    "Add BEA-60K benchmark results"

Do not commit BEA-60K itself if its redistribution terms do not allow it.
In that case commit:
- downloader/preparation script
- checksum
- benchmark results
but not benchmark data.


======================================================================
23. REPRODUCIBILITY
======================================================================

Use a main seed of:

    1337

Set:
- Python random
- NumPy
- PyTorch CPU
- PyTorch CUDA

Record package versions.

Use a lockfile.

Record git SHA in every full training run.

The following command or equivalent should reproduce training:

    python scripts/train.py --config configs/train_full.yaml

The following should reproduce final benchmark:

    python scripts/benchmark_bea60k.py \
        --model artifacts/model \
        --output reports/bea60k


======================================================================
24. DO NOT OVERENGINEER THIS FIRST RUN
======================================================================

Do NOT add:
- BLT
- local attention
- convolutional blocks
- MoE
- knowledge distillation
- multilingual support
- custom CUDA kernels
- complicated hard-negative mining
- autoregressive decoding
- huge hyperparameter sweeps

until the baseline result exists.

This experiment is answering:

    "Can a straightforward ~28M contextual byte encoder materially improve
     Hunspell's candidate ranking and beat Aspell on BEA-60K?"

Get that answer first.


======================================================================
25. IF TIME REMAINS: TWO CHEAP ABLATIONS
======================================================================

Only after the primary run succeeds:

ABLATION A — RANK PRIOR

Replace CAND_0...CAND_9 with a shared CAND token.

This removes explicit Hunspell-rank information.

Compare validation accuracy.

This tells us how much the network relies on Hunspell rank versus lexical/context
evidence.


ABLATION B — NO CONTEXT

Feed typo + candidates without surrounding sentence context.

Compare BEA conditional accuracy.

This tells us the value of the contextual encoder itself.

Do NOT let these ablations delay the primary result.


======================================================================
26. FINAL RESPONSE TO USER
======================================================================

When complete, report succinctly:

- exact model parameter count
- training examples / validation examples
- training duration
- Runpod GPU
- approximate cost
- peak VRAM
- training loss:
    initial
    final
    best validation
- Aspell BEA-60K success rate
- Hunspell top-1 success rate
- Hunspell oracle@10
- 28M reranker overall success rate
- reranker conditional success rate
- whether Aspell was beaten, and by how many percentage points
- gold-candidate Hunspell-index histogram
- location of model/checkpoint
- git commit SHA
- confirmation that Runpod Pod was stopped/deleted

Highlight any pipeline/data-quality issues discovered.

Do not hide failures. If the model does not beat Aspell, preserve the run and
analyze why before proposing the next architecture.
