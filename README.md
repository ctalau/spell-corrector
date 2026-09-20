# Spell Corrector

Can a small model correct spelling well enough to be useful, and cheaply enough
to run on a CPU? This repository is nine experiments answering that, ending in a
deployed **0.8B-parameter corrector, quantized to Q4_K_M (505 MiB), that gets
~87% of BEA-60K word errors right at a p50 of ~0.6s on 4 vCPUs**.

Total GPU spend for the whole project: about **$10**.

> **Status: proof of concept, done on purpose.** The line of experiments reached
> a working answer and stopped. Nothing here is a product, and the headline
> accuracy numbers below are small-sample — see [Health warnings](#health-warnings).

## Read these two first

| | |
|---|---|
| **[reports/WORKLOG.md](reports/WORKLOG.md)** | The whole arc: every experiment, what worked (Part I), what failed and why (Part II), what it cost, and what it all adds up to. If you read one file, read this one. |
| **[reports/TAKEAWAY.md](reports/TAKEAWAY.md)** | The conclusions worth carrying to the next project — on training small models, distillation, tooling, and what serving actually costs. Personal notes, written in Romanian. |

[reports/README.md](reports/README.md) is the reference index behind them: every
measured number in one table with its sample size, plus what has been ruled out.
Per-experiment write-ups live in [reports/experiments/](reports/experiments/).

## Where it landed

Two things ended up mattering, and they point the same way.

**1. A candidate list is a liability, not an asset.** The project started as a
*reranker*: Hunspell proposes words, a trained model picks one. That approach is
capped at ~81% overall, because Hunspell's list simply does not contain the right
word for ~19% of BEA-60K errors. Three independent systems hit that wall. Dropping
candidates entirely and having the model emit the correction directly cost **one
point of accuracy and bought a 53x latency win**.

**2. Distillation beats direct training at this size.** A 2B teacher distilled
into an 0.8B student loses 1.45 points and serves sub-second on CPU — better than
training the 0.8B on examples directly.

| System | n | Acc@1 | Latency (p50) | Where |
|---|---:|---:|---:|---|
| Hunspell top-1 (do-nothing baseline) | 68,429 | 53.67% | — | [worklog §1](reports/WORKLOG.md) |
| Aspell top-1 (the baseline to beat) | 68,429 | 60.56% | — | [worklog §1](reports/WORKLOG.md) |
| 87M byte reranker, trained from scratch | 68,429 | 64.82% | — | [exp 2](reports/experiments/02-byte-reranker-87m/README.md) |
| gemma-4-E2B-it, zero-shot, answering freely | 100 | 90.0% | 960ms (q4_0) | [exp 6](reports/experiments/06-llm-judge-cpu-llamacpp/README.md) |
| Qwen3.5-2B QLoRA direct corrector (teacher) | 2,000 | 88.75% | 0.318s (3090) | [worklog §6](reports/WORKLOG.md) |
| **Qwen3.5-0.8B distilled student, Q4_K_M, CPU** | **2,000** | **86.60%** | **0.597s** | [exp 8](reports/experiments/08-distill-2b-to-08b/README.md) |

The last row is the artifact that got deployed.

### Health warnings

- **Only three rows above use the full 68,429-error benchmark.** Everything else
  is n=100-2,000. A 100-row proportion near 88% carries roughly ±6 pp; the
  gemma row's ±8-10 pp is wide enough to swallow most of its lead.
- **The two best numbers were never scored against each other at scale.** No
  fine-tuned model has been run on all of BEA-60K.
- **BEA-60K itself is imperfect** — some "errors" are real-word substitutions,
  some gold corrections are wrong. It was chosen for being easy to use.

## Try it

`public/` + `api/` deploy to Vercel as a static page plus two serverless routes.
[`public/index.html`](public/index.html) corrects one word in context;
[`public/checker.html`](public/checker.html) highlights a whole document. Both
let you pick a backend:

| Choice | Route | What it runs |
|---|---|---|
| **Local (Vercel GGUF)** | `POST /api/correct` | The distilled 0.8B Q4_K_M student via `node-llama-cpp`, in the Lambda |
| **RunPod GPU** | `POST /api/correct-runpod` | Server-side proxy to RunPod Serverless Flex (`ctalau/qwen35-08b-spell-m7-distill` on `worker-vllm`) |

The browser never talks to RunPod directly. Set these on the Vercel project
(Settings → Environment Variables) — never in client JS, never committed:

| Variable | Required | Default |
|---|---|---|
| `RUNPOD_API_KEY` | yes, for the RunPod path | — |
| `RUNPOD_ENDPOINT_ID` | no | `835g1wte9tgcor` |

Without `RUNPOD_API_KEY` the Local path still works and the RunPod option returns
a 503 with a clear error. Both paths use greedy decoding and the same
`direct_correct_v1.txt` prompt. GPU workers scale to zero, so a true cold start
(image pull + vLLM compile) takes minutes; the proxy waits ~55s and returns 504
with a retry message. Once warm, a correction is sub-second.

```bash
npm install
npm run test:web          # node --test tests/test_api_spelling.mjs
```

## Repo map

| Path | What's in it |
|---|---|
| `reports/` | **The substance.** Worklog, takeaways, the experiment index, and one directory per experiment with its plan, write-up and raw results. |
| `artifacts/` | Trained checkpoints and their metrics. `spell_slm_m7_q4/` is the deployed 0.8B Q4_K_M student; `model/` is the 87M byte reranker. |
| `scripts/` | Data building, training, benchmarking. `distill/` is the teacher→student pipeline, `runpod/` the GPU pod helpers, `kev/` the chooser harness. |
| `spelling_reranker/` | The byte-level reranker package (experiments 1-2). |
| `api/`, `public/` | The Vercel demo. |
| `configs/` | Training configs for the reranker and the frozen-encoder pilot. |
| `tests/` | CPU tests, including the one that fails the build on benchmark leakage. |

## Ground rules

- **BEA-60K is a locked benchmark.** Never train, validate, tune or prompt-search
  on it; never commit BEA files. `tests/test_dataset.py` fails the build if a
  training-construction file so much as mentions it. The one documented exception
  is experiment 8's held-out, sentence-disjoint distillation split —
  see [its plan](reports/experiments/08-distill-2b-to-08b/PLAN.md).
- **Aspell's suggestions stay out of the candidate pool.** Adding them would lift
  the reranking ceiling from ~81% to 87.7%, which is why it is tempting; it is
  out of scope by explicit decision. Do not quietly reintroduce it to hit a number.
- **Runpod pods bill for as long as they exist.** Always finish a GPU run with
  `python scripts/runpod/terminate.py --all`.
- Seed **1337** everywhere, for data and training.

## Install (Python side)

System packages (Debian/Ubuntu):

```bash
sudo apt-get install -y hunspell libhunspell-dev hunspell-en-us aspell aspell-en
```

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate

# GPU (Runpod / CUDA 12.x). Use the CPU index on machines without NVIDIA.
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -e ".[dev]"
```

The `hunspell` binding builds from source and fails against setuptools >= 60
(`AttributeError: install_layout`). On Python 3.11 and older:

```bash
pip install "setuptools<60" wheel
pip install --no-build-isolation hunspell==0.5.5
```

On Python 3.12 that workaround does not apply — there is no `distutils` for old
setuptools to patch — so install the distro package, which is the same version
already compiled for the interpreter:

```bash
sudo apt-get install -y python3-hunspell
```

Check it, then run the tests:

```bash
echo teh | hunspell -d en_US -a
python -m pytest tests/ -q
```

## Reproduce the experiments

Each experiment directory has its own commands and front matter. The common
entry points:

```bash
# Baselines and the full benchmark (CPU, free)
python scripts/download_bea60k.py
python scripts/benchmark_aspell.py
python scripts/benchmark_bea60k.py --model artifacts/model --output reports/bea60k

# LLM judges, no training (exp 5-6)
python scripts/llm_judge_bea60k_cpu.py --help   # CPU / llama.cpp, four answer modes
python scripts/llm_judge_bea60k.py --help       # GPU-targeted index mode

# The byte reranker (exp 1-2)
python scripts/download_sources.py
python scripts/build_training_data.py --target-train 4000000 --target-valid 60000
python scripts/train.py --config configs/train_full.yaml

# Distillation, teacher -> student (exp 8)
ls scripts/distill/     # build_data, dump_teacher_logits, train_student, eval_gguf
```

GPU runs go through [`scripts/runpod/`](scripts/runpod/README.md).

## If you pick this up

The open threads, in the order they are worth doing:

1. **Score the deployed student on the full 68,429 errors.** Every fine-tuned
   number in this repo is n≤2,000; the comparison against Aspell and the 87M
   reranker is not yet apples-to-apples.
2. **Chase the free-answering ceiling.** gemma's 90% in open mode, zero-shot and
   untrained, is still the highest number measured here.
3. **Revisit the serving stack.** The same weights scored differently under
   different runtimes — see [TAKEAWAY.md](reports/TAKEAWAY.md).

Experiments 3, 4 and 7 are dead — defunded, NaN'd mid-run, and abandoned in
flight respectively. [WORKLOG Part II](reports/WORKLOG.md) says what each was
for, so nobody repeats them by accident.
