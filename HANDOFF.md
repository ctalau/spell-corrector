# HANDOFF.md — retired

This file was a session handoff written **before** experiment 2 produced a
number: at the time there was no trained model, the last run had died at step 18
of 8,130 with a CUDA OOM, and the document told the next session how to relaunch
it. Experiment 2 has since completed (64.82% overall on BEA-60K, pod
`xwvjd2w980kk00`, commit `b9b66cf`), so the file no longer describes the state of
anything.

Nothing was thrown away. Its durable content now lives here:

| What you were looking for | Now at |
|---|---|
| The nine fixed bugs, including the CUDA OOM post-mortem | [reports/experiments/02-byte-reranker-87m/RUN_NOTES.md §5](reports/experiments/02-byte-reranker-87m/RUN_NOTES.md) |
| Known issues not yet fixed (typo-table sharding, `gold0_fraction`, `nproc` in containers, preflight on CPU) | [RUN_NOTES.md §6](reports/experiments/02-byte-reranker-87m/RUN_NOTES.md) |
| Runpod operating procedure: self-driving pod, HTTP monitoring, fetch-then-terminate, phase timings | [RUN_NOTES.md §3](reports/experiments/02-byte-reranker-87m/RUN_NOTES.md) |
| Budget and the balance-check query | [RUN_NOTES.md §8](reports/experiments/02-byte-reranker-87m/RUN_NOTES.md) |
| Benchmark-integrity rules and the typo calibration table | [RUN_NOTES.md §1](reports/experiments/02-byte-reranker-87m/RUN_NOTES.md) |
| Why the ceiling is ~81% and why Aspell is out of scope | [RUN_NOTES.md §2](reports/experiments/02-byte-reranker-87m/RUN_NOTES.md) |
| The data-distribution evidence and the OOMed run's statistics | [RUN_NOTES.md §4](reports/experiments/02-byte-reranker-87m/RUN_NOTES.md), [reports/run3_partial/](reports/run3_partial/) |
| The goal, the diagnosis, the nine changes, the run configuration | [reports/experiments/02-byte-reranker-87m/PLAN.md](reports/experiments/02-byte-reranker-87m/PLAN.md) |
| The result itself | [reports/experiments/02-byte-reranker-87m/README.md](reports/experiments/02-byte-reranker-87m/README.md) |
| What has happened since | [reports/README.md](reports/README.md) — the experiment index |

Its "recommended next steps" are obsolete: step 1 was "relaunch", which happened
and succeeded. The current queue is at the bottom of
[reports/README.md](reports/README.md).
