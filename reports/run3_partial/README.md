# Partial artifacts from the run that OOMed

Pod `3gxtgufkzcvl40`, commit `09b7978`. The run completed the whole data
pipeline and died at step 18 of 8,130 in the full train with a CUDA OOM
(microbatch 256 on a 44.5 GiB L40S). No model was produced.

Kept here because the data build is expensive (~19 min) and these statistics
are the evidence that the data-side fixes worked:

- `data_stats.json` — 2,081,171 train / 20,773 validation examples; edit-distance
  mixture, gold-index histogram, context-noise rate, phase timings.
- `manifest.json` — dataset hashes.

The headline: after the "gold must be in the Hunspell pool" filter the training
set is 81.7 / 16.9 / 1.4 percent ED1 / ED2 / ED3+, against 83.0 / 15.7 / 1.3 for
the BEA-60K errors where gold is in the pool. That near-match is the evidence
that experiment 1's core defect (an ED1-only training set) is addressed.

See [reports/experiments/02-byte-reranker-87m/RUN_NOTES.md](../experiments/02-byte-reranker-87m/RUN_NOTES.md) sections 4-6 for the full post-mortem.
