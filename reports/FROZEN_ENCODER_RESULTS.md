# Frozen ModernBERT pilot — results

Date: 2026-09-10 (Europe/Bucharest). Pod `k9x8j2kl9dultm`, commit `287b43d`, ~$0.38.

## Summary

Frozen ModernBERT pilot did not beat Hunspell. H1 scalar BEA-1k peaked at 52.4%/64.1% then stopped by >1pp gate. H2/H3 hit non-finite features/loss. H0 d-pair remains ~81%. Admission to full-data expansion is not met.

## Metrics

| Arm | D-pair / notes | BEA-1k best |
|---|---|---|
| H0 Hunspell | overall 81.11% / cond 81.69% | — |
| H1 scalar | val cond 78.89%; stopped by BEA gate | overall 52.40% / cond 64.14% (step 200) |
| H2 linear | soft-fail rc=2, nan_seen | step0 35.40% / 43.33% |
| H3 MLP | hard-fail after 3e-4 retry (non-finite features) | attempt1 ~32.3% / 39.5% |

## Artifacts

- Local: `/workspace/spell-corrector-frozen-artifacts6/`
- Monitor: `/workspace/spell-corrector-frozen-monitor6/summary.json`

Full machine-readable summary attached as `FROZEN_ENCODER_RESULTS.json`.
