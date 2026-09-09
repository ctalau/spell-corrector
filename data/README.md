# Data

## Training / validation (this repository)

**Source:** [WikiText-103 raw](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/)
(`wikitext-103-raw-v1`). HuggingFace identifier: `Salesforce/wikitext` config
`wikitext-103-raw-v1`.

**License:** Creative Commons Attribution-ShareAlike (CC BY-SA). Derived
synthetic examples inherit ShareAlike obligations. Attribution: Merity et al.,
*Pointer Sentinel Mixture Models* / Salesforce WikiText.

**What is stored here**

Prepared parquet under `data/processed/` contains **synthetic typos** only:

1. Take a clean WikiText sentence/document from the official train or valid
   split (no cross-split leakage).
2. Pick an alphabetic dictionary word (length ~3–25).
3. Apply one corruption (keyboard adjacency, random substitution, delete,
   insert, transpose, duplicate, or a common English pattern).
4. Keep the example only if Hunspell flags the typo **and** the original word
   appears in Hunspell's first 10 suggestions.

Schema: `example_id`, `source`, `context_before`, `typo`, `context_after`,
`gold`, `cand_0`…`cand_9`, `gold_index`, `corruption_type`,
`source_document_id`, `original_sentence_hash`.

RNG seed: **1337**. Rebuild:

```bash
python scripts/download_sources.py
python scripts/build_training_data.py --seed 1337
```

The original Salesforce S3 zip is gone. `download_sources.py` fetches the
191,984,949-byte `wikitext-103-raw-v1.zip` from the HuggingFace mirror
`mattdangerw/wikitext-103-raw` (same archive size as the historical release)
and falls back to official `Salesforce/wikitext` parquet.

Raw WikiText (`data/raw/`) is gitignored.

### Committed processed split (seed 1337)

| Split | Examples | SHA-256 | Size |
|-------|----------|---------|------|
| train | 235,626 | `1f609447255599343633abf5794a2b31725588aedd74f98ef6f8e4263997914d` | 26,527,114 |
| validation | 19,642 | `f8aa13c56498f54026ba8deed3a02dfea1439cd1834a3b86ba1eb8d1e8c3bdd1` | 2,406,202 |

Built at 240k/20k then filtered: WikiText `= = heading = =` lines (shared
boilerplate across articles) were dropped so train/valid sentence-hash
overlap is zero. Counts remain above the 150k/10k minimum.

Gold Hunspell index 0 is common (~68%), which is expected; the long tail
(indices 1–9) is the reranking signal. See `data/processed/data_stats.json`.

## Authentic typo corpora (omitted)

The GitHub Typo Corpus and similar sources were **not** included. Licensing and
redistribution of mined repository text is unclear. PLAN.md prefers a legally
clean synthetic first experiment over delaying for authentic data.

## Locked benchmark (do not commit)

**BEA-60K** as redistributed by [NeuSpell](https://github.com/neuspell/neuspell)
(`test.bea60k` / `test.bea60k.noise`, 63,044 sentence pairs) is a **final
benchmark only**.

- Download: `python scripts/download_bea60k.py` → `data/bea60k/` (gitignored).
- Do not use these files for training, validation, architecture choices, or
  hyperparameter selection.
- Do not commit the files if redistribution terms do not allow it. Commit the
  downloader, recorded checksums after a local run, and metric reports instead.

Upstream shared task: [BEA-2019](https://www.cl.cam.ac.uk/research/nl/bea2019st/).
