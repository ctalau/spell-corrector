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

Raw WikiText (`data/raw/`) is gitignored; download it with the script.

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
