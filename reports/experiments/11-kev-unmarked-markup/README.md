# Experiment 11 — kev as a judge for missing DITA markup in the user guide

| | |
|---|---|
| **Status** | **completed** |
| **When** | 2026-09-22, branch `claude/detect-unmarked-content-asl2fm`. Target: `ctalau/userguide` at `77b0450` (2,740 DITA topics). |
| **Question** | Where does the guide name a UI control, a file or a piece of code in plain prose, when it should use `<uicontrol>`, `<filepath>` or `<codeph>` ("click Save" with no `<uicontrol>`)? |
| **Method** | Regexes and structural rules propose spans (high recall, poor precision). `kev-4b` then decides what each span is. |
| **Headline** | The chosen rule flags **5,403 spans**: 1,257 `uicontrol`, 716 `filepath` and 3,430 `codeph`. On a blind, hand-labelled sample of 200 candidates, **74% of flags name the right element** and 78% of the spans that need markup get flagged. On markup the writers already have, shown to kev as plain text, it recovers **60% of `<uicontrol>`, 91% of `<filepath>` and 64% of `<codeph>`**. |
| **Cost** | Two Runpod pods, ~$0.40 in total. The main run took 67 min on an RTX 4000 Ada at $0.28/hr, about 240 ms per span with four questions each. The first attempt ran a choice-only question on an RTX A5000. Future runs use RTX 3090s ([`launch_kev_markup.py`](../../../scripts/runpod/launch_kev_markup.py)). |
| **Output** | [`reports/markup_audit/findings.csv`](../../markup_audit/findings.csv): one row per flag with the suggested element, the score, file, line and sentence. Rows are ordered by element, most confident first. |

## Pipeline

1. **Nominate** ([`find_unmarked.py`](../../../scripts/kev/find_unmarked.py)).
   - Each block (`p`, `li`, `cmd`, `entry`, `dt`/`dd`, `note`, …) is flattened to plain text.
   - Text already inside semantic markup is skipped: `uicontrol`, `menucascade`, `codeph`, `filepath`, `xmlelement`, `xmlatt`, `xref`, `codeblock`, `keyword`, `term`, …
   - Titles are skipped as well.
   - `<b>`, `<i>`, `<u>` and `<ph>` are treated as transparent, so text inside them counts as unmarked.
   - Five sources propose spans:

   | source | rule | spans |
   |---|---|---:|
   | UI regexes | a UI verb followed by a capitalized label (`click Save`, `select the Format and Indent`), a label followed by a UI noun (`Advanced tab`, `Outline view`), and `A > B` cascades | 809 |
   | filepath regexes | a word with a known file extension, and path shapes (`C:\`, `~/`, `${var}/`, `[OXYGEN_INSTALL_DIR]/`, `a/b/c`) | 207 |
   | codeph regexes | `<tag>`, `@attr`, `xsl:template`, `f()`, `${var}`, `$VAR`, `--flag`, camelCase / snake_case / dotted identifiers, `name="value"` | 2,293 |
   | emphasis | a `<b>`, `<i>` or `<u>` span of 6 words or fewer: the writer marked it, but not semantically | 10,973 |
   | `<dt>` convention | a `<dt>` with no `<uicontrol>`, in a `<dl>` where at least half the terms use one | 157 |

   - Short blocks also get their neighbours appended as context: a `<dt>` gets its `<dd>`, a table cell gets its row.

2. **Judge** ([`judge_unmarked.py`](../../../scripts/kev/judge_unmarked.py)).
   - For each span, kev gets one request that marks the span with `[[…]]` and asks four questions:
     - a `choice` between uicontrol, filepath, codeph and plain, each option described in one sentence;
     - three `noul` questions, one per element (for example, "Is "X" the name of a specific control … of the application's user interface?").
   - The wording was fixed before any run and not tuned.

3. **Score and decide** ([`report_unmarked.py`](../../../scripts/kev/report_unmarked.py)).
   - Recall is measured on 900 spans the writers did mark up (300 per element), extracted with `--gold` and shown as plain text.
   - Precision is measured on 200 candidates I labelled by hand before seeing any kev output: 40 per source, stored in [`hand_labels.jsonl`](../../markup_audit/hand_labels.jsonl).
   - The labels follow the guide's house style, checked by counting in the corpus:
     - modes, views and tabs are `<uicontrol>` (`<uicontrol>Author</uicontrol> mode` appears 593 times against 45 plain);
     - "X editor" is plain (`Markdown editor` is never marked up);
     - parameter, property, element and attribute names count as code.

## Results

Gold recall is the share of spans the writers marked up that kev also calls that element (n=300 per element). Precision is on the 200 hand-labelled candidates: a flag counts as right only if the span needs markup *and* the element is the right one.

| rule | uicontrol | filepath | codeph | sample flags | precision | sample recall | total flags |
|---|---:|---:|---:|---:|---:|---:|---:|
| `choice` alone | 0.33 | **0.91** | 0.65 | 99 | 0.75 | 0.66 | 4,714 |
| `noul@0.5` (best yes/no answer) | 0.66 | 0.75 | 0.88 | 179 | 0.51 | 0.82 | 12,193 |
| `noul@0.8` | 0.57 | 0.69 | 0.58 | 99 | 0.66 | 0.58 | 5,711 |
| `hybrid@0.7` (element from the choice, gated by its yes/no) | 0.48 | 0.85 | 0.72 | 129 | 0.63 | 0.72 | 7,593 |
| **`combined@0.8`** (the choice, but uicontrol wins when p(yes) ≥ 0.8 and is the top yes/no) | **0.60** | **0.91** | 0.64 | 117 | **0.74** | **0.78** | **5,403** |

What each question is good at:
- **The choice is precise but will not call things UI.** It sends two in three real `<uicontrol>` spans to "plain", and in the sample it catches 2 of the 10 UI regex hits that need markup.
- **The yes/no questions are the opposite.** `uicontrol` recall doubles. But the `codeph` question says yes to almost anything short and technical, which is how it flags 9.8k spans as code.
- **`combined` takes the best of each.** Filepath and codeph decisions come from the choice, and the UI-label decisions come from the yes/no.

Precision of `combined@0.8` by source (right flags / flags; spans needing markup in the sample):

| source | right / flagged | need markup / sampled | flags in the guide |
|---|---:|---:|---:|
| codeph regexes | 29 / 32 | 36 / 40 | 1,700 |
| filepath regexes | 28 / 36 | 33 / 40 | 176 |
| `<dt>` convention | 17 / 24 | 24 / 40 | 102 |
| emphasis | 6 / 11 | 9 / 40 | 3,079 |
| UI regexes | 7 / 14 | 10 / 40 | 346 |

The files with the most flags are all reference tables:
- `ch_css_extensions.dita` (213: CSS values in `<b>`);
- the WebHelp parameter reusables (210);
- `reusables-editor-variables.dita` (187: `${var}` in `<b>`);
- `dcpp_parameters.dita` (172).

## What a reviewer should know

- **Treat each span as a location, not an exact edit.**
  - A regex hit can be partial: `Help > Install` for the menu path "Help > Install new add-ons", or `AI Positron` for the option "Enable AI Positron".
  - "codeph" is the umbrella verdict. For element and attribute names the guide mostly uses `<xmlelement>` / `<xmlatt>`, and parameter names would fit `<parmname>`.
- **Emphasis is the noisiest source.**
  - Recurring false positives: `<b>Result</b>`, `<b>Step Result</b>`, `<b>Example</b>`, `<i>well-formed</i>`.
  - The recurring *true* flags there are `true`/`false`/`yes`/`no` and CSS values set in bold, and bold `${editor variables}`.
- **A 0.5-precision source is still useful as a review queue**, but not for automatic fixes. The codeph (91%) and filepath (78%) regex flags are the only ones clean enough to consider fixing with a script.
- **The UI regexes are weak nominators.** Of 40 sampled hits, 30 are technology names before a UI noun (`XSLT editor`, `AI action`, `HTML page`), not labels.

## Honest caveats

- **The hand labels are one person's (the session's), n=200, 40 per source.** A per-source precision like 7/14 carries ±25 pp.
- **The rule and its threshold were picked on the same 200 labels they are scored on**, so 0.74 is optimistic. It was chosen from 16 rule/threshold pairs, with `choice` (0.75) and `combined` (0.72–0.75) essentially tied on precision. `combined` won on recall, which the gold rows measure independently of the labels.
- **Gold recall is measured on spans that are *already* marked up**, so the context around them is the writers' marked-up prose read as plain text. Real unmarked spans may sit in sloppier sentences.
- **Keyrefs and conrefs render empty.** `<ph keyref="product"/>` becomes nothing, so kev sees "…the  editor".
- **This was a first run:** kev-4b in bf16 on a GPU, without the `flash-linear-attention` kernels. An earlier attempt on this box ran out of memory merging the fp32 LoRA (a 15 GB box against 16 GB of weights).

## Reproduce

```bash
python scripts/kev/find_unmarked.py ../userguide/DITA --out reports/markup_audit/candidates.jsonl
python scripts/kev/find_unmarked.py ../userguide/DITA --gold --out /tmp/gold.jsonl   # then sample 300/element
python scripts/runpod/launch_kev_markup.py --branch <branch>        # RTX 3090; writes judged_*.jsonl on the pod
python scripts/runpod/fetch_artifacts.py <pod-id> --dest /tmp/kev && python scripts/runpod/terminate.py <pod-id>
python scripts/kev/report_unmarked.py reports/markup_audit/judged_gold_sample.jsonl <judged_candidates.jsonl> \
    --labels reports/markup_audit/hand_labels.jsonl --rule combined --threshold 0.8 --out reports/markup_audit
```

`judged_candidates.jsonl.gz` is committed compressed. Unzip it before re-scoring.
