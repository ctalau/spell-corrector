"""LLM-judge experiment helpers -- CPU-local track.

A separate track from the trained byte-level reranker (spelling_reranker/model.py):
here a general-purpose instruction-tuned LLM is prompted zero-shot with
Hunspell's numbered suggestion list and asked to pick the best one, or (in
"open"/"beam" answer modes) to correct the typo without being restricted to
Hunspell's list at all. See scripts/llm_judge_bea60k_cpu.py for the runnable
CLI and reports/llm_judge_cpu/ for results.

Named "_cpu" to sit alongside `spelling_reranker/llm_judge.py` (the GPU/Runpod
track, index-mode only, with the torch/cudnn pinning a real GPU pod needs).
This module requires torch + transformers but was written for and validated
on a CPU-only box directly via the CLI -- no pod, no bootstrap script.
Expect it to be slow (multi-second per call for a ~1-10B model on 4 vCPUs),
which is exactly what the results in reports/EXPERIMENT_LLM_JUDGE_CPU.md
document, latency included.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

SYSTEM_PROMPT = (
    "You are an expert English spelling-correction assistant. You will be "
    "shown a sentence with one misspelled word marked <TYPO>...</TYPO>, and "
    "a numbered list of candidate corrections from a spell-checker, in the "
    "spell-checker's own ranked order. Choose the single best replacement "
    "for the marked word given the sentence context. "
    "Reply with ONLY the candidate number and nothing else."
)

#: Open-answer variant: the same Hunspell candidates are shown as a hint, but
#: the model is free to write a different word instead of picking one of
#: them. Used to measure how much the forced-choice-from-candidates format
#: (SYSTEM_PROMPT above) itself caps accuracy, versus the model's own
#: unconstrained spelling knowledge.
SYSTEM_PROMPT_OPEN = (
    "You are an expert English spelling-correction assistant. You will be "
    "shown a sentence with one misspelled word marked <TYPO>...</TYPO>, and "
    "a numbered list of candidate corrections from a spell-checker, in the "
    "spell-checker's own ranked order, as a hint. Give the single best "
    "corrected spelling for the marked word given the sentence context. You "
    "are not limited to the candidate list -- if none of them are right, "
    "write the correct word yourself. "
    "Reply with ONLY the corrected word and nothing else."
)

_NUMBER_RE = re.compile(r"\d+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def build_messages(
    context_before: str, typo: str, context_after: str, candidates: Sequence[str]
) -> list[dict]:
    lines = [
        f"Sentence: {context_before}<TYPO>{typo}</TYPO>{context_after}",
        "Candidates:",
    ]
    lines += [f"{i + 1}. {cand}" for i, cand in enumerate(candidates)]
    lines.append("Answer with only the candidate number.")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def build_open_messages(
    context_before: str, typo: str, context_after: str, candidates: Sequence[str]
) -> list[dict]:
    lines = [
        f"Sentence: {context_before}<TYPO>{typo}</TYPO>{context_after}",
        "Spell-checker candidates (hint, you may ignore these):",
    ]
    lines += [f"{i + 1}. {cand}" for i, cand in enumerate(candidates)]
    lines.append("Answer with only the corrected word.")
    return [
        {"role": "system", "content": SYSTEM_PROMPT_OPEN},
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse_choice(text: str, n_candidates: int) -> int | None:
    """1-indexed candidate number parsed from free-form model output."""
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    n = int(match.group())
    return n if 1 <= n <= n_candidates else None


def parse_open_word(text: str) -> str | None:
    """First word-like token from a free-form model output, or None."""
    match = _WORD_RE.search(text)
    return match.group() if match else None


@dataclass
class LoadedModel:
    model_id: str
    model: object
    tokenizer: object
    device: object
    supports_system_role: bool
    supports_enable_thinking: bool
    load_class: str


def load_llm(model_id: str, *, dtype: str = "auto", trust_remote_code: bool = True) -> LoadedModel:
    import torch
    import transformers
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype == "auto":
        # bf16 halves resident memory versus fp32, which matters most on a
        # CPU box with limited RAM; CPU matmul kernels support it in recent
        # PyTorch, just slower than fp32 in some paths.
        dtype = "bfloat16"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    torch_dtype = getattr(torch, dtype)

    model = None
    load_class = None
    last_exc: Exception | None = None
    # Newest small instruct releases are frequently multimodal (vision/audio
    # encoder attached) and are published expecting AutoModelForMultimodalLM;
    # fall back through the other classes until one works for text-only
    # generation.
    for class_name in (
        "AutoModelForMultimodalLM",
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoModel",
    ):
        loader = getattr(transformers, class_name, None)
        if loader is None:
            continue
        try:
            model = loader.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )
            load_class = class_name
            break
        except Exception as exc:  # noqa: BLE001 - trying multiple loaders on purpose
            last_exc = exc
            continue
    if model is None:
        raise RuntimeError(f"could not load {model_id} with any AutoModel class: {last_exc}")

    model.to(device)
    model.eval()

    supports_system_role = True
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
            tokenize=False,
        )
    except Exception:
        supports_system_role = False

    supports_enable_thinking = True
    try:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "y"}], tokenize=False, enable_thinking=False
        )
    except TypeError:
        supports_enable_thinking = False
    except Exception:
        # Template exists and accepts the kwarg but failed for some other
        # (template-specific) reason; that is not our concern here.
        supports_enable_thinking = True

    return LoadedModel(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        device=device,
        supports_system_role=supports_system_role,
        supports_enable_thinking=supports_enable_thinking,
        load_class=load_class,
    )


def _prepare_inputs(loaded: LoadedModel, messages: list[dict]):
    """Fold system role / thinking-mode handling and tokenize the prompt.
    Shared by every generation entry point below."""
    tok = loaded.tokenizer
    msgs = messages
    if not loaded.supports_system_role:
        system = [m for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        if system and rest:
            rest[0] = {"role": rest[0]["role"], "content": system[0]["content"] + "\n\n" + rest[0]["content"]}
        msgs = rest

    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if loaded.supports_enable_thinking:
        template_kwargs["enable_thinking"] = False
    prompt = tok.apply_chat_template(msgs, **template_kwargs)
    return tok(prompt, return_tensors="pt", add_special_tokens=False).to(loaded.device)


def generate_once(
    loaded: LoadedModel,
    messages: list[dict],
    *,
    max_new_tokens: int = 8,
    prompt_lookup_num_tokens: int = 0,
) -> tuple[str, float]:
    """Greedy-decode a short completion; returns (text, wall-clock seconds).

    `prompt_lookup_num_tokens` > 0 turns on prompt-lookup speculative decoding:
    at each step transformers drafts that many tokens by matching the last few
    generated tokens against the prompt and copying what followed there, then
    verifies the whole draft in a single forward pass, keeping the longest
    prefix greedy decoding would have produced anyway. The output is therefore
    identical to plain greedy decoding -- this is a speed setting, not a
    behaviour setting -- and it pays off exactly when the answer copies the
    prompt, which is what an answer mode that rewrites the input sentence does
    (see reports/EXPERIMENT_LLM_JUDGE_CPU.md for the measured ~2-3x). It is
    useless for the modes that emit a candidate number or a single word, since
    there is nothing long enough to copy.
    """
    import torch

    tok = loaded.tokenizer
    inputs = _prepare_inputs(loaded, messages)

    extra = {}
    if prompt_lookup_num_tokens:
        extra["prompt_lookup_num_tokens"] = prompt_lookup_num_tokens

    if loaded.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = loaded.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            **extra,
        )
    if loaded.device.type == "cuda":
        torch.cuda.synchronize()
    latency = time.perf_counter() - t0

    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    return text, latency


#: Generative mode: the model corrects the typo from its own knowledge, with
#: no Hunspell candidate list at all -- beam search below supplies multiple
#: candidate words instead, reranked by edit distance to the typo.
SYSTEM_PROMPT_GENERATIVE = (
    "You are an expert English spelling-correction assistant. You will be "
    "shown a sentence with one misspelled word marked <TYPO>...</TYPO>. "
    "Give the single best corrected spelling for the marked word given the "
    "sentence context. "
    "Reply with ONLY the corrected word and nothing else."
)


def build_generative_messages(context_before: str, typo: str, context_after: str) -> list[dict]:
    content = (
        f"Sentence: {context_before}<TYPO>{typo}</TYPO>{context_after}\n"
        "Answer with only the corrected word."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_GENERATIVE},
        {"role": "user", "content": content},
    ]


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (insert/delete/substitute, cost 1 each)."""
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def beam_word_candidates(
    loaded: LoadedModel, messages: list[dict], *, beam_width: int = 3, max_new_tokens: int = 6
) -> tuple[list[dict], float]:
    """Beam search (width `beam_width`) for a short completion, truncated at
    each beam's own first word boundary.

    Uses transformers' own beam search (`num_beams=num_return_sequences=
    beam_width`), which keeps only the `beam_width` highest cumulative-
    log-probability sequences at every generation step -- the standard
    reading of "branch by the top-k tokens at every step". Each of the
    `beam_width` returned sequences is then cut at its first word boundary
    (whitespace/punctuation/EOS) via `parse_open_word`, and its
    log-probability is re-summed over only the tokens up to that cut using
    `compute_transition_scores`, so a beam that kept generating past the word
    is not penalized for tokens beyond it.

    Returns (candidates, wall-clock seconds), where each candidate is
    {"word": str | None, "logprob": float, "n_tokens": int}.
    """
    import torch

    tok = loaded.tokenizer
    inputs = _prepare_inputs(loaded, messages)
    prompt_len = inputs["input_ids"].shape[1]

    if loaded.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = loaded.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=beam_width,
            num_return_sequences=beam_width,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )
    if loaded.device.type == "cuda":
        torch.cuda.synchronize()
    latency = time.perf_counter() - t0

    gen_ids_batch = out.sequences[:, prompt_len:]
    transition_scores = loaded.model.compute_transition_scores(
        out.sequences, out.scores, getattr(out, "beam_indices", None), normalize_logits=True
    )

    candidates: list[dict] = []
    for row_ids, row_scores in zip(gen_ids_batch.tolist(), transition_scores.tolist()):
        cum_logprob = 0.0
        used_tokens = 0
        word: str | None = None
        for tid, score in zip(row_ids, row_scores):
            is_pad = tid == tok.pad_token_id
            is_eos = tok.eos_token_id is not None and (
                tid == tok.eos_token_id if isinstance(tok.eos_token_id, int) else tid in tok.eos_token_id
            )
            if is_pad or is_eos or score == float("-inf"):
                break
            cum_logprob += score
            used_tokens += 1
            text_so_far = tok.decode(row_ids[:used_tokens], skip_special_tokens=True)
            candidate_word = parse_open_word(text_so_far)
            if candidate_word is not None and len(text_so_far.strip()) > len(candidate_word):
                word = candidate_word
                break
        if word is None and used_tokens:
            word = parse_open_word(tok.decode(row_ids[:used_tokens], skip_special_tokens=True))
        candidates.append({"word": word, "logprob": cum_logprob, "n_tokens": used_tokens})
    return candidates, latency


def select_by_edit_distance_and_probability(
    candidates: list[dict], typo: str, *, edit_distance_weight: float = 1.0
) -> list[dict]:
    """Score each beam candidate as `logprob - weight * edit_distance(word,
    typo)` and return the surviving candidates sorted best-first (best =
    argmax). Edit distance is to the *typo*, not the gold correction (unknown
    at inference time) -- a cheap noisy-channel-style prior that a genuine
    correction is usually a small number of edits from the misspelling, given
    a weight rather than used as an absolute decider on its own.

    A candidate identical to the typo (case-insensitive) is dropped before
    scoring, not merely disadvantaged: it is not a correction at all, and its
    edit distance of 0 would otherwise let it beat a real correction on any
    near-tied logprob -- the exact failure mode this word list is built to
    avoid, not a byproduct of `edit_distance_weight` to be tuned away."""
    typo_n = typo.strip().lower()
    scored = []
    for cand in candidates:
        word = cand.get("word")
        if word is not None and word.strip().lower() == typo_n:
            continue
        dist = edit_distance(word, typo) if word is not None else None
        score = (cand["logprob"] - edit_distance_weight * dist) if dist is not None else float("-inf")
        scored.append({**cand, "edit_distance": dist, "combined_score": score})
    scored.sort(key=lambda c: c["combined_score"], reverse=True)
    return scored


#: Fixed latency-histogram bin edges in milliseconds. Deliberately fine near
#: the expected range for a <1B-effective-parameter model on a single GPU
#: (sub-second) and coarse above it.
LATENCY_BIN_EDGES_MS = [0, 50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 5000, float("inf")]


def latency_stats(latencies_s: Sequence[float]) -> dict:
    import numpy as np

    arr = np.asarray(latencies_s, dtype=float) * 1000.0  # -> ms
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "stdev_ms": float(arr.std(ddof=0)) if arr.size > 1 else 0.0,
        "min_ms": float(arr.min()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
    }


def write_latency_histogram(latencies_s: Sequence[float], out_dir, stem: str) -> None:
    import csv
    import json
    from pathlib import Path

    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(latencies_s, dtype=float) * 1000.0
    edges = LATENCY_BIN_EDGES_MS
    counts, _ = np.histogram(arr, bins=edges) if arr.size else (np.zeros(len(edges) - 1, dtype=int), None)
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels.append(f">={lo:.0f}ms" if hi == float("inf") else f"{lo:.0f}-{hi:.0f}ms")

    payload = {
        "n": int(arr.size),
        "bin_edges_ms": [e if e != float("inf") else None for e in edges],
        "bins": [{"label": lbl, "count": int(c)} for lbl, c in zip(labels, counts)],
        "stats": latency_stats(latencies_s),
    }
    (out_dir / f"{stem}_latency_histogram.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / f"{stem}_latency_histogram.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bin", "count"])
        writer.writeheader()
        for lbl, c in zip(labels, counts):
            writer.writerow({"bin": lbl, "count": int(c)})

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(labels)), counts)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("count")
    ax.set_title(f"{stem} per-call latency (n={int(arr.size)})")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_latency_histogram.png", dpi=120)
    plt.close(fig)


#: Sentence mode: the same Hunspell candidate list as index mode is shown, but
#: instead of answering with a candidate *number*, the model rewrites the whole
#: sentence with the typo replaced by its chosen correction, wrapped in
#: <corrected_sentence>...</corrected_sentence>. The correction is then
#: recovered by aligning the rewritten sentence against the original one (see
#: `extract_corrected_word`), so the scored unit stays a single word and the
#: numbers remain comparable with index/open mode.
SYSTEM_PROMPT_SENTENCE = (
    "You are an expert English spelling-correction assistant. You will be "
    "shown a sentence with one misspelled word marked <TYPO>...</TYPO>, and "
    "a numbered list of candidate corrections from a spell-checker, in the "
    "spell-checker's own ranked order. Rewrite the sentence with the marked "
    "word replaced by its best correction given the sentence context. Change "
    "nothing else: keep every other word, its spelling, capitalisation and "
    "punctuation exactly as given, and drop the <TYPO> and </TYPO> markers. "
    "Reply with ONLY the rewritten sentence wrapped in "
    "<corrected_sentence></corrected_sentence> tags and nothing else."
)


def build_sentence_messages(
    context_before: str, typo: str, context_after: str, candidates: Sequence[str]
) -> list[dict]:
    lines = [
        f"Sentence: {context_before}<TYPO>{typo}</TYPO>{context_after}",
        "Candidates:",
    ]
    lines += [f"{i + 1}. {cand}" for i, cand in enumerate(candidates)]
    lines.append(
        "Answer with only <corrected_sentence>the full sentence, with the marked "
        "word corrected</corrected_sentence>."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_SENTENCE},
        {"role": "user", "content": "\n".join(lines)},
    ]


_SENTENCE_TAG_RE = re.compile(
    r"<corrected_sentence>(.*?)(?:</corrected_sentence>|<corrected_sentence>|$)",
    re.DOTALL | re.IGNORECASE,
)


def parse_corrected_sentence(text: str) -> str | None:
    """Pull the rewritten sentence out of the model's tagged output.

    Deliberately lenient about the closing tag: an unterminated span (the
    generation budget ran out) and a repeated *opening* tag used as the
    terminator both still yield the sentence. Returns None only when no
    opening tag was produced at all -- that is a genuine format failure and is
    counted as one, not silently patched up by falling back to the raw text.
    """
    match = _SENTENCE_TAG_RE.search(text)
    if not match:
        return None
    inner = match.group(1).strip()
    return inner or None


def extract_corrected_word(
    corrected_sentence: str, context_before: str, typo: str, context_after: str
) -> tuple[str | None, int]:
    """Recover the single replacement word the model wrote for `typo`.

    Aligns the rewritten sentence against the original one on whitespace
    tokens -- the same tokenisation BEA-60K's own error extraction uses, so a
    recovered token is directly comparable with the gold token. Whatever the
    model wrote in the typo's slot (between the surviving matched tokens on
    either side) is the answer.

    Returns (word, n_span_tokens). `n_span_tokens` is how many output tokens
    filled the typo's slot: 1 is a clean one-for-one replacement, 0 a
    deletion (word is None), and >1 means the anchors on either side did not
    both survive -- the model reworded or repunctuated its neighbours too, so
    the slot swallowed them. In that last case the word returned is the token
    in the slot closest to the typo by edit distance, which is what the model
    actually wrote in the typo's place; `n_span_tokens` is reported alongside
    so this recovery stays visible rather than passing as a clean parse.
    """
    original_tokens = (context_before + typo + context_after).split()
    typo_start = len(context_before.split())
    typo_end = typo_start + len(typo.split())
    out_tokens = corrected_sentence.split()

    matcher = SequenceMatcher(
        a=[t.lower() for t in original_tokens], b=[t.lower() for t in out_tokens], autojunk=False
    )
    mapping: dict[int, int] = {}
    for i, j, size in matcher.get_matching_blocks():
        for k in range(size):
            mapping[i + k] = j + k

    # Anchor on the nearest surviving matched token on each side of the typo;
    # everything the model put between those anchors is its replacement.
    left = max((mapping[i] for i in range(typo_start) if i in mapping), default=-1)
    right = min(
        (mapping[i] for i in range(typo_end, len(original_tokens)) if i in mapping), default=len(out_tokens)
    )
    span = out_tokens[left + 1 : right]
    if not span:
        return None, 0
    if len(span) == 1:
        return span[0], 1
    best = min(span, key=lambda tok: edit_distance(strip_outer_punctuation(tok), typo))
    return best, len(span)


def strip_outer_punctuation(word: str) -> str:
    """Trim leading/trailing punctuation, keeping case and inner apostrophes."""
    return word.strip(".,;:!?\"'()[]{}<>")
