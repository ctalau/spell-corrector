"""LLM-judge experiment helpers.

A separate track from the trained byte-level reranker (spelling_reranker/model.py):
here a general-purpose instruction-tuned LLM is prompted zero-shot with
Hunspell's numbered suggestion list and asked to pick the best one. See
scripts/llm_judge_bea60k.py for the runnable CLI and reports/llm_judge/ for
results. Requires torch + transformers + a GPU; run on Runpod
(scripts/runpod/bootstrap_llm_judge.sh), never on the local CPU-only box.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Sequence

SYSTEM_PROMPT = (
    "You are an expert English spelling-correction assistant. You will be "
    "shown a sentence with one misspelled word marked <TYPO>...</TYPO>, and "
    "a numbered list of candidate corrections from a spell-checker, in the "
    "spell-checker's own ranked order. Choose the single best replacement "
    "for the marked word given the sentence context. "
    "Reply with ONLY the candidate number and nothing else."
)

_NUMBER_RE = re.compile(r"\d+")


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


def parse_choice(text: str, n_candidates: int) -> int | None:
    """1-indexed candidate number parsed from free-form model output."""
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    n = int(match.group())
    return n if 1 <= n <= n_candidates else None


@dataclass
class LoadedModel:
    model_id: str
    model: object
    tokenizer: object
    device: object
    supports_system_role: bool
    supports_enable_thinking: bool
    load_class: str


def load_llm(model_id: str, *, dtype: str = "bfloat16", trust_remote_code: bool = True) -> LoadedModel:
    import torch
    import transformers
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    torch_dtype = getattr(torch, dtype)

    model = None
    load_class = None
    last_exc: Exception | None = None
    # Newest small instruct releases are frequently multimodal (vision/audio
    # encoder attached); AutoModelForCausalLM fails to load those, so fall
    # back through the classes that do until one of them works for
    # text-only generation.
    for class_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
        loader = getattr(transformers, class_name, None)
        if loader is None:
            continue
        try:
            model = loader.from_pretrained(
                model_id, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code
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


def generate_once(loaded: LoadedModel, messages: list[dict], *, max_new_tokens: int = 8) -> tuple[str, float]:
    """Greedy-decode a short completion; returns (text, wall-clock seconds)."""
    import torch

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
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(loaded.device)

    if loaded.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = loaded.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )
    if loaded.device.type == "cuda":
        torch.cuda.synchronize()
    latency = time.perf_counter() - t0

    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    return text, latency


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
