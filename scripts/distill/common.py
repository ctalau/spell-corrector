"""Shared pieces of the M7 distillation pipeline.

Teacher and student must tokenize an example *identically* for token-level
distillation to mean anything, so prompt rendering, chat templating and answer
tokenization all live here and are imported by both sides.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")


def apply_chat(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # noqa: BLE001
            return user_text + "\nAnswer:"


def tokenizer_fingerprint(tokenizer) -> str:
    """Identity of the token<->id mapping, so KD across two checkpoints can be
    refused when the vocabularies are not the same one."""
    probe = "The <TYPO>recieve</TYPO> quick brown fox — naïve café 42\n"
    ids = tokenizer(probe, add_special_tokens=False)["input_ids"]
    payload = json.dumps(
        {
            "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
            "len": len(tokenizer),
            "eos": tokenizer.eos_token,
            "eos_id": tokenizer.eos_token_id,
            "probe_ids": ids,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def encode_example(tokenizer, row: dict, max_len: int) -> dict:
    """Prompt tokens (masked) + answer tokens (supervised), the M4/M5/M6 recipe.

    Returns the pieces both the teacher dump and the student trainer need:
    `prompt_len` is where the answer starts, so the logits that predict answer
    token j sit at position prompt_len + j - 1.
    """
    prompt = apply_chat(tokenizer, row["user_text"])
    answer = str(row.get("target") or row.get("gold") or "").strip()
    eos = tokenizer.eos_token or ""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer + eos, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_len]
    prompt_len = min(len(prompt_ids), len(input_ids))
    answer_ids = input_ids[prompt_len:]
    labels = [-100] * prompt_len + list(answer_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "prompt_len": prompt_len,
        "answer_ids": answer_ids,
    }


def normalize_prediction(text: str) -> str:
    """First whitespace-delimited token, stripped of the punctuation a chat
    model likes to wrap a single word in."""
    out = (text or "").strip()
    for marker in ("\n", "\r"):
        if marker in out:
            out = out.split(marker, 1)[0]
    out = out.strip().strip('"').strip("'").strip()
    if out:
        out = out.split()[0] if out.split() else out
    return out.strip().strip('"').strip("'").strip(".,;:!?")
