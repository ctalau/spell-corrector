"""Loading and word-level error extraction for the locked BEA-60K benchmark.

Shared by scripts/benchmark_bea60k.py (the trained reranker) and
scripts/llm_judge_bea60k.py (the LLM-judge experiment) so both score the
exact same error set.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path


def load_bea_pairs(bea_dir: Path) -> list[tuple[str, str]]:
    clean_path = bea_dir / "test.bea60k"
    noise_path = bea_dir / "test.bea60k.noise"
    if not clean_path.is_file() or not noise_path.is_file():
        raise FileNotFoundError(
            f"BEA files missing in {bea_dir}. Run scripts/download_bea60k.py"
        )
    cleans = clean_path.read_text(encoding="utf-8", errors="replace").splitlines()
    noises = noise_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(cleans) != len(noises):
        raise ValueError(f"line count mismatch: clean={len(cleans)} noise={len(noises)}")
    return list(zip(noises, cleans))


def align_errors(noisy: str, clean: str) -> list[dict]:
    n_toks = noisy.split()
    c_toks = clean.split()
    matcher = SequenceMatcher(a=n_toks, b=c_toks, autojunk=False)
    errors: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        n_span = n_toks[i1:i2]
        c_span = c_toks[j1:j2]
        if len(n_span) == 1 and len(c_span) == 1:
            left = " ".join(n_toks[:i1])
            right = " ".join(n_toks[i2:])
            if left:
                left += " "
            if right:
                right = " " + right
            errors.append(
                {
                    "typo": n_span[0],
                    "gold": c_span[0],
                    "context_before": left,
                    "context_after": right,
                    "noisy_sentence": noisy,
                    "clean_sentence": clean,
                }
            )
    return errors


def extract_word_errors(pairs: list[tuple[str, str]]) -> list[dict]:
    errors: list[dict] = []
    for noisy, clean in pairs:
        errors.extend(align_errors(noisy, clean))
    return errors
