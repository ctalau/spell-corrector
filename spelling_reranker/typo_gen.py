"""Synthetic typo generation with a fixed corruption mixture."""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from spelling_reranker.byte_encoding import nfc

CORRUPTION_PROBS: dict[str, float] = {
    "adjacent_key": 0.25,
    "random_sub": 0.15,
    "deletion": 0.15,
    "insertion": 0.12,
    "transposition": 0.13,
    "duplicate": 0.10,
    "common_pattern": 0.10,
}

QWERTY_NEIGHBORS: dict[str, str] = {
    "q": "wa",
    "w": "qeas",
    "e": "wrds",
    "r": "etdf",
    "t": "ryfg",
    "y": "tugh",
    "u": "yihj",
    "i": "uojk",
    "o": "ipkl",
    "p": "ol",
    "a": "qwsz",
    "s": "awedxz",
    "d": "serfcx",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "j": "huiknm",
    "k": "jiolm",
    "l": "kop",
    "z": "asx",
    "x": "zsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhjm",
    "m": "njk",
}

VOWELS = "aeiou"
CONSONANTS = "bcdfghjklmnpqrstvwxyz"
DOUBLED_CONSONANT = re.compile(r"([bcdfghjklmnpqrstvwxyz])\1", re.IGNORECASE)
IE_RE = re.compile(r"ie", re.IGNORECASE)
EI_RE = re.compile(r"ei", re.IGNORECASE)
ALPHA_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'-]{2,24}$")
URL_RE = re.compile(r"(https?://|www\.|@|\.com|\.org|\.net|/)", re.IGNORECASE)


def _match_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src.islower():
        return repl.lower()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def _rand_letter(rng: np.random.Generator, upper: bool) -> str:
    letter = chr(int(rng.integers(ord("a"), ord("z") + 1)))
    return letter.upper() if upper else letter


def is_eligible_word(word: str) -> bool:
    if not ALPHA_WORD_RE.match(word):
        return False
    if URL_RE.search(word):
        return False
    letters = [c for c in word if c.isalpha()]
    return 3 <= len(letters) <= 25


def _adjacent_key(word: str, rng: np.random.Generator) -> str | None:
    idxs = [i for i, ch in enumerate(word) if ch.lower() in QWERTY_NEIGHBORS]
    if not idxs:
        return None
    i = int(idxs[int(rng.integers(0, len(idxs)))])
    neigh = QWERTY_NEIGHBORS[word[i].lower()]
    repl = neigh[int(rng.integers(0, len(neigh)))]
    return word[:i] + _match_case(word[i], repl) + word[i + 1 :]


def _random_sub(word: str, rng: np.random.Generator) -> str | None:
    idxs = [i for i, ch in enumerate(word) if ch.isalpha()]
    if not idxs:
        return None
    i = int(idxs[int(rng.integers(0, len(idxs)))])
    src = word[i]
    repl = _rand_letter(rng, src.isupper())
    while repl.lower() == src.lower():
        repl = _rand_letter(rng, src.isupper())
    return word[:i] + repl + word[i + 1 :]


def _deletion(word: str, rng: np.random.Generator) -> str | None:
    if len(word) < 2:
        return None
    i = int(rng.integers(0, len(word)))
    return word[:i] + word[i + 1 :]


def _insertion(word: str, rng: np.random.Generator) -> str | None:
    i = int(rng.integers(0, len(word) + 1))
    if word and rng.random() < 0.7:
        anchor = word[min(i, len(word) - 1)]
        neigh = QWERTY_NEIGHBORS.get(anchor.lower(), "")
        if neigh:
            ch = _match_case(anchor, neigh[int(rng.integers(0, len(neigh)))])
            return word[:i] + ch + word[i:]
    ch = _rand_letter(rng, bool(word and word[min(i, len(word) - 1)].isupper()))
    return word[:i] + ch + word[i:]


def _transposition(word: str, rng: np.random.Generator) -> str | None:
    if len(word) < 2:
        return None
    i = int(rng.integers(0, len(word) - 1))
    if word[i] == word[i + 1]:
        return None
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def _duplicate(word: str, rng: np.random.Generator) -> str | None:
    idxs = [i for i, ch in enumerate(word) if ch.isalpha()]
    if not idxs:
        return None
    i = int(idxs[int(rng.integers(0, len(idxs)))])
    return word[: i + 1] + word[i] + word[i + 1 :]


def _common_pattern(word: str, rng: np.random.Generator) -> str | None:
    options: list[str] = []
    if IE_RE.search(word):
        options.append(IE_RE.sub("ei", word, count=1))
    if EI_RE.search(word):
        options.append(EI_RE.sub("ie", word, count=1))
    doubled = list(DOUBLED_CONSONANT.finditer(word))
    if doubled:
        m = doubled[int(rng.integers(0, len(doubled)))]
        options.append(word[: m.start()] + m.group(1) + word[m.end() :])
    cons_idxs = [i for i, ch in enumerate(word) if ch.lower() in CONSONANTS]
    if cons_idxs:
        i = int(cons_idxs[int(rng.integers(0, len(cons_idxs)))])
        options.append(word[: i + 1] + word[i] + word[i + 1 :])
    if word.lower().endswith("e") and len(word) > 3:
        options.append(word[:-1])
    elif len(word) >= 3 and word[-1].lower() in CONSONANTS and word[-2].lower() in VOWELS:
        options.append(word + ("E" if word[-1].isupper() else "e"))
    vowel_idxs = [i for i, ch in enumerate(word) if ch.lower() in VOWELS]
    if vowel_idxs:
        i = int(vowel_idxs[int(rng.integers(0, len(vowel_idxs)))])
        others = [v for v in VOWELS if v != word[i].lower()]
        repl = others[int(rng.integers(0, len(others)))]
        options.append(word[:i] + _match_case(word[i], repl) + word[i + 1 :])
    options = [o for o in options if o and o != word]
    if not options:
        return _adjacent_key(word, rng)
    return options[int(rng.integers(0, len(options)))]


_HANDLERS = {
    "adjacent_key": _adjacent_key,
    "random_sub": _random_sub,
    "deletion": _deletion,
    "insertion": _insertion,
    "transposition": _transposition,
    "duplicate": _duplicate,
    "common_pattern": _common_pattern,
}


def choose_corruption_type(rng: np.random.Generator) -> str:
    names = list(CORRUPTION_PROBS)
    probs = np.array([CORRUPTION_PROBS[n] for n in names], dtype=np.float64)
    probs = probs / probs.sum()
    return str(rng.choice(names, p=probs))


def corrupt_word(word: str, rng: np.random.Generator, corruption_type: str | None = None) -> tuple[str, str]:
    """Return (typo, corruption_type). May equal the original if all attempts fail."""
    kind = corruption_type or choose_corruption_type(rng)
    handler = _HANDLERS[kind]
    typo = handler(word, rng)
    if typo is None or typo == word:
        for fallback in ("adjacent_key", "deletion", "insertion", "random_sub"):
            typo = _HANDLERS[fallback](word, rng)
            if typo and typo != word:
                kind = fallback
                break
    if typo is None:
        typo = word
    return nfc(typo), kind


def tokenize_sentence(sentence: str) -> list[tuple[int, int, str]]:
    """Return (start, end, token) spans for whitespace-separated tokens."""
    tokens: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\S+", sentence):
        tokens.append((match.start(), match.end(), match.group()))
    return tokens


def eligible_token_indices(tokens: Sequence[tuple[int, int, str]]) -> list[int]:
    return [i for i, (_, _, tok) in enumerate(tokens) if is_eligible_word(tok)]
