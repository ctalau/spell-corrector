"""Synthetic typo generation calibrated against authentic human misspellings.

The first experiment generated exclusively edit-distance-1 typos (single
keystroke slips). Real misspellings are not like that. Measured on Wikipedia's
"Lists of common misspellings" (~4.3k authentic misspelling/correction pairs, a
public list unrelated to any held-out benchmark -- see
`scripts/calibrate_typo_model.py`):

    ED1 72.6%   ED2 25.0%   ED3+ 2.4%

and Hunspell's top-1 accuracy falls off sharply as edit distance grows, so the
ED>=2 quarter is exactly the slice a reranker is supposed to earn its keep on.
Training on ED1-only data left the model blind there.

Two changes follow:

1. `corrupt_word` composes 1-3 primitive edits, with the mixture tuned so the
   realised Levenshtein distribution matches the table above.
2. A phonetic/orthographic rule set models the *cognitive* errors that dominate
   authentic misspellings: doubling, silent letters, reduced unstressed vowels,
   and suffix confusion. These naturally produce ED2 edits ("ph"->"f",
   "ance"->"ence") the way a human misspelling does, rather than the uniform
   keyboard noise the old generator produced.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from spelling_reranker.byte_encoding import nfc

#: Probability of applying 1 / 2 / 3 primitive edits. Tuned so the *realised*
#: edit distance (edits can overlap or partially cancel, and one phonetic rule
#: can already be worth two edits) lands on the authentic ED1/ED2/ED3+ mixture
#: above. `tests/test_typo_gen.py` asserts the realised distribution.
N_EDIT_PROBS: tuple[float, float, float] = (0.82, 0.14, 0.04)

#: Mixture over primitive corruption types for a single edit. Weighted towards
#: the phonetic/orthographic classes: authentic misspellings are dominated by
#: how a writer *thinks* a word is spelled, not by keyboard slips.
CORRUPTION_PROBS: dict[str, float] = {
    "phonetic": 0.30,
    "vowel_reduction": 0.12,
    "doubling": 0.11,
    "silent_letter": 0.08,
    "adjacent_key": 0.15,
    "transposition": 0.09,
    "deletion": 0.06,
    "insertion": 0.05,
    "random_sub": 0.04,
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

#: (pattern, replacement) orthographic confusions, applied at one random
#: matching site. Modelled on the classes visible in the public misspelling
#: list: suffix confusion, phoneme-to-grapheme ambiguity, and unstressed-vowel
#: spelling.
PHONETIC_RULES: tuple[tuple[str, str], ...] = (
    ("ph", "f"), ("f", "ph"),
    ("ie", "ei"), ("ei", "ie"),
    ("ck", "k"), ("k", "ck"), ("c", "k"), ("k", "c"),
    ("ce", "se"), ("se", "ce"), ("ci", "si"), ("si", "ci"),
    ("s", "z"), ("z", "s"),
    ("tion", "sion"), ("sion", "tion"), ("tion", "tian"),
    ("able", "ible"), ("ible", "able"),
    ("ance", "ence"), ("ence", "ance"),
    ("ant", "ent"), ("ent", "ant"),
    ("ary", "ery"), ("ery", "ary"), ("ory", "ary"),
    ("ous", "ious"), ("ious", "ous"), ("ous", "us"),
    ("ful", "full"), ("full", "ful"),
    ("ea", "e"), ("e", "ea"), ("ee", "e"), ("ea", "ee"),
    ("oo", "o"), ("ou", "o"), ("o", "ou"), ("au", "a"),
    ("ai", "a"), ("ay", "ai"), ("ai", "ay"),
    ("y", "i"), ("i", "y"), ("y", "ie"),
    ("gh", "g"), ("gh", ""), ("ght", "t"),
    ("qu", "q"), ("wh", "w"), ("wr", "r"),
    ("mb", "m"), ("mn", "m"), ("kn", "n"), ("ps", "s"),
    ("er", "or"), ("or", "er"), ("er", "ar"), ("ar", "er"),
    ("re", "er"), ("er", "re"),
    ("le", "el"), ("el", "le"),
    ("ate", "ait"), ("ate", "at"),
    ("ing", "eing"), ("ing", "in"),
    ("ed", "t"), ("d", "t"), ("t", "d"),
    ("th", "f"), ("v", "w"), ("w", "v"),
)

#: Consonants that commonly go silent / get dropped inside clusters.
SILENT_DROPPABLE = "hlrwgtdnebu"

ALPHA_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'-]{2,24}$")
URL_RE = re.compile(r"(https?://|www\.|@|\.com|\.org|\.net|/)", re.IGNORECASE)
DOUBLE_RE = re.compile(r"([bcdfghjklmnpqrstvwxyz])\1", re.IGNORECASE)


def _match_case(src: str, repl: str) -> str:
    """Carry `src`'s capitalisation onto `repl`."""
    if not repl:
        return repl
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def _rand_letter(rng: np.random.Generator, upper: bool) -> str:
    letter = chr(int(rng.integers(ord("a"), ord("z") + 1)))
    return letter.upper() if upper else letter


def _pick(rng: np.random.Generator, items: Sequence):
    return items[int(rng.integers(0, len(items)))]


def is_eligible_word(word: str) -> bool:
    if not ALPHA_WORD_RE.match(word):
        return False
    if URL_RE.search(word):
        return False
    letters = [c for c in word if c.isalpha()]
    return 3 <= len(letters) <= 25


# --------------------------------------------------------------------------
# Primitive edits. Each returns None when it cannot apply to `word`.
# --------------------------------------------------------------------------


def _adjacent_key(word: str, rng: np.random.Generator) -> str | None:
    idxs = [i for i, ch in enumerate(word) if ch.lower() in QWERTY_NEIGHBORS]
    if not idxs:
        return None
    i = int(_pick(rng, idxs))
    neigh = QWERTY_NEIGHBORS[word[i].lower()]
    repl = str(_pick(rng, neigh))
    return word[:i] + _match_case(word[i], repl) + word[i + 1 :]


def _random_sub(word: str, rng: np.random.Generator) -> str | None:
    idxs = [i for i, ch in enumerate(word) if ch.isalpha()]
    if not idxs:
        return None
    i = int(_pick(rng, idxs))
    src = word[i]
    repl = _rand_letter(rng, src.isupper())
    while repl.lower() == src.lower():
        repl = _rand_letter(rng, src.isupper())
    return word[:i] + repl + word[i + 1 :]


def _deletion(word: str, rng: np.random.Generator) -> str | None:
    if len(word) < 4:
        return None
    i = int(rng.integers(0, len(word)))
    return word[:i] + word[i + 1 :]


def _insertion(word: str, rng: np.random.Generator) -> str | None:
    i = int(rng.integers(0, len(word) + 1))
    if word and rng.random() < 0.7:
        anchor = word[min(i, len(word) - 1)]
        neigh = QWERTY_NEIGHBORS.get(anchor.lower(), "")
        if neigh:
            ch = _match_case(anchor, str(_pick(rng, neigh)))
            return word[:i] + ch + word[i:]
    ch = _rand_letter(rng, bool(word and word[min(i, len(word) - 1)].isupper()))
    return word[:i] + ch + word[i:]


def _transposition(word: str, rng: np.random.Generator) -> str | None:
    options = [i for i in range(len(word) - 1) if word[i].lower() != word[i + 1].lower()]
    if not options:
        return None
    i = int(_pick(rng, options))
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def _doubling(word: str, rng: np.random.Generator) -> str | None:
    """Double a single consonant, or collapse an existing double.

    "imposible"/"impossible", "tommorow"/"tomorrow", "visitting"/"visiting".
    """
    doubles = list(DOUBLE_RE.finditer(word))
    singles = [
        i
        for i in range(1, len(word) - 1)
        if word[i].lower() in CONSONANTS
        and word[i - 1].lower() != word[i].lower()
        and word[i + 1].lower() != word[i].lower()
    ]
    choices: list[str] = []
    if doubles:
        m = _pick(rng, doubles)
        choices.append(word[: m.start()] + m.group(1) + word[m.end() :])
    if singles:
        i = int(_pick(rng, singles))
        choices.append(word[: i + 1] + word[i] + word[i + 1 :])
    if not choices:
        return None
    return str(_pick(rng, choices))


def _silent_letter(word: str, rng: np.random.Generator) -> str | None:
    """Drop a letter that carries little sound. "rythm", "lenth", "guity"."""
    idxs = [
        i
        for i in range(1, len(word))
        if word[i].lower() in SILENT_DROPPABLE
        and (i + 1 >= len(word) or word[i + 1].lower() in CONSONANTS or word[i].lower() in "eu")
    ]
    if not idxs:
        return None
    i = int(_pick(rng, idxs))
    return word[:i] + word[i + 1 :]


def _vowel_reduction(word: str, rng: np.random.Generator) -> str | None:
    """Respell or drop an unstressed vowel. "Moters", "domitory", "seperate"."""
    idxs = [i for i in range(1, len(word)) if word[i].lower() in VOWELS]
    if not idxs:
        return None
    i = int(_pick(rng, idxs))
    if rng.random() < 0.25 and len(word) > 4:
        return word[:i] + word[i + 1 :]
    others = [v for v in VOWELS if v != word[i].lower()]
    return word[:i] + _match_case(word[i], str(_pick(rng, others))) + word[i + 1 :]


def _phonetic(word: str, rng: np.random.Generator) -> str | None:
    lowered = word.lower()
    applicable: list[tuple[int, str, str]] = []
    for pattern, replacement in PHONETIC_RULES:
        start = lowered.find(pattern)
        while start != -1:
            applicable.append((start, pattern, replacement))
            start = lowered.find(pattern, start + 1)
    if not applicable:
        return None
    start, pattern, replacement = applicable[int(rng.integers(0, len(applicable)))]
    end = start + len(pattern)
    return word[:start] + _match_case(word[start:end], replacement) + word[end:]


_HANDLERS = {
    "phonetic": _phonetic,
    "vowel_reduction": _vowel_reduction,
    "doubling": _doubling,
    "silent_letter": _silent_letter,
    "adjacent_key": _adjacent_key,
    "transposition": _transposition,
    "deletion": _deletion,
    "insertion": _insertion,
    "random_sub": _random_sub,
}

_FALLBACKS = ("adjacent_key", "transposition", "deletion", "insertion", "random_sub")


def choose_corruption_type(rng: np.random.Generator) -> str:
    names = list(CORRUPTION_PROBS)
    probs = np.array([CORRUPTION_PROBS[n] for n in names], dtype=np.float64)
    probs = probs / probs.sum()
    return str(rng.choice(names, p=probs))


def _apply_one(word: str, rng: np.random.Generator, corruption_type: str | None) -> tuple[str, str] | None:
    kind = corruption_type or choose_corruption_type(rng)
    typo = _HANDLERS[kind](word, rng)
    if typo and typo != word:
        return typo, kind
    for fallback in _FALLBACKS:
        typo = _HANDLERS[fallback](word, rng)
        if typo and typo != word:
            return typo, fallback
    return None


def corrupt_word(
    word: str,
    rng: np.random.Generator,
    corruption_type: str | None = None,
    *,
    n_edits: int | None = None,
) -> tuple[str, str]:
    """Return (typo, corruption_type_label). May equal `word` if nothing applied.

    Composes `n_edits` primitive edits (sampled from `N_EDIT_PROBS` when not
    given). The label records every primitive applied, joined by '+'.
    """
    if n_edits is None:
        n_edits = 1 + int(rng.choice(3, p=np.asarray(N_EDIT_PROBS) / np.sum(N_EDIT_PROBS)))
    current = word
    kinds: list[str] = []
    for step in range(n_edits):
        applied = _apply_one(current, rng, corruption_type if step == 0 else None)
        if applied is None:
            break
        current, kind = applied
        kinds.append(kind)
    if not kinds:
        return nfc(word), "none"
    return nfc(current), "+".join(kinds)


def tokenize_sentence(sentence: str) -> list[tuple[int, int, str]]:
    """Return (start, end, token) spans for whitespace-separated tokens."""
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", sentence)]


def eligible_token_indices(tokens: Sequence[tuple[int, int, str]]) -> list[int]:
    return [i for i, (_, _, tok) in enumerate(tokens) if is_eligible_word(tok)]
