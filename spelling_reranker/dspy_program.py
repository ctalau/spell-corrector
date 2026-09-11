"""DSPy programs for single-word spelling correction.

Why DSPy here
-------------
Every answer mode in `spelling_reranker/llm_judge_cpu.py` is a prompt a human
wrote and a human tuned. The measured differences between them are large (81.0%
index vs 87.0% open vs 66.0% strict sentence on the same q4_0 sample), and the
post-mortem of the weakest one found its dominant error was *output format*, not
spelling judgement. That is exactly the kind of thing an optimizer is better at
than a person: it can try demonstrations and instruction wordings against a
score instead of against intuition.

DSPy's contribution is that the prompt stops being a string in the source and
becomes a *program* -- a signature (typed input/output fields) plus demos the
optimizer bootstraps. The rendered text is dumped verbatim by
`scripts/dspy_prompt_search.py` so the result is still readable and auditable.

The hard constraint
-------------------
Optimizers train. The locked benchmark may never be trained on, so every
program here is scored and optimized against `spelling_reranker/dev_set.py`
only. Nothing in this module knows the benchmark exists.

The two signatures, and why these two
-------------------------------------
Both mirror a mode that already has a baseline number, so an optimized prompt
can be compared with a hand-written prompt on like terms:

* :class:`CandidateGuidedCorrection` -- the strongest existing mode ("open"):
  Hunspell's list is a hint, the model answers with a word and may leave the
  list. One output field, one or two decoded tokens, cheapest to optimize.
* :class:`StructuredSentenceCorrection` -- option 1 of the report's "Revised
  prompt options": rewrite the sentence *and* emit the replacement token in its
  own field, then score the field. The rewrite stays as the model's reasoning
  surface, but scoring no longer depends on aligning a reflowed sentence back
  onto pre-tokenised input, which is what produced 15 of 27 strict errors.

A `dspy.ChainOfThought` variant is available behind ``use_cot`` but is **off by
default**. gemma-4 under llama.cpp is served with ``--reasoning off``, so a
chain of thought has to be emitted as ordinary visible output; every reasoning
token is decode time on a CPU-bound box, and the measured cost is ~215ms/token
bf16 (~70ms q4_0). `StructuredSentenceCorrection` already provides a cheap,
bounded reasoning surface (the rewrite) that is at most one sentence long,
which is the same idea with a hard length bound. Turn `use_cot` on only when
the budget is known to allow it.

Metric
------
Two numbers, always reported together:

* **strict** -- NFC-exact string equality with the gold token, the number the
  existing baselines are quoted in;
* **lenient** -- equality after trimming surrounding punctuation and case,
  which is the "punctuation-insensitive" column in the same report.

The optimizer maximizes the strict metric. Reporting both is what keeps a
format artifact from being mistaken for a judgement win (or loss).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from spelling_reranker.byte_encoding import nfc

#: Trimmed from both ends before the lenient comparison. Inner apostrophes and
#: hyphens are kept: `do n't` and `well-known` are real tokens here.
_OUTER_PUNCTUATION = ".,;:!?\"'()[]{}<>`"

DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0


# --------------------------------------------------------------------------
# LM configuration
# --------------------------------------------------------------------------


def require_dspy():
    """Import dspy with an actionable error instead of a bare ImportError."""
    try:
        import dspy  # noqa: PLC0415 - optional dependency, imported on demand
    except ImportError as exc:  # pragma: no cover - exercised by the CLI
        raise ImportError(
            "DSPy is required for prompt optimization but is not installed. "
            "Install it with `pip install dspy` (the current package name; the "
            "old `dspy-ai` distribution is a legacy alias), or "
            "`pip install -e '.[dspy]'` from the repository root."
        ) from exc
    return dspy


def build_lm(
    *,
    base_url: str,
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: str = "llama-cpp-no-key",
    cache: bool = False,
    **kwargs: Any,
):
    """A `dspy.LM` pointed at an OpenAI-compatible server (i.e. llama-server).

    `model` is prefixed with ``openai/`` because that is how LiteLLM (DSPy's
    transport) is told to speak the OpenAI protocol; the name after the slash is
    only an identifier the server echoes back, not a model to download.
    `api_key` is a placeholder -- llama-server does not authenticate by default,
    but the OpenAI client refuses to send a request without one.

    Caching is **off** by default. DSPy's on-disk cache is keyed on the rendered
    prompt, so leaving it on would silently serve an earlier run's completions
    and make a "number of LM calls" count meaningless.
    """
    dspy = require_dspy()
    model_name = model if "/" in model else f"openai/{model}"
    return dspy.LM(
        model_name,
        api_base=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        **kwargs,
    )


def configure(lm) -> None:
    """Install `lm` as the process-wide DSPy LM."""
    dspy = require_dspy()
    dspy.configure(lm=lm)


# --------------------------------------------------------------------------
# Signatures
# --------------------------------------------------------------------------


def _signatures():
    """Signature classes, defined lazily so importing this module never needs dspy."""
    dspy = require_dspy()

    class CandidateGuidedCorrection(dspy.Signature):
        """Correct the single misspelled word marked <TYPO>...</TYPO> in the sentence.

        The spell-checker's suggestions are a hint in its own ranked order, not a
        restriction: if none of them fits the sentence, write the correct word
        yourself. Answer with the replacement for the marked word only.
        """

        sentence: str = dspy.InputField(
            desc="Sentence with exactly one misspelled word wrapped in <TYPO></TYPO>."
        )
        typo: str = dspy.InputField(desc="The misspelled word, as written.")
        candidates: str = dspy.InputField(
            desc="Spell-checker suggestions, best-first, comma-separated. May be empty."
        )
        corrected_word: str = dspy.OutputField(
            desc="The corrected spelling of the marked word: one word, no punctuation, no quotes."
        )

    class StructuredSentenceCorrection(dspy.Signature):
        """Correct the single misspelled word marked <TYPO>...</TYPO> in the sentence.

        First rewrite the sentence with the marked word replaced by its best
        correction, copying every other character exactly as given -- the text is
        pre-tokenised, so spacing around punctuation must be reproduced
        byte-for-byte and no word may be reflowed, recapitalised or
        repunctuated. Then give just the replacement token on its own.
        """

        sentence: str = dspy.InputField(
            desc="Pre-tokenised sentence with one misspelled word wrapped in <TYPO></TYPO>."
        )
        typo: str = dspy.InputField(desc="The misspelled word, as written.")
        candidates: str = dspy.InputField(
            desc="Spell-checker suggestions, best-first, comma-separated. A hint, not a restriction."
        )
        corrected_sentence: str = dspy.OutputField(
            desc="The sentence with the marked word corrected and the markers removed; "
            "everything else character-identical to the input."
        )
        corrected_word: str = dspy.OutputField(
            desc="Only the word that replaced the marked one: one token, no punctuation, no quotes."
        )

    return {
        "candidate_guided": CandidateGuidedCorrection,
        "structured_sentence": StructuredSentenceCorrection,
    }


PROGRAM_NAMES = ("candidate_guided", "structured_sentence")


def build_program(name: str = "candidate_guided", *, use_cot: bool = False):
    """Instantiate one of the programs by name.

    Returns a `dspy.Module` whose forward takes ``sentence``/``typo``/
    ``candidates`` and returns a prediction carrying ``corrected_word``.
    """
    dspy = require_dspy()
    signatures = _signatures()
    if name not in signatures:
        raise ValueError(f"unknown program {name!r}; expected one of {sorted(signatures)}")
    predictor = dspy.ChainOfThought if use_cot else dspy.Predict

    class SpellingCorrector(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.correct = predictor(signatures[name])

        def forward(self, sentence: str, typo: str, candidates: str):
            return self.correct(sentence=sentence, typo=typo, candidates=candidates)

    program = SpellingCorrector()
    program.program_name = name
    program.uses_cot = use_cot
    return program


# --------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------


def marked_sentence(example: Any) -> str:
    """`context_before<TYPO>typo</TYPO>context_after`, the marking every mode uses."""
    return f"{example.context_before}<TYPO>{example.typo}</TYPO>{example.context_after}"


def candidate_string(candidates: Sequence[str]) -> str:
    return ", ".join(candidates) if candidates else "(none)"


def to_dspy_example(example: Any):
    """Convert a `dev_set.DevExample`-shaped object into a `dspy.Example`."""
    dspy = require_dspy()
    return dspy.Example(
        sentence=marked_sentence(example),
        typo=example.typo,
        candidates=candidate_string(example.candidates),
        corrected_word=example.gold,
        example_id=getattr(example, "example_id", ""),
    ).with_inputs("sentence", "typo", "candidates")


def to_dspy_examples(examples: Iterable[Any]) -> list:
    return [to_dspy_example(ex) for ex in examples]


# --------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------


def normalize_word(word: str | None) -> str:
    """First whitespace token, NFC-normalised. Empty string for no answer.

    Small models like to answer `"internet."` or `internet (the network)`; the
    strict metric still rejects the trailing period, but a multi-word ramble is
    cut to its first token here rather than scored as a format failure -- the
    format failures worth counting are the ones where no word comes back at all.
    """
    if not word:
        return ""
    text = str(word).strip()
    if not text:
        return ""
    return nfc(text.split()[0])


def strict_match(gold: str, predicted: str | None) -> bool:
    """NFC-exact equality, the convention the existing baselines are quoted in."""
    return normalize_word(gold) == normalize_word(predicted)


def lenient_match(gold: str, predicted: str | None) -> bool:
    """Equality ignoring surrounding punctuation and case."""
    a = normalize_word(gold).strip(_OUTER_PUNCTUATION).casefold()
    b = normalize_word(predicted).strip(_OUTER_PUNCTUATION).casefold()
    return bool(a) and a == b


def _predicted_word(prediction: Any) -> str | None:
    if prediction is None:
        return None
    word = getattr(prediction, "corrected_word", None)
    if word:
        return str(word)
    # A program that only produced a sentence still gets scored, as a miss --
    # never by guessing a word out of the rewrite, which would hide the
    # format failure the dedicated field exists to prevent.
    return None


def correction_metric(example: Any, prediction: Any, trace: Any = None) -> bool:
    """DSPy metric: strict exact match on the corrected word.

    Returns a bool because DSPy's bootstrapping path (``trace is not None``)
    treats the metric as a pass/fail gate on whether a demonstration is kept.
    """
    return strict_match(getattr(example, "corrected_word", ""), _predicted_word(prediction))


def lenient_correction_metric(example: Any, prediction: Any, trace: Any = None) -> bool:
    """Secondary metric: punctuation- and case-insensitive match."""
    return lenient_match(getattr(example, "corrected_word", ""), _predicted_word(prediction))


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


@dataclass
class EvalResult:
    n: int
    strict: int
    lenient: int
    errors: int
    empty: int
    wall_clock_s: float
    records: list[dict] = field(default_factory=list)

    @property
    def strict_accuracy(self) -> float:
        return self.strict / self.n if self.n else 0.0

    @property
    def lenient_accuracy(self) -> float:
        return self.lenient / self.n if self.n else 0.0

    @property
    def p50_latency_s(self) -> float:
        """Median per-call latency. DSPy's field markup costs decoded tokens on
        top of the answer itself, so this is how that shows up against the
        hand-written prompt's one-word answer."""
        values = sorted(r["latency_s"] for r in self.records if "latency_s" in r)
        if not values:
            return 0.0
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2

    def to_dict(self, *, include_records: bool = False) -> dict:
        out = {
            "n": self.n,
            "strict_correct": self.strict,
            "lenient_correct": self.lenient,
            "strict_accuracy": self.strict_accuracy,
            "lenient_accuracy": self.lenient_accuracy,
            "prediction_errors": self.errors,
            "empty_answers": self.empty,
            "wall_clock_s": self.wall_clock_s,
            "p50_latency_s": self.p50_latency_s,
        }
        if include_records:
            out["records"] = self.records
        return out


def evaluate(
    predict: Callable[[Any], Any],
    examples: Sequence[Any],
    *,
    progress_every: int = 0,
) -> EvalResult:
    """Score `predict` over `examples`, recording strict and lenient accuracy.

    `predict` takes one `dspy.Example`-shaped object and returns a prediction
    with a ``corrected_word`` attribute; a raised exception is counted as a
    wrong answer rather than aborting the run, since one malformed completion
    should not throw away a whole evaluation. The count is reported.
    """
    records: list[dict] = []
    strict = lenient = errors = empty = 0
    t0 = time.perf_counter()
    for i, example in enumerate(examples, 1):
        error: str | None = None
        prediction = None
        t_example = time.perf_counter()
        try:
            prediction = predict(example)
        except Exception as exc:  # noqa: BLE001 - a bad completion is a wrong answer
            error = f"{type(exc).__name__}: {exc}"
            errors += 1
        latency_s = time.perf_counter() - t_example
        word = _predicted_word(prediction)
        if not normalize_word(word):
            empty += 1
        gold = getattr(example, "corrected_word", getattr(example, "gold", ""))
        is_strict = strict_match(gold, word)
        is_lenient = lenient_match(gold, word)
        strict += int(is_strict)
        lenient += int(is_lenient)
        records.append(
            {
                "example_id": getattr(example, "example_id", ""),
                "typo": getattr(example, "typo", ""),
                "gold": gold,
                "predicted": word,
                "strict": is_strict,
                "lenient": is_lenient,
                "latency_s": latency_s,
                "error": error,
            }
        )
        if progress_every and i % progress_every == 0:
            print(f"  [{i}/{len(examples)}] strict={strict / i:.1%}", flush=True)
    return EvalResult(
        n=len(examples),
        strict=strict,
        lenient=lenient,
        errors=errors,
        empty=empty,
        wall_clock_s=time.perf_counter() - t0,
        records=records,
    )


def evaluate_program(program, examples: Sequence[Any], *, progress_every: int = 0) -> EvalResult:
    """Score a DSPy program over `dspy.Example`s using :func:`evaluate`."""

    def predict(example):
        return program(**example.inputs())

    return evaluate(predict, examples, progress_every=progress_every)


# --------------------------------------------------------------------------
# Prompt rendering
# --------------------------------------------------------------------------


def render_messages(program, example) -> list[dict]:
    """The chat messages DSPy would send for `example` -- without calling the LM.

    This is what makes an optimized program auditable: the optimizer's output is
    an instruction string plus a set of bootstrapped demonstrations, and this
    renders both exactly as the adapter will send them.
    """
    dspy = require_dspy()
    predictor = program.predictors()[0]
    adapter = getattr(dspy.settings, "adapter", None) or dspy.ChatAdapter()
    demos = list(getattr(predictor, "demos", []) or [])
    inputs = example.inputs() if hasattr(example, "inputs") else dict(example)
    return adapter.format(signature=predictor.signature, demos=demos, inputs=dict(inputs))


def render_prompt(program, example) -> str:
    """`render_messages` flattened to plain text, for dumping to a file."""
    parts = []
    for message in render_messages(program, example):
        content = message.get("content")
        if isinstance(content, list):  # multimodal payloads render as a list
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        parts.append(f"===== {message.get('role', '?')} =====\n{content}")
    return "\n\n".join(parts) + "\n"


def describe_program(program) -> dict:
    """Machine-readable summary: signature, instructions, and demo count."""
    predictor = program.predictors()[0]
    signature = predictor.signature
    demos = list(getattr(predictor, "demos", []) or [])
    return {
        "program_name": getattr(program, "program_name", type(program).__name__),
        "uses_cot": bool(getattr(program, "uses_cot", False)),
        "signature": str(signature),
        "instructions": signature.instructions,
        "input_fields": list(signature.input_fields),
        "output_fields": list(signature.output_fields),
        "n_demos": len(demos),
        "demos": [dict(d) if not isinstance(d, dict) else d for d in demos],
    }
