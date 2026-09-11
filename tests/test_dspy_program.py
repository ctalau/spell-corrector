"""DSPy programs: metric semantics, evaluation bookkeeping, prompt rendering.

Everything here runs without a live LM. The programs are exercised against
`dspy.utils.DummyLM`, which returns canned field values through the real
adapter, so the rendering and parsing path under test is the one a pod would
run -- only the network call is replaced.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spelling_reranker import dspy_program as dp


def _pred(word):
    return SimpleNamespace(corrected_word=word)


def _example(gold: str, **kwargs):
    return SimpleNamespace(corrected_word=gold, typo=kwargs.pop("typo", "teh"), **kwargs)


# --------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------


def test_normalize_word_takes_the_first_token() -> None:
    assert dp.normalize_word("  internet  ") == "internet"
    assert dp.normalize_word("internet (the network)") == "internet"
    assert dp.normalize_word("") == ""
    assert dp.normalize_word(None) == ""


def test_strict_match_is_exact() -> None:
    assert dp.strict_match("internet", "internet")
    assert not dp.strict_match("internet", "internet.")
    assert not dp.strict_match("English", "english")
    assert not dp.strict_match("internet", "")


def test_lenient_match_ignores_surrounding_punctuation_and_case() -> None:
    assert dp.lenient_match("internet", "internet.")
    assert dp.lenient_match("English", "english")
    assert dp.lenient_match("Good", '"good",')
    assert not dp.lenient_match("internet", "internets")
    assert not dp.lenient_match("internet", "")


def test_lenient_match_keeps_inner_punctuation() -> None:
    """`n't` and `well-known` are real tokens in pre-tokenised text."""
    assert dp.lenient_match("n't", "n't")
    assert not dp.lenient_match("well-known", "wellknown")


def test_correction_metric_reads_the_dedicated_field() -> None:
    assert dp.correction_metric(_example("internet"), _pred("internet")) is True
    assert dp.correction_metric(_example("internet"), _pred("internet.")) is False
    assert dp.lenient_correction_metric(_example("internet"), _pred("internet.")) is True


def test_a_missing_word_field_is_a_miss_not_a_salvage() -> None:
    """A program that only wrote a sentence scores zero rather than being parsed.

    The dedicated word field exists precisely so scoring never has to align a
    rewritten sentence back onto the input; quietly falling back to the
    sentence would hide the format failure it is there to prevent.
    """
    sentence_only = SimpleNamespace(corrected_sentence="Now I must buy it on the internet.")
    assert dp.correction_metric(_example("internet"), sentence_only) is False
    assert dp.correction_metric(_example("internet"), None) is False


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def test_evaluate_counts_strict_lenient_errors_and_empties() -> None:
    examples = [_example("internet"), _example("English"), _example("store"), _example("house")]
    answers = {"internet": "internet", "English": "english", "store": None, "house": "boom"}

    def predict(example):
        answer = answers[example.corrected_word]
        if answer == "boom":
            raise RuntimeError("bad completion")
        return _pred(answer)

    result = dp.evaluate(predict, examples)
    assert result.n == 4
    assert result.strict == 1  # only the exact match
    assert result.lenient == 2  # plus the case-only miss
    assert result.errors == 1
    assert result.empty == 2  # the None answer and the raised one
    assert result.strict_accuracy == 0.25
    assert [r["gold"] for r in result.records] == ["internet", "English", "store", "house"]
    assert result.records[3]["error"].startswith("RuntimeError")


def test_eval_result_to_dict_hides_records_by_default() -> None:
    result = dp.evaluate(lambda ex: _pred("internet"), [_example("internet")])
    payload = result.to_dict()
    assert payload["strict_accuracy"] == 1.0
    assert "records" not in payload
    assert "records" in result.to_dict(include_records=True)


# --------------------------------------------------------------------------
# Programs (DummyLM, no network)
# --------------------------------------------------------------------------


def _dev_example(**overrides):
    from spelling_reranker.dev_set import DevExample

    payload = {
        "example_id": "dev-0001",
        "sentence": "Now I must buy it on the interenet .",
        "context_before": "Now I must buy it on the ",
        "typo": "interenet",
        "context_after": " .",
        "gold": "internet",
        "candidates": ("internet", "interment"),
        "gold_index": 0,
        "source": "authentic",
        "corruption_type": "wikipedia_list",
        "edit_distance": 1,
    }
    payload.update(overrides)
    return DevExample(**payload)


def test_marked_sentence_and_candidate_string() -> None:
    ex = _dev_example()
    assert dp.marked_sentence(ex) == "Now I must buy it on the <TYPO>interenet</TYPO> ."
    assert dp.candidate_string(ex.candidates) == "internet, interment"
    assert dp.candidate_string([]) == "(none)"


def test_to_dspy_example_marks_inputs_and_keeps_the_gold() -> None:
    pytest.importorskip("dspy")
    ex = dp.to_dspy_example(_dev_example())
    assert set(ex.inputs().keys()) == {"sentence", "typo", "candidates"}
    assert ex.corrected_word == "internet"
    assert "<TYPO>interenet</TYPO>" in ex.sentence


@pytest.mark.parametrize("name", dp.PROGRAM_NAMES)
def test_programs_round_trip_through_the_adapter(name: str) -> None:
    dspy = pytest.importorskip("dspy")
    from dspy.utils.dummies import DummyLM

    answers = {"corrected_word": "internet"}
    if name == "structured_sentence":
        answers = {"corrected_sentence": "Now I must buy it on the internet .", **answers}
    dspy.configure(lm=DummyLM([answers]))
    program = dp.build_program(name)
    prediction = program(**dp.to_dspy_example(_dev_example()).inputs())
    assert prediction.corrected_word == "internet"


def test_unknown_program_name_is_rejected() -> None:
    pytest.importorskip("dspy")
    with pytest.raises(ValueError, match="unknown program"):
        dp.build_program("does-not-exist")


def test_rendered_prompt_contains_the_instruction_and_the_inputs() -> None:
    pytest.importorskip("dspy")
    program = dp.build_program("candidate_guided")
    example = dp.to_dspy_example(_dev_example())
    text = dp.render_prompt(program, example)
    assert "===== system =====" in text and "===== user =====" in text
    assert "corrected_word" in text
    assert "<TYPO>interenet</TYPO>" in text
    assert "internet, interment" in text
    # No demos yet: the optimizer has not run.
    assert text.count("===== assistant =====") == 0


def test_rendered_prompt_shows_bootstrapped_demos() -> None:
    dspy = pytest.importorskip("dspy")
    program = dp.build_program("candidate_guided")
    program.predictors()[0].demos = [
        dspy.Example(
            sentence="The <TYPO>teh</TYPO> cat sat .",
            typo="teh",
            candidates="the",
            corrected_word="the",
        )
    ]
    text = dp.render_prompt(program, dp.to_dspy_example(_dev_example()))
    assert text.count("===== assistant =====") == 1
    assert "<TYPO>teh</TYPO>" in text


def test_describe_program_is_machine_readable() -> None:
    pytest.importorskip("dspy")
    program = dp.build_program("structured_sentence")
    described = dp.describe_program(program)
    assert described["program_name"] == "structured_sentence"
    assert described["uses_cot"] is False
    assert described["output_fields"] == ["corrected_sentence", "corrected_word"]
    assert "pre-tokenised" in described["instructions"]
    assert described["n_demos"] == 0


def test_build_lm_prefixes_the_openai_provider() -> None:
    pytest.importorskip("dspy")
    lm = dp.build_lm(base_url="http://127.0.0.1:8080/v1", model="gemma-4-e2b-q4", max_tokens=32)
    assert lm.model == "openai/gemma-4-e2b-q4"
    assert lm.kwargs["temperature"] == 0.0  # greedy, as every other mode is
    assert lm.kwargs["max_tokens"] == 32
    assert lm.cache is False  # a cached completion would fake the call count
