from spelling_reranker.llm_judge import (
    LATENCY_BIN_EDGES_MS,
    build_generative_messages,
    build_messages,
    build_open_messages,
    edit_distance,
    latency_stats,
    parse_choice,
    parse_open_word,
    select_by_edit_distance_and_probability,
)


def test_build_messages_marks_typo_and_numbers_candidates():
    messages = build_messages("I went to the ", "stroe", " yesterday.", ["store", "strove", "stole"])
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "<TYPO>stroe</TYPO>" in user
    assert "I went to the <TYPO>stroe</TYPO> yesterday." in user
    assert "1. store" in user
    assert "2. strove" in user
    assert "3. stole" in user


def test_parse_choice_extracts_in_range_number():
    assert parse_choice("2", 3) == 2
    assert parse_choice("The answer is 3.", 3) == 3
    assert parse_choice("I choose option 1", 5) == 1


def test_parse_choice_rejects_out_of_range_or_missing():
    assert parse_choice("7", 3) is None
    assert parse_choice("none of these", 3) is None
    assert parse_choice("", 3) is None


def test_latency_stats_empty():
    assert latency_stats([]) == {"n": 0}


def test_latency_stats_basic():
    stats = latency_stats([0.1, 0.2, 0.3])
    assert stats["n"] == 3
    assert round(stats["mean_ms"], 1) == 200.0
    assert stats["min_ms"] == 100.0
    assert stats["max_ms"] == 300.0


def test_latency_bin_edges_are_sorted_and_open_ended():
    assert LATENCY_BIN_EDGES_MS == sorted(LATENCY_BIN_EDGES_MS)
    assert LATENCY_BIN_EDGES_MS[-1] == float("inf")


def test_build_open_messages_shows_candidates_as_hint_not_constraint():
    messages = build_open_messages("I went to the ", "stroe", " yesterday.", ["store", "strove"])
    assert messages[0]["role"] == "system"
    assert "not limited to the candidate list" in messages[0]["content"]
    user = messages[1]["content"]
    assert "<TYPO>stroe</TYPO>" in user
    assert "1. store" in user


def test_parse_open_word_extracts_first_word_token():
    assert parse_open_word("store") == "store"
    assert parse_open_word("The word is store.") == "The"
    assert parse_open_word('"store"') == "store"
    assert parse_open_word("well-known") == "well-known"
    assert parse_open_word("don't") == "don't"


def test_parse_open_word_none_when_no_word_found():
    assert parse_open_word("") is None
    assert parse_open_word("123") is None


def test_build_generative_messages_has_no_candidate_list():
    messages = build_generative_messages("I went to the ", "stroe", " yesterday.")
    assert "<TYPO>stroe</TYPO>" in messages[1]["content"]
    assert "Candidates" not in messages[1]["content"]
    assert "1." not in messages[1]["content"]


def test_edit_distance_basic():
    assert edit_distance("store", "store") == 0
    assert edit_distance("store", "stroe") == 2  # transposition = 2 substitutions under Levenshtein
    assert edit_distance("store", "stor") == 1
    assert edit_distance("Store", "store") == 0  # case-insensitive
    assert edit_distance("kitten", "sitting") == 3


def test_select_by_edit_distance_and_probability_prefers_closer_word_at_equal_logprob():
    # edit_distance("stroke", "stroe") == 1 (one insertion); edit_distance("store", "stroe") == 2
    # (a transposition costs two substitutions under plain Levenshtein).
    candidates = [
        {"word": "store", "logprob": -1.0, "n_tokens": 1},
        {"word": "stroke", "logprob": -1.0, "n_tokens": 2},
    ]
    ranked = select_by_edit_distance_and_probability(candidates, "stroe", edit_distance_weight=1.0)
    assert ranked[0]["word"] == "stroke"
    assert ranked[0]["edit_distance"] == 1
    assert ranked[1]["word"] == "store"


def test_select_by_edit_distance_and_probability_lets_probability_override_small_edit_gap():
    candidates = [
        {"word": "close", "logprob": -0.01, "n_tokens": 1},  # far, but very confident
        {"word": "distant", "logprob": -50.0, "n_tokens": 3},  # closer, but very unlikely
    ]
    ranked = select_by_edit_distance_and_probability(candidates, "stroe", edit_distance_weight=1.0)
    assert ranked[0]["word"] == "close"


def test_select_by_edit_distance_and_probability_handles_missing_word():
    candidates = [{"word": None, "logprob": -0.5, "n_tokens": 0}, {"word": "store", "logprob": -3.0, "n_tokens": 1}]
    ranked = select_by_edit_distance_and_probability(candidates, "stroe")
    assert ranked[0]["word"] == "store"
    assert ranked[-1]["word"] is None
