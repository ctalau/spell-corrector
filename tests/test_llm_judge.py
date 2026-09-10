from spelling_reranker.llm_judge import (
    LATENCY_BIN_EDGES_MS,
    build_messages,
    latency_stats,
    parse_choice,
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
