from spelling_reranker.llm_judge_cpu import (
    LATENCY_BIN_EDGES_MS,
    build_generative_messages,
    build_messages,
    build_open_messages,
    build_sentence_messages,
    edit_distance,
    extract_corrected_word,
    latency_stats,
    parse_choice,
    parse_corrected_sentence,
    parse_open_word,
    select_by_edit_distance_and_probability,
    strip_outer_punctuation,
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


def test_select_by_edit_distance_and_probability_drops_candidates_that_echo_the_typo():
    # Regression test: a candidate identical to the typo has edit_distance 0,
    # which used to let it beat a real correction on any near-tied logprob
    # (observed live: gemma-4-E2B-it echoed "ugry" unchanged instead of "ugly"
    # despite "ugly" having the better logprob). Such candidates must be
    # dropped before scoring, not merely outscored.
    candidates = [
        {"word": "ugry", "logprob": -0.73, "n_tokens": 3},  # echoes the typo verbatim
        {"word": "ugly", "logprob": -0.70, "n_tokens": 2},  # the real, slightly more likely correction
    ]
    ranked = select_by_edit_distance_and_probability(candidates, "ugry", edit_distance_weight=1.0)
    assert [c["word"] for c in ranked] == ["ugly"]


def test_select_by_edit_distance_and_probability_case_insensitive_echo_check():
    candidates = [{"word": "Store", "logprob": -0.1, "n_tokens": 1}]
    ranked = select_by_edit_distance_and_probability(candidates, "store")
    assert ranked == []


def test_select_by_edit_distance_and_probability_empty_when_only_echo_candidates():
    candidates = [{"word": "stroe", "logprob": -0.1, "n_tokens": 1}]
    ranked = select_by_edit_distance_and_probability(candidates, "stroe")
    assert ranked == []


def test_build_sentence_messages_shows_candidates_and_asks_for_tagged_sentence():
    messages = build_sentence_messages("I went to the ", "stroe", " yesterday.", ["store", "strove"])
    assert "<corrected_sentence>" in messages[0]["content"]
    user = messages[1]["content"]
    assert "I went to the <TYPO>stroe</TYPO> yesterday." in user
    assert "1. store" in user
    assert "2. strove" in user
    assert "<corrected_sentence>" in user


def test_parse_corrected_sentence_well_formed():
    assert (
        parse_corrected_sentence("<corrected_sentence>I went to the store yesterday.</corrected_sentence>")
        == "I went to the store yesterday."
    )


def test_parse_corrected_sentence_tolerates_missing_or_repeated_opening_tag():
    # Generation budget ran out before the closing tag.
    assert parse_corrected_sentence("<corrected_sentence>I went to the store") == "I went to the store"
    # The malformed "closing" tag some models emit is another opening tag.
    assert (
        parse_corrected_sentence("<corrected_sentence>I went to the store<corrected_sentence>")
        == "I went to the store"
    )


def test_parse_corrected_sentence_none_without_opening_tag():
    assert parse_corrected_sentence("I went to the store yesterday.") is None
    assert parse_corrected_sentence("") is None
    assert parse_corrected_sentence("<corrected_sentence></corrected_sentence>") is None


def test_extract_corrected_word_recovers_single_replacement():
    word, span = extract_corrected_word(
        "I went to the store yesterday.", "I went to the ", "stroe", " yesterday."
    )
    assert (word, span) == ("store", 1)


def test_extract_corrected_word_detects_unchanged_typo():
    word, span = extract_corrected_word(
        "I went to the stroe yesterday.", "I went to the ", "stroe", " yesterday."
    )
    assert (word, span) == ("stroe", 1)


def test_extract_corrected_word_handles_typo_at_sentence_edges():
    assert extract_corrected_word("Store hours are posted.", "", "Stroe", " hours are posted.") == ("Store", 1)
    assert extract_corrected_word("Open the store", "Open the ", "stroe", "") == ("store", 1)


def test_extract_corrected_word_reports_deletion_and_multiword_rewrites():
    assert extract_corrected_word("I went to the yesterday.", "I went to the ", "stroe", " yesterday.") == (
        None,
        0,
    )
    word, span = extract_corrected_word(
        "I went to the corner store yesterday.", "I went to the ", "stroe", " yesterday."
    )
    # The slot held two tokens; the one nearest the typo is the correction,
    # and the span size stays reported so the messy parse is visible.
    assert (word, span) == ("store", 2)


def test_extract_corrected_word_survives_edits_elsewhere_in_the_sentence():
    # The model was told to change nothing else, but when it does anyway the
    # typo slot must still be recovered from the surrounding anchors.
    word, span = extract_corrected_word(
        "I walked to the store yesterday!", "I went to the ", "stroe", " yesterday."
    )
    assert word == "store"
    assert span > 1  # the repunctuated tail cost the right-hand anchor


def test_strip_outer_punctuation_keeps_inner_characters():
    assert strip_outer_punctuation("store.") == "store"
    assert strip_outer_punctuation('"store,"') == "store"
    assert strip_outer_punctuation("don't") == "don't"
    assert strip_outer_punctuation("well-known") == "well-known"
