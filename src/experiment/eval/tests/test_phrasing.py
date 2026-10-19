import pytest

from experiment.preprocessing.phrasing import mutate_question, set_phrasing_cache

"""
tests for mutate_question in phrasing.py
uses the first row of data/queries/timeqa_queries.jsonl once built:

    "What was the position of Ian Gibson (politician) from May 1997 to May 2001?"
    time_start_raw="May 1997", time_end_raw="May 2001"

"language" tests use a fixture cache, not the committed one, so they don't break when it's rebuilt
"""

QUESTION = "What was the position of Ian Gibson (politician) from May 1997 to May 2001?"
START_RAW = "May 1997"
END_RAW = "May 2001"

FIXTURE_CACHE = {
    "May 1997": "May nineteen ninety-seven",
    "May 2001": "May two thousand one",
}


@pytest.fixture
def phrasing_cache():
    set_phrasing_cache(FIXTURE_CACHE)
    yield FIXTURE_CACHE
    set_phrasing_cache(None)


def test_source_is_a_true_noop():
    assert mutate_question(QUESTION, "source", START_RAW, END_RAW) == QUESTION


def test_numeric_substitutes_iso_values():
    mutated = mutate_question(QUESTION, "numeric", START_RAW, END_RAW)
    assert mutated == "What was the position of Ian Gibson (politician) from 1997-05 to 2001-05?"
    assert mutated.startswith("What was the position of Ian Gibson (politician) from ")


def test_numeric_year_precision_expands_to_month_bounds():
    mutated = mutate_question("Team played in 1997 through 2001.", "numeric", "1997", "2001")
    assert mutated == "Team played in 1997-01 through 2001-12."


def test_numeric_year_precision_no_longer_identical_to_source():
    question = "What organization did Altbach join in 1997?"
    assert mutate_question(question, "numeric", "1997", "1997") != question


def test_phrasing_applied_true_for_year_precision_numeric():
    question = "What organization did Altbach join in 1997?"
    mutated = mutate_question(question, "numeric", "1997", "1997")
    assert mutated != question


def test_fuzzy_bucketing_matches_archive_exactly():
    mutated = mutate_question(QUESTION, "fuzzy", START_RAW, END_RAW)
    assert "late 1990s" in mutated
    assert "early 2000s" in mutated
    assert mutated == "What was the position of Ian Gibson (politician) from late 1990s to early 2000s?"


@pytest.mark.parametrize(
    "raw,expected_bucket,expected_decade",
    [
        ("1930", "early", 1930),
        ("1934", "mid", 1930),
        ("1936", "mid", 1930),
        ("1937", "late", 1930),
        ("1997", "late", 1990),
        ("2001", "early", 2000),
    ],
)
def test_fuzzy_bucket_boundaries(raw, expected_bucket, expected_decade):
    question = f"Event happened in {raw}."
    mutated = mutate_question(question, "fuzzy", raw, raw)
    assert f"{expected_bucket} {expected_decade}s" in mutated


def test_language_substitutes_both_tokens(phrasing_cache):
    mutated = mutate_question(QUESTION, "language", START_RAW, END_RAW)
    assert mutated == (
        "What was the position of Ian Gibson (politician) from "
        "May nineteen ninety-seven to May two thousand one?"
    )


def test_language_never_emits_undefined(phrasing_cache):
    assert "UNDEFINED" not in mutate_question(QUESTION, "language", START_RAW, END_RAW)


def test_language_cache_miss_raises_and_names_the_token(phrasing_cache):
    with pytest.raises(KeyError) as excinfo:
        mutate_question(QUESTION, "language", "Jan 1900", END_RAW)
    assert "Jan 1900" in str(excinfo.value)


def test_language_mutates_without_the_range_substring(phrasing_cache):
    question = "What position did they hold in May 1997?"
    mutated = mutate_question(question, "language", START_RAW, START_RAW)
    assert mutated == "What position did they hold in May nineteen ninety-seven?"
    assert mutated != question


def test_delete_removes_both_tokens_and_tidies_the_connective():
    mutated = mutate_question(QUESTION, "delete", START_RAW, END_RAW)
    assert START_RAW not in mutated
    assert END_RAW not in mutated
    assert mutated == "What was the position of Ian Gibson (politician)?"


def test_delete_handles_a_single_date_template():
    mutated = mutate_question("What organization did Altbach join in 1997?", "delete", "1997", "1997")
    assert mutated == "What organization did Altbach join?"


def test_delete_keeps_a_verb_particle():
    mutated = mutate_question("Which school did Marian Pepler go to in 1925?", "delete", "1925", "1925")
    assert mutated == "Which school did Marian Pepler go to?"


def test_delete_keeps_a_trailing_particle_after_a_range():
    mutated = mutate_question(
        "Which team did the player Martin Devaney belong to from 2011 to 2012?", "delete", "2011", "2012",
    )
    assert mutated == "Which team did the player Martin Devaney belong to?"


def test_delete_leaves_no_double_spacing():
    mutated = mutate_question(QUESTION, "delete", START_RAW, END_RAW)
    assert "  " not in mutated
    assert " ?" not in mutated


def test_phrasing_applied_is_false_for_source_and_true_for_language(phrasing_cache):
    assert mutate_question(QUESTION, "source", START_RAW, END_RAW) == QUESTION
    assert mutate_question(QUESTION, "language", START_RAW, END_RAW) != QUESTION


def test_year_fallback_when_the_raw_token_is_absent():
    question = "Which country did Northern Region, Nigeria belong to from Oct 1963 to 1966?"
    assert mutate_question(question, "numeric", "Oct 1963", "Jan 1966") == (
        "Which country did Northern Region, Nigeria belong to from 1963-10 to 1966-01?"
    )
    assert mutate_question(question, "delete", "Oct 1963", "Jan 1966") == (
        "Which country did Northern Region, Nigeria belong to?"
    )


def test_no_op_when_neither_token_nor_year_is_present():
    question = "What position did they hold?"
    assert mutate_question(question, "numeric", "May 1997", "May 2001") == question
    assert mutate_question(question, "delete", "May 1997", "May 2001") == question


def test_substitution_does_not_compound():
    assert mutate_question("What organization did Altbach join in 1?", "numeric", "1", "1") == (
        "What organization did Altbach join in 0001-01?"
    )
    assert mutate_question("Which team did Juan Iturbe play for in 2017?", "numeric", "2017", "2") == (
        "Which team did Juan Iturbe play for in 2017-01?"
    )


def test_substitution_anchors_on_the_range_not_a_title():
    question = "Who owned No. 5, 1948 from 1948 to 1949?"
    assert mutate_question(question, "numeric", "1948", "1949") == "Who owned No. 5, 1948 from 1948-01 to 1949-12?"
    assert mutate_question(question, "delete", "1948", "1949") == "Who owned No. 5, 1948?"


def test_unknown_phrasing_raises():
    with pytest.raises(ValueError):
        mutate_question(QUESTION, "not-a-real-condition", START_RAW, END_RAW)
