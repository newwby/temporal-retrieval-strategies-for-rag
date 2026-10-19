import json

import pytest

from experiment.preprocessing.build_phrasing_cache import (
    collect_tokens,
    format_batch_prompt,
    parse_batch_response,
)
from experiment.preprocessing.phrasing import load_phrasing_cache, resolve_phrasing_cache_path, set_phrasing_cache

"""
tests the batch reply parser in build_phrasing_cache.py and the cache loader in phrasing.py

if 32 tokens come back as 31 lines and the parser pads or cuts, every token after the gap gets the
wrong words, so it raises instead and build_cache() retries in halves
"""


def test_parses_a_well_formed_numbered_response():
    response = "1. May nineteen ninety-seven\n2. two thousand one\n3. twelve June nineteen forty-three"
    assert parse_batch_response(response, 3) == [
        "May nineteen ninety-seven",
        "two thousand one",
        "twelve June nineteen forty-three",
    ]


def test_parses_paren_numbering_and_surrounding_noise():
    response = "Here you go:\n1) nineteen thirty-five\n2) nineteen thirty-seven\n"
    assert parse_batch_response(response, 2) == ["nineteen thirty-five", "nineteen thirty-seven"]


def test_too_few_items_raises():
    response = "1. nineteen thirty-five\n2. nineteen thirty-seven"
    with pytest.raises(ValueError, match="expected 3"):
        parse_batch_response(response, 3)


def test_too_many_items_raises():
    response = "1. a\n2. b\n3. c\n4. d"
    with pytest.raises(ValueError, match="expected 3"):
        parse_batch_response(response, 3)


def test_shuffled_indices_raise():
    response = "1. nineteen thirty-five\n3. nineteen thirty-seven\n2. nineteen thirty-six"
    with pytest.raises(ValueError, match="out of order"):
        parse_batch_response(response, 3)


def test_empty_conversion_raises():
    response = "1. nineteen thirty-five\n2. ."
    with pytest.raises(ValueError, match="empty"):
        parse_batch_response(response, 2)


def test_leading_preposition_is_stripped():
    assert parse_batch_response("1. from May nineteen ninety-seven", 1) == ["May nineteen ninety-seven"]


def test_batch_prompt_numbers_every_token():
    prompt = format_batch_prompt(["May 1997", "2001"])
    assert prompt.endswith("1. May 1997\n2. 2001")


def test_collect_tokens_dedupes_and_skips_unscored_rows(tmp_path):
    queries_path = tmp_path / "queries.jsonl"
    rows = [
        {"time_start_raw": "May 1997", "time_end_raw": "May 2001", "gold_doc_ids": ["a"]},
        {"time_start_raw": "May 1997", "time_end_raw": "1999", "gold_doc_ids": ["b"]},
        {"time_start_raw": "Jan 1800", "time_end_raw": "Jan 1801", "gold_doc_ids": []},
    ]
    queries_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    assert collect_tokens(queries_path) == ["1999", "May 1997", "May 2001"]


def test_cache_path_is_named_for_the_model():
    assert resolve_phrasing_cache_path("llama3.1:8b").name == "phrasing_cache__llama3.1-8b.json"


def test_missing_cache_raises_with_the_build_command(tmp_path):
    set_phrasing_cache(None)
    try:
        with pytest.raises(FileNotFoundError, match="build_phrasing_cache.py"):
            load_phrasing_cache(tmp_path / "absent.json")
    finally:
        set_phrasing_cache(None)


def test_empty_cache_raises(tmp_path):
    cache_path = tmp_path / "phrasing_cache__test.json"
    cache_path.write_text("{}", encoding="utf-8")
    set_phrasing_cache(None)
    try:
        with pytest.raises(ValueError, match="empty"):
            load_phrasing_cache(cache_path)
    finally:
        set_phrasing_cache(None)
