import copy

import pytest

from experiment.preprocessing.corpus_encoding import (
    CORPUS_ENCODINGS,
    apply_corpus_encoding,
    mutate_paragraph,
)


def _docs():
    return [
        {"id": "a__para0", "text": "He served as MP from 1997 to 2001, then retired.",
         "metadata": {"entity": "A", "para_index": 0, "date_start": 1997, "date_end": 2001}},
        {"id": "a__para1", "text": "Born in 1997.",
         "metadata": {"entity": "A", "para_index": 1, "date_start": 1997, "date_end": 1997}},
        {"id": "a__para2", "text": "No dates here at all.",
         "metadata": {"entity": "A", "para_index": 2}},
    ]



def test_source_is_an_identity_no_op():
    text = "He served as MP from 1997 to 2001."
    assert mutate_paragraph(text, "source") is text


def test_paragraph_without_dates_is_returned_unchanged():
    text = "No dates here at all."
    for encoding in CORPUS_ENCODINGS:
        assert mutate_paragraph(text, encoding) == text



@pytest.mark.parametrize("text,expected", [
    ("He served as MP from 1997 to 2001.", "He served as MP from 1997-01 to 2001-12."),
    ("Born in 1997.", "Born in 1997-01."),
    ("Signed in May 1997.", "Signed in 1997-05."),
    ("Signed on 5 May 1997.", "Signed on 1997-05-05."),
])
def test_numeric_encoding(text, expected):
    assert mutate_paragraph(text, "numeric") == expected


def test_numeric_uses_the_same_rule_as_the_query_side():
    from experiment.preprocessing.phrasing import numeric_iso
    assert mutate_paragraph("Born in 1997.", "numeric") == f"Born in {numeric_iso('1997', 'min_date')}."


@pytest.mark.parametrize("year,expected", [
    (1990, "early 1990s"), (1993, "early 1990s"), (1994, "mid 1990s"),
    (1996, "mid 1990s"), (1997, "late 1990s"), (1999, "late 1990s"),
])
def test_fuzzy_bucket_boundaries(year, expected):
    assert mutate_paragraph(f"Born in {year}.", "fuzzy") == f"Born in {expected}."


def test_language_encoding_needs_no_cache_and_never_emits_a_sentinel():
    out = mutate_paragraph("Founded in 1066 and refounded in 2010.", "language")
    assert out == "Founded in ten sixty-six and refounded in twenty ten."
    assert "UNDEFINED" not in out


def test_language_handles_day_and_month_precision():
    assert mutate_paragraph("Signed on 5 May 1997.", "language") == "Signed on fifth May nineteen ninety-seven."


def test_delete_removes_the_mention_with_its_preposition():
    assert mutate_paragraph("He served as MP from 1997 to 2001, then retired.", "delete") == \
        "He served as MP, then retired."
    assert mutate_paragraph("Born in 1997.", "delete") == "Born."


def test_delete_leaves_no_double_spacing_or_stranded_punctuation():
    out = mutate_paragraph("Atari was founded in 1972, sold in 1976, and again in 1984.", "delete")
    assert "  " not in out
    assert " ," not in out and " ." not in out
    assert out == out.strip()



def test_every_mention_in_a_paragraph_is_rewritten():
    text = "Atari was founded in 1972, sold in 1976, and again in 1984."
    assert mutate_paragraph(text, "numeric") == \
        "Atari was founded in 1972-01, sold in 1976-01, and again in 1984-01."


def test_a_range_is_consumed_as_one_unit():
    out = mutate_paragraph("Served from 1997 to 2001.", "fuzzy")
    assert out == "Served from late 1990s to early 2000s."
    assert out.count("to") == 1


def test_repeated_token_does_not_compound():
    out = mutate_paragraph("In 1997 and again in 1997.", "numeric")
    assert out == "In 1997-01 and again in 1997-01."
    assert "1997-01-01" not in out


def test_mutation_is_not_idempotent_and_must_not_be_applied_twice():
    once = mutate_paragraph("Born in 1997.", "numeric")
    twice = mutate_paragraph(once, "numeric")
    assert once == "Born in 1997-01."
    assert twice != once


def test_date_re_false_positive_is_pinned_not_fixed():
    assert mutate_paragraph("A population of 1500 people.", "numeric") == "A population of 1500-01 people."


def test_unknown_encoding_raises():
    with pytest.raises(ValueError, match="Unknown corpus encoding"):
        mutate_paragraph("Born in 1997.", "not-a-real-encoding")
    with pytest.raises(ValueError, match="Unknown corpus encoding"):
        apply_corpus_encoding(_docs(), "not-a-real-encoding")



@pytest.mark.parametrize("encoding", CORPUS_ENCODINGS)
def test_metadata_is_identical_across_every_encoding(encoding):
    docs = _docs()
    encoded = apply_corpus_encoding(docs, encoding)
    for original, new in zip(docs, encoded.docs):
        assert new["metadata"] == original["metadata"]
        assert new["metadata"].get("date_start") == original["metadata"].get("date_start")
        assert new["metadata"].get("date_end") == original["metadata"].get("date_end")
        assert new["metadata"] is not original["metadata"]


@pytest.mark.parametrize("encoding", CORPUS_ENCODINGS)
def test_input_docs_are_never_mutated(encoding):
    docs = _docs()
    before = copy.deepcopy(docs)
    apply_corpus_encoding(docs, encoding)
    assert docs == before


def test_ids_are_preserved_so_gold_doc_ids_still_match():
    docs = _docs()
    for encoding in CORPUS_ENCODINGS:
        assert [d["id"] for d in apply_corpus_encoding(docs, encoding).docs] == [d["id"] for d in docs]


def test_coverage_divides_by_dated_paragraphs_only():
    docs = _docs()
    encoded = apply_corpus_encoding(docs, "numeric")
    assert encoded.dated_count == 2
    assert encoded.changed_count == 2
    assert encoded.coverage == 1.0


def test_source_encoding_reports_zero_coverage():
    encoded = apply_corpus_encoding(_docs(), "source")
    assert encoded.changed_count == 0
    assert encoded.coverage == 0.0
    assert [d["text"] for d in encoded.docs] == [d["text"] for d in _docs()]
