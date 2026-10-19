import json
from pathlib import Path

import pytest

from experiment.preprocessing.question_form import (
    QUESTION_FORMS,
    load_question_forms,
    point_expression,
    range_expression,
    render_from_template,
    render_question,
    subject_from_link,
)
from experiment.util.consts import QUERIES_OUTPUT_PATH


@pytest.fixture(scope="module")
def forms():
    return load_question_forms()


@pytest.fixture(scope="module")
def dev_rows():
    if not QUERIES_OUTPUT_PATH.exists():
        pytest.skip(f"no built queries at {QUERIES_OUTPUT_PATH} - run 'python -m experiment.main -b dev'")
    with QUERIES_OUTPUT_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _row(**overrides):
    row = {
        "query_id": "/wiki/Test#P39#0",
        "link": "/wiki/Ian_Gibson_(politician)",
        "relation_code": "P39",
        "relation_meaning": "position held",
        "relation_mode": "accumulate",
        "question_text": "What was the position of Ian Gibson (politician) from May 1997 to May 2001?",
        "time_start_raw": "May 1997",
        "time_end_raw": "May 2001",
    }
    row.update(overrides)
    return row



@pytest.mark.parametrize("link,expected", [
    ("/wiki/Ian_Gibson_(politician)", "Ian Gibson (politician)"),
    ("/wiki/No._5,_1948", "No. 5, 1948"),
    ("/wiki/MS_Princess_Seaways", "MS Princess Seaways"),
    ("/wiki/Fran%C3%A7ois_Mitterrand", "François Mitterrand"),
])
def test_subject_from_link(link, expected):
    assert subject_from_link(link) == expected


def test_range_expression_collapses_degenerate_range_to_a_point():
    assert range_expression("Nov 2003", "Nov 2003") == "in Nov 2003"
    assert range_expression("May 1997", "May 2001") == "from May 1997 to May 2001"


def test_january_bound_renders_as_the_bare_year():
    assert range_expression("May 1996", "Jan 1999") == "from May 1996 to 1999"
    assert point_expression("Jan 2019") == "in 2019"



def test_template_rendering_reproduces_timeqa(dev_rows):
    reproduced = sum(
        1 for row in dev_rows
        if any(render_from_template(row, i) == row["question_text"] for i in range(4))
    )
    rate = reproduced / len(dev_rows)
    assert len(dev_rows) == 3021
    assert rate >= 0.984, f"template fidelity fell to {rate:.4f} ({reproduced}/{len(dev_rows)})"


def test_source_control_is_the_shipped_string_verbatim(dev_rows, forms):
    for row in dev_rows[:200]:
        out = render_question(row, "timeqa_source", forms)
        assert out.text == row["question_text"]
        assert out.applied is False
        assert out.applicable is True



def test_authored_forms_apply_to_every_scored_row(dev_rows, forms):
    scored = [r for r in dev_rows if r["gold_doc_ids"]]
    for form in ("declarative", "keyword", "time_first", "no_relation"):
        applied = sum(1 for r in scored if render_question(r, form, forms).applied)
        assert applied == len(scored), f"{form} left {len(scored) - applied} row(s) unchanged"


def test_timeqa_alt_never_reproduces_the_control(dev_rows, forms):
    scored = [r for r in dev_rows if r["gold_doc_ids"]]
    applicable = [r for r in scored if render_question(r, "timeqa_alt", forms).applicable]
    assert len(applicable) > 2000
    for row in applicable:
        out = render_question(row, "timeqa_alt", forms)
        assert out.text != row["question_text"]
        assert out.applied is True


def test_timeqa_alt_is_inapplicable_when_the_relation_has_one_template(forms):
    relations = {"P1": {"meaning": "x", "mode": "accumulate", "template": ["Who is $1 $4?"]}}
    row = _row(relation_code="P1", question_text="Who is Ian Gibson (politician) from May 1997 to May 2001?")

    out = render_question(row, "timeqa_alt", forms, relations)

    assert out.applicable is False
    assert out.applied is False
    assert out.text == row["question_text"]


def test_relation_mode_selects_the_authored_variant(forms):
    accumulate = render_question(_row(relation_mode="accumulate"), "no_relation", forms).text
    pointwise = render_question(_row(relation_mode="pointwise"), "no_relation", forms).text

    assert accumulate == "Ian Gibson (politician) from May 1997 to May 2001"
    assert pointwise == "Ian Gibson (politician) in May 1997"


def test_na_mode_falls_back_to_the_range_variant(forms):
    out = render_question(_row(relation_mode="n/a"), "no_relation", forms)
    assert out.text == "Ian Gibson (politician) from May 1997 to May 2001"


def test_declarative_and_time_first_are_a_word_order_minimal_pair(forms):
    declarative = render_question(_row(), "declarative", forms).text
    time_first = render_question(_row(), "time_first", forms).text

    assert declarative == "Ian Gibson (politician) position held from May 1997 to May 2001"
    assert time_first == "from May 1997 to May 2001 Ian Gibson (politician) position held"
    assert sorted(declarative.split()) == sorted(time_first.split())


def test_keyword_form_strips_function_words_and_punctuation(forms):
    text = render_question(_row(), "keyword", forms).text
    assert text == "Ian Gibson politician position held May 1997 May 2001"
    assert "(" not in text and "?" not in text


def test_rendered_forms_have_no_double_spacing(dev_rows, forms):
    rendered_forms = [f for f in QUESTION_FORMS if f != "timeqa_source"]
    for row in dev_rows[:300]:
        for form in rendered_forms:
            text = render_question(row, form, forms).text
            assert "  " not in text, f"{form}: {text!r}"
            assert text == text.strip()


def test_source_control_is_untidied(dev_rows, forms):
    dirty = [r for r in dev_rows if "  " in r["question_text"]]
    assert dirty, "expected TimeQA's own text to contain double-spaced questions"
    for row in dirty[:20]:
        assert render_question(row, "timeqa_source", forms).text == row["question_text"]


def test_unknown_form_raises(forms):
    with pytest.raises(ValueError, match="Unknown question_form"):
        render_question(_row(), "not-a-real-form", forms)


def test_substitution_is_a_single_pass(forms):
    row = _row(link="/wiki/$4_(album)", relation_mode="accumulate")
    text = render_question(row, "no_relation", forms).text
    assert text == "$4 (album) from May 1997 to May 2001"
