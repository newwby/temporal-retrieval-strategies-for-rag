import pytest

from experiment.classes.class_query import EvalQuery
from experiment.eval.metrics import (
    chrf_score, compute_exact, compute_f1, compute_metrics, exact_match_score, f1_score,
    normalize_answer,
)


def _make_query(gold_answers: list[str]) -> EvalQuery:
    return EvalQuery(
        query_id="/wiki/test#0",
        relation_code="P000",
        relation_mode="pointwise",
        time_start="",
        time_end="",
        time_start_raw="",
        time_end_raw="",
        question_text="When did this happen?",
        gold_answers=gold_answers,
    )


def test_normalize_answer_strips_case_punctuation_articles_and_whitespace():
    assert normalize_answer("The Cavaliers!") == "cavaliers"
    assert normalize_answer("cavaliers") == "cavaliers"
    assert normalize_answer("  A, B.  ") == "b"


def test_compute_exact_identical_strings():
    assert compute_exact("Ohio", "Ohio") == 1.0


def test_compute_exact_normalization_equivalence():
    assert compute_exact("The Cavaliers", "cavaliers") == 1.0


def test_compute_exact_different_strings():
    assert compute_exact("Ohio", "Michigan") == 0.0


def test_compute_f1_partial_token_overlap():
    assert compute_f1("Cleveland Cavaliers", "Cavaliers") == pytest.approx(0.6667, abs=1e-3)


def test_compute_f1_no_overlap_returns_zero():
    assert compute_f1("Ohio", "Michigan") == 0.0



def test_compute_f1_both_null_is_one():
    assert compute_f1("", "") == 1.0


def test_compute_f1_gold_null_pred_nonnull_is_zero():
    assert compute_f1("", "Ohio") == 0.0


def test_compute_f1_gold_nonnull_pred_null_is_zero():
    assert compute_f1("Ohio", "") == 0.0


def test_compute_exact_null_handling_matches_f1_table():
    assert compute_exact("", "") == 1.0
    assert compute_exact("", "Ohio") == 0.0
    assert compute_exact("Ohio", "") == 0.0



def test_exact_match_score_multi_gold_picks_max():
    assert exact_match_score(["Ohio", "Columbus"], "columbus") == 1.0


def test_exact_match_score_multi_gold_no_match():
    assert exact_match_score(["Ohio", "Columbus"], "Michigan") == 0.0


def test_f1_score_multi_gold_picks_max():
    assert f1_score(["Ohio", "Cleveland Cavaliers"], "Cavaliers") == pytest.approx(0.6667, abs=1e-3)



def test_exact_match_score_empty_gold_list_and_empty_prediction_is_one():
    assert exact_match_score([], "") == 1.0


def test_f1_score_empty_gold_list_and_empty_prediction_is_one():
    assert f1_score([], "") == 1.0


def test_exact_match_score_empty_gold_list_nonempty_prediction_is_zero():
    assert exact_match_score([], "Ohio") == 0.0


def test_f1_score_empty_gold_list_nonempty_prediction_is_zero():
    assert f1_score([], "Ohio") == 0.0


def test_exact_match_score_nonempty_gold_empty_prediction_is_zero():
    assert exact_match_score(["Ohio", "Columbus"], "") == 0.0


def test_f1_score_nonempty_gold_empty_prediction_is_zero():
    assert f1_score(["Ohio", "Columbus"], "") == 0.0


def test_paper_example_commander_in_chief():
    gold_answers = ["Commander in Chief of the United States Army"]
    prediction = "Commander in Chief"

    assert exact_match_score(gold_answers, prediction) == 0.0
    f1 = f1_score(gold_answers, prediction)
    assert 0.0 < f1 < 1.0
    assert f1 == pytest.approx(2 * 1.0 * (3 / 7) / (1.0 + 3 / 7), abs=1e-6)



def test_chrf_score_exact_match_is_one():
    assert chrf_score(["Cleveland Cavaliers"], "Cleveland Cavaliers") == pytest.approx(1.0)


def test_chrf_score_no_character_overlap_is_near_zero():
    assert chrf_score(["Ohio"], "xyz qrs") == pytest.approx(0.0, abs=1e-6)


def test_chrf_score_multi_reference_picks_max():
    against_first = chrf_score(["Ohio"], "the Cleveland Cavaliers roster")
    against_second = chrf_score(["Cleveland Cavaliers"], "the Cleveland Cavaliers roster")
    combined = chrf_score(["Ohio", "Cleveland Cavaliers"], "the Cleveland Cavaliers roster")
    assert against_second > against_first
    assert combined == pytest.approx(against_second)


def test_chrf_score_empty_gold_answers_returns_zero():
    assert chrf_score([], "some candidate text") == 0.0


def test_chrf_score_empty_candidate_returns_zero():
    assert chrf_score(["1997"], "") == 0.0


def test_chrf_score_all_empty_string_references_return_zero():
    assert chrf_score(["", ""], "some candidate text") == 0.0


def test_chrf_score_beta_parameter_is_forwarded():
    f1_weighted = chrf_score(["Cleveland Cavaliers"], "he later joined the Cavaliers", beta=1.0)
    recall_weighted = chrf_score(["Cleveland Cavaliers"], "he later joined the Cavaliers", beta=3.0)
    assert recall_weighted != f1_weighted


def test_chrf_score_wrong_year_still_scores_above_zero():
    score = chrf_score(["1997"], "some paragraph text containing 1998 instead")
    assert score > 0.0


def test_compute_metrics_leaves_answer_f1_none_without_predicted_answer():
    query = _make_query(gold_answers=["Ohio"])
    result = compute_metrics(query, retrieved_ids=[], predicted_answer=None)
    assert result.answer_f1 is None


def test_compute_metrics_sets_answer_f1_from_predicted_answer():
    query = _make_query(gold_answers=["Cleveland Cavaliers"])
    result = compute_metrics(query, retrieved_ids=[], predicted_answer="Cavaliers")
    assert result.answer_f1 == pytest.approx(f1_score(["Cleveland Cavaliers"], "Cavaliers"))


def test_compute_metrics_leaves_chrf_none_without_predicted_answer():
    query = _make_query(gold_answers=["Ohio"])
    result = compute_metrics(query, retrieved_ids=[], predicted_answer=None)
    assert result.chrf is None


def test_compute_metrics_sets_chrf_from_predicted_answer():
    query = _make_query(gold_answers=["Cleveland Cavaliers"])
    result = compute_metrics(query, retrieved_ids=[], predicted_answer="the Cavaliers roster")
    assert result.chrf == pytest.approx(chrf_score(["Cleveland Cavaliers"], "the Cavaliers roster"))


def test_compute_metrics_leaves_answer_exact_match_none_without_predicted_answer():
    query = _make_query(gold_answers=["Ohio"])
    result = compute_metrics(query, retrieved_ids=[], predicted_answer=None)
    assert result.answer_exact_match is None


def test_compute_metrics_sets_answer_exact_match_from_predicted_answer():
    query = _make_query(gold_answers=["The Cavaliers"])
    result = compute_metrics(query, retrieved_ids=[], predicted_answer="cavaliers")
    assert result.answer_exact_match == pytest.approx(exact_match_score(["The Cavaliers"], "cavaliers"))
    assert result.answer_exact_match == 1.0

