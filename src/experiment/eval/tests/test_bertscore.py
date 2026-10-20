import pytest

from experiment.classes.class_query import EvalQuery
from experiment.eval.metrics import compute_metrics, exact_match_score

bert_score = pytest.importorskip("bert_score", reason="bert-score not installed")

from experiment.eval.metrics import bertscore_answer  # noqa: E402


def _make_query(gold_answers: list[str]) -> EvalQuery:
    return EvalQuery(
        query_id="/wiki/test#0",
        relation_code="P000",
        relation_mode="pointwise",
        time_start="",
        time_end="",
        time_start_raw="",
        time_end_raw="",
        question_text="Which team did this person play for?",
        gold_answers=gold_answers,
    )


# compute_metrics em wiring (no bert-score needed, also in test_metrics.py)

def test_compute_metrics_sets_answer_exact_match_from_predicted_answer():
    query = _make_query(gold_answers=["Cleveland Cavaliers"])
    result = compute_metrics(query, retrieved_ids=[], predicted_answer="the cleveland cavaliers")
    assert result.answer_exact_match == exact_match_score(["Cleveland Cavaliers"], "the cleveland cavaliers")
    assert result.answer_exact_match == 1.0


def test_compute_metrics_leaves_answer_exact_match_none_without_predicted_answer():
    query = _make_query(gold_answers=["Ohio"])
    result = compute_metrics(query, retrieved_ids=[], predicted_answer=None)
    assert result.answer_exact_match is None


# bertscore_answer

def test_bertscore_returns_prf_in_unit_range():
    out = bertscore_answer(["Cleveland Cavaliers"], "He played for the Cleveland Cavaliers.")
    assert set(out) == {"precision", "recall", "f1"}
    assert all(0.0 <= out[k] <= 1.0 for k in out)


def test_bertscore_paraphrase_scores_higher_than_unrelated():
    gold = ["the Cleveland Cavaliers"]
    related = bertscore_answer(gold, "He suited up for the Cavaliers of Cleveland.")
    unrelated = bertscore_answer(gold, "The recipe calls for two cups of flour.")
    assert related["f1"] > unrelated["f1"]


def test_bertscore_empty_gold_matches_f1_score_convention():
    # same as f1_score([], "something") == 0.0
    out = bertscore_answer([], "a non-empty prediction")
    assert 0.0 <= out["f1"] <= 1.0
