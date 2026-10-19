import pytest

from experiment.analysis.answer_bertscore import resolve_gold, run_means


def test_resolve_gold_prefers_row():
    row = {"query_id": "q1", "gold_answers": ["from row"]}
    assert resolve_gold(row, {"q1": ["from index"]}) == ["from row"]


def test_resolve_gold_falls_back_to_index_when_row_missing():
    assert resolve_gold({"query_id": "q1"}, {"q1": ["from index"]}) == ["from index"]
    assert resolve_gold({"query_id": "q1", "gold_answers": []}, {"q1": ["from index"]}) == ["from index"]


def test_resolve_gold_returns_empty_when_unknown():
    assert resolve_gold({"query_id": "q9"}, {}) == []


def test_run_means_averages_each_key():
    scored = [
        {"answer_exact_match": 0.0, "answer_f1": 0.2, "answer_bertscore_p": 0.4,
         "answer_bertscore_r": 0.6, "answer_bertscore_f1": 0.5},
        {"answer_exact_match": 1.0, "answer_f1": 0.4, "answer_bertscore_p": 0.6,
         "answer_bertscore_r": 0.8, "answer_bertscore_f1": 0.7},
    ]
    means = run_means(scored)
    assert means["answer_exact_match"] == pytest.approx(0.5)
    assert means["answer_f1"] == pytest.approx(0.3)
    assert means["answer_bertscore_f1"] == pytest.approx(0.6)


def test_run_means_handles_empty():
    assert run_means([])["answer_bertscore_f1"] == 0.0
