import json

import pytest

from experiment.analysis.chrf_divergence import (
    build_records,
    is_numeric_answer,
    load_gold_answers,
    pick_threshold,
    summarise,
)


def _write_jsonl(path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _result_row(query_id: str, hit: bool, top1: str) -> dict:
    return {"query_id": query_id, "hit_at_k": hit, "k_results": [{"id": "d", "text": top1, "rank": 1}]}


@pytest.fixture
def run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    queries_path = tmp_path / "queries.jsonl"
    _write_jsonl(queries_path, [
        {"query_id": "num_hit", "gold_answers": ["1997"]},
        {"query_id": "num_miss", "gold_answers": ["1997"]},
        {"query_id": "txt_hit", "gold_answers": ["Cleveland Cavaliers"]},
        {"query_id": "txt_miss", "gold_answers": ["Cleveland Cavaliers"]},
        {"query_id": "no_gold", "gold_answers": []},
    ])
    _write_jsonl(run_dir / "arm_results.jsonl", [
        _result_row("num_hit", True, "the season ended in 1997 after a long run"),
        _result_row("num_miss", False, "unrelated paragraph about geography and rivers"),
        _result_row("txt_hit", True, "he played for the Cleveland Cavaliers for years"),
        _result_row("txt_miss", False, "unrelated paragraph about geography and rivers"),
        _result_row("no_gold", False, "anything at all"),
    ])
    _write_jsonl(run_dir / "_metadata.jsonl", [{"id": "arm"}])
    return run_dir, queries_path


def test_is_numeric_answer():
    assert is_numeric_answer(["1997"])
    assert is_numeric_answer(["1997-03-01"])
    assert not is_numeric_answer(["March 1997"])
    assert not is_numeric_answer(["Ohio"])
    assert not is_numeric_answer([])


def test_load_gold_answers_drops_empty_strings(run):
    _, queries_path = run
    gold = load_gold_answers(queries_path)
    assert gold["num_hit"] == ["1997"]
    assert gold["no_gold"] == []


def test_build_records_skips_rows_without_gold_and_ignores_metadata(run):
    run_dir, queries_path = run
    records = build_records(run_dir, load_gold_answers(queries_path), beta=1.0)
    assert {r["query_id"] for r in records} == {"num_hit", "num_miss", "txt_hit", "txt_miss"}
    assert {r["answer_class"] for r in records if r["query_id"].startswith("num")} == {"numeric"}


def test_pick_threshold_is_mean_chrf_of_hit_rows(run):
    run_dir, queries_path = run
    records = build_records(run_dir, load_gold_answers(queries_path), beta=1.0)
    hit_scores = [r["chrf"] for r in records if r["hit_at_k"]]
    assert pick_threshold(records) == pytest.approx(sum(hit_scores) / len(hit_scores))


def test_summarise_reports_numeric_textual_split(run):
    run_dir, queries_path = run
    records = build_records(run_dir, load_gold_answers(queries_path), beta=1.0)
    summary = summarise(records, threshold=pick_threshold(records))
    assert summary["overall"]["n"] == 4
    assert summary["numeric"]["n"] == 2
    assert summary["textual"]["n"] == 2
    assert summary["numeric"]["miss_count"] == 1
    assert "overstatement_pp" in summary
