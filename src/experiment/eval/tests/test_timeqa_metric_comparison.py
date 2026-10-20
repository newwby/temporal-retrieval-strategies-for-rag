import json

import pytest

from experiment.analysis.timeqa_metric_comparison import (
    MEASUREMENT_NOTE,
    TIMEQA_REFERENCE_ROWS,
    project_scores,
    resolve_metadata_path,
    write_comparison,
)

RUN_LABEL = "multi_gold_train_test_k5_all-minilm-l6-v2"
SOURCE_METRICS = {
    "hit_at_k": 0.71,
    "mrr": 0.19,
    "answer_f1": 0.1873421,
    "answer_exact_match": 0.0421765,
}


def _write_metadata(tmp_path):
    row = {
        "id": RUN_LABEL,
        "metrics": SOURCE_METRICS,
        "datestamp": "2026_09_03",
        "timestamp": "00-00-00",
        "model_name": "all-MiniLM-L6-v2",
        "k": 5,
        "gold_filter": "multi_gold",
    }
    metadata_path = tmp_path / "_metadata.jsonl"
    metadata_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return metadata_path


def test_project_scores_match_source_metadata_metrics_exactly(tmp_path):
    metadata_path = _write_metadata(tmp_path)
    row = json.loads(metadata_path.read_text(encoding="utf-8").splitlines()[0])

    scores = project_scores(row)

    assert scores["exact_match"] == SOURCE_METRICS["answer_exact_match"]
    assert scores["f1"] == SOURCE_METRICS["answer_f1"]


def test_write_comparison_produces_table_offline(tmp_path):
    metadata_path = _write_metadata(tmp_path)

    out_path = write_comparison(metadata_path)

    assert out_path.exists()
    assert out_path.parent == tmp_path
    text = out_path.read_text(encoding="utf-8")

    assert f"| {RUN_LABEL} | {SOURCE_METRICS['answer_exact_match'] * 100:.1f} | {SOURCE_METRICS['answer_f1'] * 100:.1f} |" in text
    for reference in TIMEQA_REFERENCE_ROWS:
        assert f"| {reference['system']} | {reference['exact_match']:.1f} | {reference['f1']:.1f} |" in text
    assert MEASUREMENT_NOTE in text


def test_resolve_metadata_path_accepts_directory_or_file(tmp_path):
    metadata_path = _write_metadata(tmp_path)

    assert resolve_metadata_path(tmp_path) == metadata_path
    assert resolve_metadata_path(metadata_path) == metadata_path


def test_missing_answer_metrics_raises(tmp_path):
    metadata_path = tmp_path / "_metadata.jsonl"
    metadata_path.write_text(json.dumps({"id": "r", "metrics": {"hit_at_k": 0.5}}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="answer_exact_match"):
        write_comparison(metadata_path)
