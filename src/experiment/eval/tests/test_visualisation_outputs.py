import csv
import json

import matplotlib
matplotlib.use("Agg")

import pytest

from experiment.visualisation import visualise

MODEL_LABEL = "fixture_run_k30_test-model"
# MODEL_LABEL without the suite name and shared model
SHORT_LABEL = "k30"

GOLD_RANKS = {"q0": 1, "q1": 3, "q2": 5, "q3": 11, "q4": None}
EXPECTED_HIT_AT_K = {1: 0.2, 3: 0.4, 5: 0.6, 10: 0.6, 20: 0.8, 30: 0.8}
EXPECTED_MRR_AT_K = {
    1: 0.2, 3: 1.0 / 3 / 5 + 0.2, 5: (1.0 + 1 / 3 + 1 / 5) / 5, 10: (1.0 + 1 / 3 + 1 / 5) / 5,
    20: (1.0 + 1 / 3 + 1 / 5 + 1 / 11) / 5, 30: (1.0 + 1 / 3 + 1 / 5 + 1 / 11) / 5,
}


def _k_results_row(query_id: str, gold_rank: int | None, total: int = 30, gold_doc_ids: list[str] | None = None) -> dict:
    ids = [f"filler_{query_id}_{i}" for i in range(total)]
    if gold_rank is not None:
        ids[gold_rank - 1] = f"gold_{query_id}"
    return {
        "query_id": query_id,
        "gold_doc_ids": gold_doc_ids if gold_doc_ids is not None else [f"gold_{query_id}"],
        "k_results": [{"id": doc_id, "rank": i + 1} for i, doc_id in enumerate(ids)],
    }


def _write_jsonl(path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _metadata_row(with_latency: bool, declared_metrics: list[str] | None = None, gold_filter: str | None = None) -> dict:
    metrics = {"hit_at_k": 0.8, "precision_at_k": 0.0267, "recall_at_k": 0.8, "mrr": 0.3248}
    if with_latency:
        metrics.update({
            "retrieval_latency": 0.012, "index_resolve_latency": 0.03, "mutate_latency": 0.0001,
            "total_latency": 0.0421, "index_build_count": 2, "index_build_total_seconds": 0.06,
            "index_build_mean_seconds": 0.03, "index_cache_hit_rate": 0.5,
        })
    return {
        "id": MODEL_LABEL, "metrics": metrics, "datestamp": "2026_01_01", "timestamp": "00-00-00",
        "model_name": "test-model", "k": 30, "retriever": "faiss", "entity_filter": False,
        "temporal_mode": "none", "phrasing": "source", "invalid_reason": None,
        "declared_metrics": declared_metrics or [], "gold_filter": gold_filter,
    }


def _write_fixture(tmp_path, with_latency: bool, declared_metrics=None, gold_filter=None, result_rows=None):
    metadata_path = tmp_path / "_metadata.jsonl"
    _write_jsonl(metadata_path, [_metadata_row(with_latency, declared_metrics, gold_filter)])
    if result_rows is None:
        result_rows = [_k_results_row(query_id, rank) for query_id, rank in GOLD_RANKS.items()]
    _write_jsonl(tmp_path / f"{MODEL_LABEL}.jsonl", result_rows)
    return metadata_path


def test_default_visualisation_writes_expected_files_and_nowhere_else(tmp_path):
    metadata_path = _write_fixture(tmp_path, with_latency=True)

    written = visualise.main(metadata_path)

    for key in ("metrics_png", "metrics_csv", "latency_png", "latency_csv", "k_sweep_png", "k_sweep_csv", "table_png", "table_md"):
        assert written[key].exists(), f"{key} was not written"
        assert written[key].parent == tmp_path, f"{key} was written outside the run directory"

    assert not (visualise.VISUALISATION_DIR / "2026_01_01").exists()


def test_missing_latency_fields_skips_latency_figure_without_raising(tmp_path):
    metadata_path = _write_fixture(tmp_path, with_latency=False)

    written = visualise.main(metadata_path)

    assert "latency_png" not in written
    assert "latency_csv" not in written
    assert written["metrics_png"].exists()


def test_metrics_csv_matches_plotted_values(tmp_path):
    metadata_path = _write_fixture(tmp_path, with_latency=True)

    written = visualise.main(metadata_path)

    with written["metrics_csv"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["run"] == SHORT_LABEL
    assert float(rows[0]["hit_at_k"]) == 0.8
    assert float(rows[0]["mrr"]) == 0.3248


def test_k_sweep_reproduces_known_curve(tmp_path):
    metadata_path = _write_fixture(tmp_path, with_latency=True)

    written = visualise.main(metadata_path)

    with written["k_sweep_csv"].open(newline="", encoding="utf-8") as f:
        rows = {int(row["k"]): row for row in csv.DictReader(f)}

    assert set(rows) == set(EXPECTED_HIT_AT_K)
    for k, expected_hit in EXPECTED_HIT_AT_K.items():
        assert float(rows[k]["hit_at_k"]) == expected_hit, f"hit_at_k mismatch at k={k}"
        assert float(rows[k]["mrr"]) == pytest.approx(EXPECTED_MRR_AT_K[k], abs=1e-6)


def test_declared_metrics_overrides_inferred_metric_keys(tmp_path):
    metadata_path = _write_fixture(tmp_path, with_latency=False, declared_metrics=["hit_at_k", "mrr"])

    written = visualise.main(metadata_path)

    with written["metrics_csv"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert set(rows[0]) == {"run", "hit_at_k", "mrr"}


def test_missing_declared_metrics_falls_back_to_inference(tmp_path):
    metadata_path = _write_fixture(tmp_path, with_latency=False)

    written = visualise.main(metadata_path)

    with written["metrics_csv"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert set(rows[0]) == {"run", "hit_at_k", "precision_at_k", "recall_at_k", "mrr"}


def test_gold_filter_multi_gold_recomputes_from_subset(tmp_path):
    result_rows = [
        _k_results_row("q0", gold_rank=1, gold_doc_ids=["gold_q0", "gold_q0_b"]),
        _k_results_row("q1", gold_rank=None, gold_doc_ids=["gold_q1", "gold_q1_b"]),
        _k_results_row("q2", gold_rank=1, gold_doc_ids=["gold_q2"]),
    ]
    metadata_path = _write_fixture(
        tmp_path, with_latency=False, declared_metrics=["hit_at_k", "mrr"],
        gold_filter="multi_gold", result_rows=result_rows,
    )

    written = visualise.main(metadata_path)

    with written["metrics_csv"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert float(rows[0]["hit_at_k"]) == 0.5
    assert float(rows[0]["mrr"]) == 0.5
    assert "(multi-gold, n=2)" in rows[0]["run"]


def test_gold_filter_multi_gold_skips_k_sweep_and_grid_figures(tmp_path):
    result_rows = [
        _k_results_row("q0", gold_rank=1, gold_doc_ids=["gold_q0", "gold_q0_b"]),
        _k_results_row("q1", gold_rank=3, gold_doc_ids=["gold_q1", "gold_q1_b"]),
    ]
    metadata_path = _write_fixture(tmp_path, with_latency=False, gold_filter="multi_gold", result_rows=result_rows)

    written = visualise.main(metadata_path)

    assert "k_sweep_png" not in written
    assert "k_sweep_csv" not in written
    assert "grid_png" not in written
    assert "grid_csv" not in written


def test_gold_filter_preserves_latency_fields(tmp_path):
    result_rows = [
        _k_results_row("q0", gold_rank=1, gold_doc_ids=["gold_q0", "gold_q0_b"]),
        _k_results_row("q1", gold_rank=1, gold_doc_ids=["gold_q1", "gold_q1_b"]),
    ]
    metadata_path = _write_fixture(tmp_path, with_latency=True, gold_filter="multi_gold", result_rows=result_rows)

    written = visualise.main(metadata_path)

    assert "latency_png" in written, "gold-filtering must not wipe out latency fields"


def test_labels_drop_the_suite_name_and_a_shared_model(tmp_path):
    metadata_path = _write_fixture(tmp_path, with_latency=False)

    written = visualise.main(metadata_path)

    with written["metrics_csv"].open(newline="", encoding="utf-8") as f:
        runs = [row["run"] for row in csv.DictReader(f)]

    assert runs == [SHORT_LABEL]
    assert "fixture_run" not in runs[0]
    assert "test-model" not in runs[0]


def test_labels_keep_the_model_when_it_varies_between_runs(tmp_path):
    rows = []
    for model in ("model-a", "model-b"):
        row = _metadata_row(with_latency=False)
        row["id"] = f"sweep_k30_{model}"
        row["model_name"] = model
        rows.append(row)
    metadata_path = tmp_path / "_metadata.jsonl"
    _write_jsonl(metadata_path, rows)
    for model in ("model-a", "model-b"):
        _write_jsonl(tmp_path / f"sweep_k30_{model}.jsonl",
                     [_k_results_row(query_id, rank) for query_id, rank in GOLD_RANKS.items()])

    written = visualise.main(metadata_path)

    with written["metrics_csv"].open(newline="", encoding="utf-8") as f:
        runs = [row["run"] for row in csv.DictReader(f)]

    assert runs == ["k30_model-a", "k30_model-b"]


def test_latency_figure_omits_index_build_panel_when_cost_is_nil(tmp_path):
    row = _metadata_row(with_latency=True)
    row["metrics"].update({"index_build_count": 0, "index_build_total_seconds": 0,
                           "index_build_mean_seconds": 0.0, "index_cache_hit_rate": 1.0})
    metadata_path = tmp_path / "_metadata.jsonl"
    _write_jsonl(metadata_path, [row])
    _write_jsonl(tmp_path / f"{MODEL_LABEL}.jsonl",
                 [_k_results_row(query_id, rank) for query_id, rank in GOLD_RANKS.items()])

    written = visualise.main(metadata_path)

    assert written["latency_png"].exists()
    assert written["latency_csv"].exists()

