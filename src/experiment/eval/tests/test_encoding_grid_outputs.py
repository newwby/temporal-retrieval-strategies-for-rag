import json

import pytest

from experiment.analysis.phrasing_encoding_summary import build_table as phrasing_build_table
from experiment.classes.class_query import ExperimentRunMetadata
from experiment.eval.metadata import MANIFEST_PATH, load_manifest_suite
from experiment.util.run_guards import assess_suite
from experiment.visualisation.grid_plot import build_grid_figure, grid_csv_rows, has_grid_data


def _metadata(**overrides):
    kwargs = dict(datestamp="d", timestamp="t", model_name="all-MiniLM-L6-v2", file_stem="stem")
    kwargs.update(overrides)
    return ExperimentRunMetadata(**kwargs)


def _docs(n):
    return [{"id": f"doc{i}", "text": f"text {i}", "metadata": {"para_index": i}} for i in range(n)]


def _grid_rows():
    rows = []
    for encoding in ("source", "numeric"):
        for phrasing in ("source", "numeric"):
            rows.append({
                "id": f"grid_{encoding}_{phrasing}",
                "corpus_encoding": encoding,
                "phrasing": phrasing,
                "metrics": {"hit_at_k": 0.7, "mrr": 0.19, "corpus_encoding_coverage": 1.0},
            })
    return rows



def test_encoding_grid_expands_to_25_distinct_runs():
    runs = load_manifest_suite("encoding_grid")

    assert len(runs) == 25
    assert len({r.file_stem for r in runs}) == 25
    for run in runs:
        assert run.file_stem.endswith(f"_phrasing-{run.phrasing}")
        assert f"_encoding-{run.corpus_encoding}_" in run.file_stem


def test_existing_suites_are_unaffected_by_the_new_axis():
    runs = load_manifest_suite("phrasing_test")
    manifest_k = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["phrasing_test"]["k"]
    ks = manifest_k if isinstance(manifest_k, list) else [manifest_k]

    assert [r.file_stem for r in runs] == [
        f"phrasing_test_k{k}_all-minilm-l6-v2_phrasing-{p}"
        for k in ks
        for p in ("source", "numeric", "fuzzy", "language", "delete")
    ]
    assert all(r.corpus_encoding == "source" for r in runs)
    assert all(r.question_form == "timeqa_source" for r in runs)


@pytest.mark.parametrize("key,bad", [("corpus_encoding", "iso"), ("question_form", "declarative_v2"), ("phrasing", "words")])
def test_invalid_axis_value_fails_at_manifest_load(tmp_path, key, bad):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"bad_suite": {"model_name": "all-MiniLM-L6-v2", "k": 30, key: bad}}))

    with pytest.raises(ValueError, match=f"Invalid '{key}'"):
        load_manifest_suite("bad_suite", manifest)



def test_embed_cost_sums_across_encodings_rather_than_maxing(tmp_path, monkeypatch):
    from experiment.util import storage
    monkeypatch.setattr(storage, "TRACKING_FILE_PATH", tmp_path / "indexed_docs.json")

    docs = _docs(100)
    runs = [_metadata(corpus_encoding=e) for e in ("source", "numeric", "fuzzy", "language", "delete")]

    summary = assess_suite("encoding_grid", runs, query_set=[None] * 10, docs=docs, query_source="dev")

    assert summary is not None
    assert summary["encodings"] == ["delete", "fuzzy", "language", "numeric", "source"]
    assert summary["docs_to_embed"] == 500
    assert len(summary["docs_to_embed_by_pair"]) == 5


def test_multiple_encodings_alone_makes_a_run_heavy(tmp_path, monkeypatch):
    from experiment.util import storage
    monkeypatch.setattr(storage, "TRACKING_FILE_PATH", tmp_path / "indexed_docs.json")

    runs = [_metadata(corpus_encoding="source"), _metadata(corpus_encoding="numeric")]
    summary = assess_suite("tiny", runs, query_set=[None] * 2, docs=_docs(3), query_source="dev")

    assert summary is not None
    assert len(summary["encodings"]) == 2


def test_single_encoding_suite_keeps_its_old_cost_shape(tmp_path, monkeypatch):
    from experiment.util import storage
    monkeypatch.setattr(storage, "TRACKING_FILE_PATH", tmp_path / "indexed_docs.json")

    summary = assess_suite("baseline", [_metadata()], query_set=[None] * 10, docs=_docs(5), query_source="dev")

    assert summary is None



def test_has_grid_data_only_fires_on_a_real_grid():
    assert has_grid_data(_grid_rows()) is True
    assert has_grid_data([{"corpus_encoding": "source", "phrasing": p} for p in ("source", "numeric")]) is False
    assert has_grid_data([{"corpus_encoding": e, "phrasing": "source"} for e in ("source", "numeric")]) is False
    assert has_grid_data([{"phrasing": p} for p in ("source", "numeric")]) is False


def test_grid_figure_builds_and_csv_is_one_row_per_cell():
    rows = _grid_rows()
    figure = build_grid_figure(rows, "Encoding Grid")
    try:
        assert len(figure.axes) == 2
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)

    csv_rows = grid_csv_rows(rows)
    assert len(csv_rows) == 4
    assert {(r["corpus_encoding"], r["phrasing"]) for r in csv_rows} == {
        ("source", "source"), ("source", "numeric"), ("numeric", "source"), ("numeric", "numeric")
    }


def test_summary_refuses_a_grid_run(tmp_path):
    (tmp_path / "_metadata.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _grid_rows()), encoding="utf-8")

    with pytest.raises(ValueError, match="corpus_encoding grid run"):
        phrasing_build_table(tmp_path)
