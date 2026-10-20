import hashlib
import json

import chromadb
import numpy as np
import pytest

from experiment.classes.class_query import EvalQuery, ExperimentRunMetadata
from experiment.eval.eval_harness import RetrievalSession, run_experiment_pipeline
from experiment.retrievers.faiss import build_index as faiss_build_index, query as faiss_query
from experiment.util.storage import refresh_embedding_db


ENTITY_A = "Q_Turing"

DOCUMENTS = [
    {"id": "old0", "text": "Turing's early work in 1936.", "metadata": {"entity": ENTITY_A, "date_start": 1936, "date_end": 1936}},
    {"id": "old1", "text": "Wartime cryptanalysis from 1939 to 1945.", "metadata": {"entity": ENTITY_A, "date_start": 1939, "date_end": 1945}},
    {"id": "new0", "text": "Retrospective published in 1990.", "metadata": {"entity": ENTITY_A, "date_start": 1990, "date_end": 1990}},
    {"id": "new1", "text": "A 1995 to 2000 biography project.", "metadata": {"entity": ENTITY_A, "date_start": 1995, "date_end": 2000}},
]


class FakeEmbeddingModel:
    def __init__(self, dim: int = 8):
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 32)
        rng = np.random.default_rng(seed)
        return rng.random(self.dim).astype(np.float32)

    def encode(self, texts):
        if isinstance(texts, str):
            return self._vector(texts)
        return np.array([self._vector(t) for t in texts], dtype=np.float32)


def _make_query(query_id: str, time_start: str, time_end: str, entity: str | None = ENTITY_A) -> EvalQuery:
    return EvalQuery(
        query_id=query_id,
        relation_code="P000",
        relation_mode="pointwise",
        time_start=time_start,
        time_end=time_end,
        time_start_raw=time_start[:4],
        time_end_raw=time_end[:4],
        question_text="When did this happen?",
        entity=entity,
        gold_doc_ids=["old0"],
    )


def _make_metadata(tmp_path, temporal_mode: str = "none", entity_filter: bool = False) -> ExperimentRunMetadata:
    return ExperimentRunMetadata(
        datestamp="test_date",
        timestamp="test_time",
        model_name="test_latency_fields",
        file_stem="test_latency_fields_results",
        k=10,
        entity_filter=entity_filter,
        temporal_mode=temporal_mode,
    )


@pytest.fixture
def session(monkeypatch, tmp_path):
    monkeypatch.setattr("experiment.util.storage.TRACKING_FILE_PATH", tmp_path / "indexed_docs.json")
    monkeypatch.setattr("experiment.classes.class_query.RESULTS_OUTPUT_DIR", tmp_path)

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="test_latency_fields", metadata={"hnsw:space": "cosine"})

    embedding_model = FakeEmbeddingModel()
    refresh_embedding_db(DOCUMENTS, embedding_model, collection, model_name="test_latency_fields")

    index, doc_ids = faiss_build_index(collection)

    return RetrievalSession(
        collection=collection,
        embedding_model=embedding_model,
        index=index,
        doc_ids=doc_ids,
        query_fn=faiss_query,
        build_fn=faiss_build_index,
    )


def test_all_five_fields_present_and_non_none(session, tmp_path):
    query_set = [_make_query("/wiki/test#0", "1990-01-01", "2000-12-31")]
    metadata = _make_metadata(tmp_path, temporal_mode="exclude")

    run_experiment_pipeline(query_set, session, metadata)
    row = _read_jsonl(metadata.get_file_path())[0]

    for field in ("index_resolve_latency", "mutate_latency", "retrieval_latency", "total_latency", "index_cache_hit"):
        assert row[field] is not None, f"{field} was None"


def test_total_latency_is_sum_of_phases(session, tmp_path):
    query_set = [_make_query("/wiki/test#0", "1990-01-01", "2000-12-31")]
    metadata = _make_metadata(tmp_path, temporal_mode="exclude")

    run_experiment_pipeline(query_set, session, metadata)
    row = _read_jsonl(metadata.get_file_path())[0]

    expected = row["index_resolve_latency"] + row["mutate_latency"] + row["retrieval_latency"]
    assert row["total_latency"] == pytest.approx(expected, abs=1e-6)


def test_latency_alias_equals_retrieval_latency(session, tmp_path):
    query_set = [_make_query("/wiki/test#0", "1990-01-01", "2000-12-31")]
    metadata = _make_metadata(tmp_path, temporal_mode="exclude")

    run_experiment_pipeline(query_set, session, metadata)
    row = _read_jsonl(metadata.get_file_path())[0]

    assert row["latency"] == row["retrieval_latency"]


def test_first_query_builds_second_query_reuses_cache(session, tmp_path):
    query_set = [
        _make_query("/wiki/test#0", "1990-01-01", "2000-12-31"),
        _make_query("/wiki/test#1", "1990-01-01", "2000-12-31"),
    ]
    metadata = _make_metadata(tmp_path, temporal_mode="exclude")

    run_experiment_pipeline(query_set, session, metadata)
    rows = _read_jsonl(metadata.get_file_path())

    assert rows[0]["index_cache_hit"] is False
    assert rows[0]["index_resolve_latency"] > 0

    assert rows[1]["index_cache_hit"] is True
    assert rows[1]["index_resolve_latency"] == 0.0

    assert metadata.metrics["index_build_count"] == 1
    assert metadata.metrics["index_build_total_seconds"] == pytest.approx(rows[0]["index_resolve_latency"])
    assert metadata.metrics["index_build_mean_seconds"] == pytest.approx(rows[0]["index_resolve_latency"])
    assert metadata.metrics["index_cache_hit_rate"] == pytest.approx(0.5)


def test_unfiltered_run_reports_all_hits_and_no_builds(session, tmp_path):
    query_set = [
        _make_query("/wiki/test#0", "1990-01-01", "2000-12-31"),
        _make_query("/wiki/test#1", "1936-01-01", "1936-12-31"),
    ]
    metadata = _make_metadata(tmp_path, temporal_mode="none")

    run_experiment_pipeline(query_set, session, metadata)
    rows = _read_jsonl(metadata.get_file_path())

    assert all(row["index_cache_hit"] is True for row in rows)
    assert metadata.metrics["index_build_count"] == 0
    assert metadata.metrics["index_build_total_seconds"] == 0.0
    assert metadata.metrics["index_build_mean_seconds"] == 0.0
    assert metadata.metrics["index_cache_hit_rate"] == pytest.approx(1.0)


def _read_jsonl(path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _latency_row(temporal_mode: str, entity_filter: bool, build_mean: float) -> dict:
    return {
        "temporal_mode": temporal_mode,
        "entity_filter": entity_filter,
        "retriever": "faiss",
        "corpus_encoding": "source",
        "metrics": {"retrieval_latency": 0.01, "index_build_mean_seconds": build_mean},
    }


def test_build_cost_is_charged_to_every_arm_that_needed_the_index():
    from experiment.visualisation.latency_plot import attributed_build_values

    rows = [
        _latency_row("none", False, 0.0),
        _latency_row("exclude", False, 1.31),
        _latency_row("rerank", False, 0.0),
        _latency_row("none", False, 0.0),
        _latency_row("exclude", False, 0.0),
        _latency_row("rerank", False, 0.0),
    ]

    assert attributed_build_values(rows) == [
        (0.0, False),
        (1.31, False),
        (0.0, False),
        (0.0, False),
        (1.31, True),
        (0.0, False),
    ]


def test_rerank_is_never_charged_a_build_cost():
    from experiment.visualisation.latency_plot import attributed_build_values

    rows = [_latency_row("exclude", False, 0.9), _latency_row("rerank", False, 0.0)]
    values = attributed_build_values(rows)

    assert values[1] == (0.0, False)


def test_entity_filter_arms_inherit_their_own_configurations_cost():
    from experiment.visualisation.latency_plot import attributed_build_values

    rows = [
        _latency_row("none", True, 0.008),
        _latency_row("none", True, 0.0),
        _latency_row("none", False, 0.0),
    ]

    assert attributed_build_values(rows) == [(0.008, False), (0.008, True), (0.0, False)]
