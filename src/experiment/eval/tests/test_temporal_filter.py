import hashlib

import chromadb
import numpy as np
import pytest

from experiment.classes.class_query import EvalQuery
from experiment.eval.eval_harness import RetrievalSession, _resolve_index, _query_year_window, _temporal_where, _result_span
from experiment.retrievers.faiss import build_index as faiss_build_index, query as faiss_query
from experiment.util.storage import refresh_embedding_db
from experiment.util.temporal import rerank_by_span


ENTITY_A = "Q_Turing"
OLD_IDS = {"old0", "old1"}
NEW_IDS = {"new0", "new1", "new2"}
ALL_IDS = OLD_IDS | NEW_IDS

DOCUMENTS = [
    {"id": "old0", "text": "Turing's early work in 1936.", "metadata": {"entity": ENTITY_A, "date_start": 1936, "date_end": 1936}},
    {"id": "old1", "text": "Wartime cryptanalysis from 1939 to 1945.", "metadata": {"entity": ENTITY_A, "date_start": 1939, "date_end": 1945}},
    {"id": "new0", "text": "Retrospective published in 1990.", "metadata": {"entity": ENTITY_A, "date_start": 1990, "date_end": 1990}},
    {"id": "new1", "text": "A 1995 to 2000 biography project.", "metadata": {"entity": ENTITY_A, "date_start": 1995, "date_end": 2000}},
    {"id": "new2", "text": "No date mention here at all.", "metadata": {"entity": ENTITY_A}},
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


def _make_query(time_start: str | None, time_end: str | None, entity: str | None = ENTITY_A) -> EvalQuery:
    return EvalQuery(
        query_id="/wiki/test#0",
        relation_code="P000",
        relation_mode="pointwise",
        time_start=time_start,
        time_end=time_end,
        time_start_raw=time_start[:4] if time_start else "",
        time_end_raw=time_end[:4] if time_end else "",
        question_text="When did this happen?",
        entity=entity,
        gold_doc_ids=[],
    )


@pytest.fixture
def session(monkeypatch, tmp_path):
    monkeypatch.setattr("experiment.util.storage.TRACKING_FILE_PATH", tmp_path / "indexed_docs.json")

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="test_temporal_filter", metadata={"hnsw:space": "cosine"})

    embedding_model = FakeEmbeddingModel()
    refresh_embedding_db(DOCUMENTS, embedding_model, collection, model_name="test_temporal_filter")

    index, doc_ids = faiss_build_index(collection)
    assert set(doc_ids) == ALL_IDS

    return RetrievalSession(
        collection=collection,
        embedding_model=embedding_model,
        index=index,
        doc_ids=doc_ids,
        query_fn=faiss_query,
        build_fn=faiss_build_index,
    )


def test_query_year_window():
    assert _query_year_window(_make_query("1990-01-01", "2000-12-31")) == (1990, 2000)
    assert _query_year_window(_make_query(None, None)) is None


def test_temporal_where_shape():
    where = _temporal_where(1990, 2000)
    assert where == {"$and": [{"date_start": {"$lte": 2000}}, {"date_end": {"$gte": 1990}}]}


def test_metadata_reaches_chroma(session):
    stored = session.collection.get(ids=["old0", "new1"], include=["metadatas"])
    metadatas = dict(zip(stored["ids"], stored["metadatas"]))
    assert metadatas["old0"]["date_start"] == 1936
    assert metadatas["new1"]["date_start"] == 1995 and metadatas["new1"]["date_end"] == 2000
    assert "date_start" not in session.collection.get(ids=["new2"], include=["metadatas"])["metadatas"][0]


def test_resolve_index_restricts_to_window(session):
    test_case = _make_query("1990-01-01", "2000-12-31", entity=None)
    filtered = _resolve_index(session, test_case, entity_filter=False, temporal_mode="exclude")
    assert set(filtered.doc_ids) == NEW_IDS - {"new2"}
    assert set(filtered.doc_ids) < ALL_IDS
    assert filtered.cache_hit is False

    unfiltered = _resolve_index(session, test_case, entity_filter=False, temporal_mode="none")
    assert set(unfiltered.doc_ids) == ALL_IDS
    assert unfiltered.cache_hit is True
    assert unfiltered.build_seconds == 0.0

    rerank = _resolve_index(session, test_case, entity_filter=False, temporal_mode="rerank")
    assert set(rerank.doc_ids) == ALL_IDS


def test_temporal_filter_changes_retrieval_results(session):
    test_case = _make_query("1990-01-01", "2000-12-31", entity=None)

    filtered = _resolve_index(session, test_case, entity_filter=False, temporal_mode="exclude")
    unfiltered = _resolve_index(session, test_case, entity_filter=False, temporal_mode="none")

    filtered_results = faiss_query(test_case.question_text, filtered.index, filtered.doc_ids, session.collection, session.embedding_model, k=len(filtered.doc_ids))
    unfiltered_results = faiss_query(test_case.question_text, unfiltered.index, unfiltered.doc_ids, session.collection, session.embedding_model, k=len(unfiltered.doc_ids))

    assert {r["id"] for r in filtered_results} == NEW_IDS - {"new2"}
    assert {r["id"] for r in unfiltered_results} == ALL_IDS


def test_combined_entity_and_temporal_filter(session):
    test_case = _make_query("1990-01-01", "2000-12-31", entity=ENTITY_A)
    combined = _resolve_index(session, test_case, entity_filter=True, temporal_mode="exclude")
    assert set(combined.doc_ids) == NEW_IDS - {"new2"}
    assert combined.cache_hit is False


def test_temporal_filter_noop_when_query_has_no_window(session, capsys):
    test_case = _make_query(None, None, entity=None)
    resolved = _resolve_index(session, test_case, entity_filter=False, temporal_mode="exclude")
    assert set(resolved.doc_ids) == ALL_IDS
    assert resolved.cache_hit is True




def test_rerank_promotes_overlap_above_non_overlap():
    unfiltered = [
        {"rank": 1, "id": "old0", "metadata": {"date_start": 1936, "date_end": 1936}},
        {"rank": 2, "id": "new0", "metadata": {"date_start": 1990, "date_end": 1990}},
    ]
    window = (1985, 1995)
    reranked = rerank_by_span(unfiltered, window, _result_span)
    assert [r["id"] for r in reranked] == ["new0", "old0"]


def test_rerank_never_drops_a_result():
    unfiltered = [
        {"id": "old0", "metadata": {"date_start": 1936, "date_end": 1936}},
        {"id": "new0", "metadata": {"date_start": 1990, "date_end": 1990}},
        {"id": "new2", "metadata": {}},
    ]
    reranked = rerank_by_span(unfiltered, (1985, 1995), _result_span)
    assert {r["id"] for r in reranked} == {r["id"] for r in unfiltered}
    assert len(reranked) == len(unfiltered)


def test_rerank_sorts_undated_between_overlap_and_no_overlap():
    unfiltered = [
        {"id": "old0", "metadata": {"date_start": 1936, "date_end": 1936}},
        {"id": "new2", "metadata": {}},
        {"id": "new0", "metadata": {"date_start": 1990, "date_end": 1990}},
    ]
    reranked = rerank_by_span(unfiltered, (1985, 1995), _result_span)
    assert [r["id"] for r in reranked] == ["new0", "new2", "old0"]


def test_rerank_noop_when_query_has_no_window():
    unfiltered = [{"id": "old0", "metadata": {"date_start": 1936, "date_end": 1936}}]
    assert rerank_by_span(unfiltered, None, _result_span) == unfiltered
