import hashlib
import json

import chromadb
import numpy as np
import pytest

from experiment.classes.class_query import EvalQuery
from experiment.eval.eval_harness import RetrievalSession, _resolve_index
from experiment.retrievers.faiss import build_index as faiss_build_index, query as faiss_query
from experiment.util.storage import refresh_embedding_db


ENTITY_A = "Q_Turing"
ENTITY_B = "Q_Einstein"

DOCUMENTS = [
    {"id": "a0", "text": "Turing paragraph zero about computation.", "metadata": {"entity": ENTITY_A, "link": "/wiki/Alan_Turing", "para_index": 0}},
    {"id": "a1", "text": "Turing paragraph one about the Bombe.", "metadata": {"entity": ENTITY_A, "link": "/wiki/Alan_Turing", "para_index": 1}},
    {"id": "a2", "text": "Turing paragraph two about Manchester.", "metadata": {"entity": ENTITY_A, "link": "/wiki/Alan_Turing", "para_index": 2}},
    {"id": "b0", "text": "Einstein paragraph zero about relativity.", "metadata": {"entity": ENTITY_B, "link": "/wiki/Albert_Einstein", "para_index": 0}},
    {"id": "b1", "text": "Einstein paragraph one about Zurich.", "metadata": {"entity": ENTITY_B, "link": "/wiki/Albert_Einstein", "para_index": 1}},
]

ENTITY_A_IDS = {d["id"] for d in DOCUMENTS if d["metadata"]["entity"] == ENTITY_A}
ALL_IDS = {d["id"] for d in DOCUMENTS}


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


def _make_query(entity: str) -> EvalQuery:
    return EvalQuery(
        query_id=f"/wiki/test#{entity}#0",
        relation_code="P000",
        relation_mode="pointwise",
        time_start="2000-01-01",
        time_end="2000-01-01",
        time_start_raw="2000",
        time_end_raw="2000",
        question_text="What did they work on?",
        entity=entity,
        gold_doc_ids=[],
    )


@pytest.fixture
def session(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "experiment.util.storage.TRACKING_FILE_PATH", tmp_path / "indexed_docs.json"
    )

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="test_entity_filter", metadata={"hnsw:space": "cosine"})

    embedding_model = FakeEmbeddingModel()
    refresh_embedding_db(DOCUMENTS, embedding_model, collection, model_name="test_entity_filter")

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


def test_metadata_reaches_chroma(session):
    stored = session.collection.get(ids=["a0", "b0"], include=["metadatas"])
    metadatas = dict(zip(stored["ids"], stored["metadatas"]))
    assert metadatas["a0"]["entity"] == ENTITY_A
    assert metadatas["b0"]["entity"] == ENTITY_B


def test_resolve_index_restricts_to_entity(session):
    test_case = _make_query(ENTITY_A)
    resolved = _resolve_index(session, test_case, entity_filter=True)

    filtered_ids = set(resolved.doc_ids)
    assert filtered_ids == ENTITY_A_IDS
    assert filtered_ids < ALL_IDS
    assert resolved.cache_hit is False
    assert resolved.build_seconds >= 0.0

    full = _resolve_index(session, test_case, entity_filter=False)
    assert set(full.doc_ids) == ALL_IDS
    assert full.index is session.index
    assert full.cache_hit is True
    assert full.build_seconds == 0.0


def test_resolve_index_no_fallback_warning(session, capsys):
    test_case = _make_query(ENTITY_A)
    _resolve_index(session, test_case, entity_filter=True)
    captured = capsys.readouterr()
    assert "falling back to unfiltered index" not in captured.out


def test_entity_filter_changes_retrieval_results(session):
    test_case = _make_query(ENTITY_A)

    filtered = _resolve_index(session, test_case, entity_filter=True)
    unfiltered = _resolve_index(session, test_case, entity_filter=False)
    filtered_index, filtered_doc_ids = filtered.index, filtered.doc_ids
    unfiltered_index, unfiltered_doc_ids = unfiltered.index, unfiltered.doc_ids

    filtered_results = faiss_query(
        test_case.question_text, filtered_index, filtered_doc_ids, session.collection, session.embedding_model, k=len(filtered_doc_ids)
    )
    unfiltered_results = faiss_query(
        test_case.question_text, unfiltered_index, unfiltered_doc_ids, session.collection, session.embedding_model, k=len(unfiltered_doc_ids)
    )

    filtered_result_ids = {r["id"] for r in filtered_results}
    unfiltered_result_ids = {r["id"] for r in unfiltered_results}

    assert filtered_result_ids == ENTITY_A_IDS
    assert unfiltered_result_ids == ALL_IDS
    assert filtered_result_ids != unfiltered_result_ids


def test_stale_text_only_hash_self_heals(monkeypatch, tmp_path):
    tracking_path = tmp_path / "indexed_docs.json"
    monkeypatch.setattr("experiment.util.storage.TRACKING_FILE_PATH", tracking_path)

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="test_self_heal", metadata={"hnsw:space": "cosine"})
    embedding_model = FakeEmbeddingModel()

    doc = DOCUMENTS[0]

    old_style_hash = hashlib.md5(doc["text"].encode()).hexdigest()
    collection.upsert(ids=[doc["id"]], embeddings=[embedding_model.encode(doc["text"]).tolist()], documents=[doc["text"]])
    tracking_path.write_text(json.dumps({"ids": [doc["id"]], "checksums": {doc["id"]: old_style_hash}}), encoding="utf-8")

    before = collection.get(ids=[doc["id"]], include=["metadatas"])
    assert not before["metadatas"][0]

    refresh_embedding_db([doc], embedding_model, collection, model_name=None)

    after = collection.get(ids=[doc["id"]], include=["metadatas"])
    assert after["metadatas"][0].get("entity") == ENTITY_A

    updated = json.loads(tracking_path.read_text(encoding="utf-8"))
    assert updated["checksums"][doc["id"]] != old_style_hash


def test_entity_filter_cache_reused(session, capsys):
    capsys.readouterr()
    test_case = _make_query(ENTITY_A)
    first = _resolve_index(session, test_case, entity_filter=True)
    assert ENTITY_A in session.entity_index_cache
    second = _resolve_index(session, test_case, entity_filter=True)
    assert first.index is second.index
    assert first.doc_ids is second.doc_ids
    assert capsys.readouterr().out.count("Built FAISS index") == 1

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.build_seconds == 0.0
