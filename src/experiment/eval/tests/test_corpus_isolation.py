import hashlib
import json

import chromadb
import numpy as np
import pytest

from experiment.eval import eval_harness
from experiment.retrievers.faiss import build_index as faiss_build_index
from experiment.util import storage
from experiment.util.storage import init_chroma, load_data, refresh_embedding_db


class FakeEmbeddingModel:
    def __init__(self, model_name: str | None = None, cache_folder: str | None = None, device: str | None = None, dim: int = 8):
        self.model_name = model_name
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 32)
        rng = np.random.default_rng(seed)
        return rng.random(self.dim).astype(np.float32)

    def encode(self, texts):
        if isinstance(texts, str):
            return self._vector(texts)
        return np.array([self._vector(t) for t in texts], dtype=np.float32)


def _write_jsonl(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


ALPHA_DOCS = [{"id": f"a{i}", "text": f"alpha paragraph {i}", "metadata": {"para_index": i}} for i in range(3)]
BETA_DOCS = [{"id": f"b{i}", "text": f"beta paragraph {i}", "metadata": {"para_index": i}} for i in range(3)]



def test_load_data_scoped_path_ignores_sibling_corpus(tmp_path):
    alpha_path = tmp_path / "alpha_corpus.jsonl"
    beta_path = tmp_path / "beta_corpus.jsonl"
    _write_jsonl(alpha_path, ALPHA_DOCS)
    _write_jsonl(beta_path, BETA_DOCS)

    docs = load_data(alpha_path)

    doc_ids = {d["id"] for d in docs}
    assert doc_ids == {f"alpha_corpus.jsonl_a{i}" for i in range(3)}
    assert not any(doc_id.startswith("beta_corpus.jsonl") for doc_id in doc_ids)


def test_load_data_legacy_no_arg_globs_and_warns(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(storage, "ROOT_DATA_PATH", tmp_path)
    alpha_path = tmp_path / "alpha_corpus.jsonl"
    beta_path = tmp_path / "beta_corpus.jsonl"
    _write_jsonl(alpha_path, ALPHA_DOCS)
    _write_jsonl(beta_path, BETA_DOCS)

    docs = load_data()

    doc_ids = {d["id"] for d in docs}
    assert doc_ids == {f"alpha_corpus.jsonl_a{i}" for i in range(3)} | {f"beta_corpus.jsonl_b{i}" for i in range(3)}

    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "alpha_corpus.jsonl" in captured.out and "beta_corpus.jsonl" in captured.out



@pytest.fixture
def shared_chroma_client(monkeypatch):
    client = chromadb.EphemeralClient()
    monkeypatch.setattr(storage.chromadb, "PersistentClient", lambda path: client)
    return client


def test_init_chroma_namespaces_collection_by_source(shared_chroma_client):
    client_a, collection_a = init_chroma("test-model", source="alpha")
    client_b, collection_b = init_chroma("test-model", source="beta")

    assert client_a is client_b is shared_chroma_client
    assert collection_a.name != collection_b.name
    assert collection_a.name == "documents__test-model__alpha"
    assert collection_b.name == "documents__test-model__beta"


def test_source_scoped_collections_never_mix_docs(tmp_path, monkeypatch, shared_chroma_client):
    monkeypatch.setattr(storage, "TRACKING_FILE_PATH", tmp_path / "indexed_docs.json")
    _, collection_a = init_chroma("test-model", source="alpha")
    _, collection_b = init_chroma("test-model", source="beta")
    embedding_model = FakeEmbeddingModel()

    refresh_embedding_db(ALPHA_DOCS, embedding_model, collection_a, model_name="test-model", source="alpha")
    refresh_embedding_db(BETA_DOCS, embedding_model, collection_b, model_name="test-model", source="beta")

    index_a, doc_ids_a = faiss_build_index(collection_a)
    index_b, doc_ids_b = faiss_build_index(collection_b)

    assert set(doc_ids_a) == {"a0", "a1", "a2"}
    assert set(doc_ids_b) == {"b0", "b1", "b2"}
    assert set(doc_ids_a).isdisjoint(doc_ids_b)


def test_refresh_embedding_db_tracking_file_namespaced_by_source(tmp_path, monkeypatch, shared_chroma_client):
    monkeypatch.setattr(storage, "TRACKING_FILE_PATH", tmp_path / "indexed_docs.json")
    _, collection_a = init_chroma("test-model", source="alpha")
    _, collection_b = init_chroma("test-model", source="beta")
    embedding_model = FakeEmbeddingModel()

    refresh_embedding_db(ALPHA_DOCS, embedding_model, collection_a, model_name="test-model", source="alpha")
    refresh_embedding_db(BETA_DOCS, embedding_model, collection_b, model_name="test-model", source="beta")

    alpha_tracking = tmp_path / "indexed_docs__test-model__alpha.json"
    beta_tracking = tmp_path / "indexed_docs__test-model__beta.json"
    assert alpha_tracking.exists()
    assert beta_tracking.exists()

    alpha_indexed = json.loads(alpha_tracking.read_text())
    beta_indexed = json.loads(beta_tracking.read_text())
    assert set(alpha_indexed["ids"]) == {"a0", "a1", "a2"}
    assert set(beta_indexed["ids"]) == {"b0", "b1", "b2"}


def test_build_session_key_distinguishes_query_source(monkeypatch, tmp_path, shared_chroma_client):
    monkeypatch.setattr(storage, "TRACKING_FILE_PATH", tmp_path / "indexed_docs.json")
    monkeypatch.setattr(eval_harness, "SentenceTransformer", FakeEmbeddingModel)

    session_dev = eval_harness.build_session("test-model", "faiss", ALPHA_DOCS, "dev")
    session_train = eval_harness.build_session("test-model", "faiss", BETA_DOCS, "train")

    assert session_dev.collection.name != session_train.collection.name
    assert set(session_dev.doc_ids) == {"a0", "a1", "a2"}
    assert set(session_train.doc_ids) == {"b0", "b1", "b2"}

    session_cache = {}
    key_dev = ("test-model", "faiss", "dev")
    key_train = ("test-model", "faiss", "train")
    assert key_dev != key_train
    session_cache[key_dev] = session_dev
    session_cache[key_train] = session_train
    assert len(session_cache) == 2


ENCODED_DOCS = [
    {"id": "e0", "text": "He served as MP from 1997 to 2001.",
     "metadata": {"entity": "E", "para_index": 0, "date_start": 1997, "date_end": 2001}},
    {"id": "e1", "text": "Born in 1997.",
     "metadata": {"entity": "E", "para_index": 1, "date_start": 1997, "date_end": 1997}},
    {"id": "e2", "text": "No dates here.",
     "metadata": {"entity": "E", "para_index": 2}},
]


def _encoded(encoding):
    from experiment.preprocessing.corpus_encoding import apply_corpus_encoding
    return apply_corpus_encoding(ENCODED_DOCS, encoding).docs


def test_source_encoding_keeps_the_legacy_collection_name(shared_chroma_client):
    _, unsuffixed = init_chroma("test-model", source="alpha")
    _, explicit_source = init_chroma("test-model", source="alpha", encoding="source")
    _, none_encoding = init_chroma("test-model", source="alpha", encoding=None)

    assert unsuffixed.name == "documents__test-model__alpha"
    assert explicit_source.name == "documents__test-model__alpha"
    assert none_encoding.name == "documents__test-model__alpha"


def test_non_source_encoding_gets_its_own_collection(shared_chroma_client):
    _, source_collection = init_chroma("test-model", source="alpha", encoding="source")
    _, numeric_collection = init_chroma("test-model", source="alpha", encoding="numeric")

    assert numeric_collection.name == "documents__test-model__alpha__numeric"
    assert source_collection.name != numeric_collection.name


def test_tracking_file_matches_the_collection_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "TRACKING_FILE_PATH", tmp_path / "indexed_docs.json")

    assert storage._tracking_path_for("m", "dev", "source").name == "indexed_docs__m__dev.json"
    assert storage._tracking_path_for("m", "dev", None).name == "indexed_docs__m__dev.json"
    assert storage._tracking_path_for("m", "dev", "numeric").name == "indexed_docs__m__dev__numeric.json"


def test_two_encodings_store_different_text_for_the_same_doc_id(tmp_path, monkeypatch, shared_chroma_client):
    monkeypatch.setattr(storage, "TRACKING_FILE_PATH", tmp_path / "indexed_docs.json")
    embedding_model = FakeEmbeddingModel()

    _, source_collection = init_chroma("test-model", source="encA", encoding="source")
    _, numeric_collection = init_chroma("test-model", source="encA", encoding="numeric")
    refresh_embedding_db(_encoded("source"), embedding_model, source_collection, "test-model", source="encA", encoding="source")
    refresh_embedding_db(_encoded("numeric"), embedding_model, numeric_collection, "test-model", source="encA", encoding="numeric")

    stored_source = source_collection.get(ids=["e1"])
    stored_numeric = numeric_collection.get(ids=["e1"])

    assert stored_source["documents"][0] == "Born in 1997."
    assert stored_numeric["documents"][0] == "Born in 1997-01."
    assert set(source_collection.get()["ids"]) == set(numeric_collection.get()["ids"])


def test_metadata_is_identical_across_encodings_at_the_storage_layer(tmp_path, monkeypatch, shared_chroma_client):
    monkeypatch.setattr(storage, "TRACKING_FILE_PATH", tmp_path / "indexed_docs.json")
    embedding_model = FakeEmbeddingModel()

    _, source_collection = init_chroma("test-model", source="encA", encoding="source")
    _, delete_collection = init_chroma("test-model", source="encB", encoding="delete")
    refresh_embedding_db(_encoded("source"), embedding_model, source_collection, "test-model", source="encA", encoding="source")
    refresh_embedding_db(_encoded("delete"), embedding_model, delete_collection, "test-model", source="encB", encoding="delete")

    source_meta = source_collection.get(ids=["e0"])["metadatas"][0]
    delete_meta = delete_collection.get(ids=["e0"])["metadatas"][0]

    assert source_meta["date_start"] == delete_meta["date_start"] == 1997
    assert source_meta["date_end"] == delete_meta["date_end"] == 2001
    assert source_collection.get(ids=["e0"])["documents"][0] != delete_collection.get(ids=["e0"])["documents"][0]


def test_re_refreshing_an_encoding_encodes_nothing(tmp_path, monkeypatch, shared_chroma_client):
    monkeypatch.setattr(storage, "TRACKING_FILE_PATH", tmp_path / "indexed_docs.json")

    encode_calls = []

    class CountingModel(FakeEmbeddingModel):
        def encode(self, texts):
            if not isinstance(texts, str):
                encode_calls.append(len(texts))
            return super().encode(texts)

    model = CountingModel()
    _, source_collection = init_chroma("test-model", source="encA", encoding="source")
    _, numeric_collection = init_chroma("test-model", source="encA", encoding="numeric")

    refresh_embedding_db(_encoded("source"), model, source_collection, "test-model", source="encA", encoding="source")
    refresh_embedding_db(_encoded("numeric"), model, numeric_collection, "test-model", source="encA", encoding="numeric")
    refresh_embedding_db(_encoded("source"), model, source_collection, "test-model", source="encA", encoding="source")

    assert encode_calls[0] == 3
    assert encode_calls[1] == 3
    assert len(encode_calls) == 2, f"third refresh re-encoded {encode_calls[2:]} docs; tracking file is not namespaced by encoding"


def test_session_cache_key_distinguishes_corpus_encoding():
    source_key = ("all-MiniLM-L6-v2", "faiss", "dev", "source")
    numeric_key = ("all-MiniLM-L6-v2", "faiss", "dev", "numeric")

    session_cache = {source_key: object()}
    assert numeric_key not in session_cache
    assert source_key != numeric_key


def test_sub_index_caches_are_per_session():
    session_a = eval_harness.RetrievalSession(
        collection=None, embedding_model=None, index=None, doc_ids=[], query_fn=None, build_fn=None)
    session_b = eval_harness.RetrievalSession(
        collection=None, embedding_model=None, index=None, doc_ids=[], query_fn=None, build_fn=None)

    session_a.entity_index_cache["X"] = ("index", ["id"])

    assert session_a.entity_index_cache is not session_b.entity_index_cache
    assert session_b.entity_index_cache == {}
    assert session_a.temporal_index_cache is not session_b.temporal_index_cache
    assert session_a.combined_index_cache is not session_b.combined_index_cache
