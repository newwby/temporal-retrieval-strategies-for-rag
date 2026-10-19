import hashlib
import json

import chromadb
import numpy as np
import pytest

from experiment.classes.class_query import EvalQuery, ExperimentRunMetadata
from experiment.eval.eval_harness import RetrievalSession, run_experiment_pipeline
from experiment.eval.metrics import chrf_score, exact_match_score, f1_score
from experiment.retrievers.faiss import build_index as faiss_build_index, query as faiss_query
from experiment.util.storage import refresh_embedding_db


ENTITY_A = "Q_Turing"
DOC_TEXT = "Turing played for the Cleveland Cavaliers from 2003 to 2010."

DOCUMENTS = [
    {"id": "doc0", "text": DOC_TEXT, "metadata": {"entity": ENTITY_A}},
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
        entity=ENTITY_A,
        gold_doc_ids=["doc0"],
        gold_answers=gold_answers,
    )


def _make_metadata(tmp_path) -> ExperimentRunMetadata:
    return ExperimentRunMetadata(
        datestamp="test_date",
        timestamp="test_time",
        model_name="test_answer_f1",
        file_stem="test_answer_f1_results",
        k=1,
    )


@pytest.fixture
def session(monkeypatch, tmp_path):
    monkeypatch.setattr("experiment.util.storage.TRACKING_FILE_PATH", tmp_path / "indexed_docs.json")
    monkeypatch.setattr("experiment.classes.class_query.RESULTS_OUTPUT_DIR", tmp_path)

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="test_answer_f1", metadata={"hnsw:space": "cosine"})

    embedding_model = FakeEmbeddingModel()
    refresh_embedding_db(DOCUMENTS, embedding_model, collection, model_name="test_answer_f1")

    index, doc_ids = faiss_build_index(collection)

    return RetrievalSession(
        collection=collection,
        embedding_model=embedding_model,
        index=index,
        doc_ids=doc_ids,
        query_fn=faiss_query,
        build_fn=faiss_build_index,
    )


def test_predicted_answer_is_top1_passage_text(session, tmp_path):
    query_set = [_make_query(gold_answers=["Cleveland Cavaliers"])]
    metadata = _make_metadata(tmp_path)

    run_experiment_pipeline(query_set, session, metadata)
    row = _read_jsonl(metadata.get_file_path())[0]

    assert row["predicted_answer"] == DOC_TEXT


def test_answer_f1_lands_in_output_row(session, tmp_path):
    gold_answers = ["Cleveland Cavaliers"]
    query_set = [_make_query(gold_answers=gold_answers)]
    metadata = _make_metadata(tmp_path)

    run_experiment_pipeline(query_set, session, metadata)
    row = _read_jsonl(metadata.get_file_path())[0]

    assert row["answer_f1"] == pytest.approx(f1_score(gold_answers, DOC_TEXT))
    assert row["answer_f1"] > 0.0


def test_answer_f1_aggregates_into_metadata_metrics(session, tmp_path):
    gold_answers = ["Cleveland Cavaliers"]
    query_set = [_make_query(gold_answers=gold_answers)]
    metadata = _make_metadata(tmp_path)

    run_experiment_pipeline(query_set, session, metadata)

    assert metadata.metrics["answer_f1"] == pytest.approx(f1_score(gold_answers, DOC_TEXT))


def test_chrf_lands_in_output_row_and_aggregates(session, tmp_path):
    gold_answers = ["Cleveland Cavaliers"]
    query_set = [_make_query(gold_answers=gold_answers)]
    metadata = _make_metadata(tmp_path)

    run_experiment_pipeline(query_set, session, metadata)
    row = _read_jsonl(metadata.get_file_path())[0]

    assert row["chrf"] == pytest.approx(chrf_score(gold_answers, DOC_TEXT))
    assert row["chrf"] > 0.0
    assert metadata.metrics["chrf"] == pytest.approx(chrf_score(gold_answers, DOC_TEXT))


def test_chrf_zero_for_unanswerable_gold(session, tmp_path):
    query_set = [_make_query(gold_answers=[])]
    metadata = _make_metadata(tmp_path)

    run_experiment_pipeline(query_set, session, metadata)
    row = _read_jsonl(metadata.get_file_path())[0]

    assert row["chrf"] == 0.0


def test_unanswerable_gold_scores_zero_against_nonempty_prediction(session, tmp_path):
    query_set = [_make_query(gold_answers=[])]
    metadata = _make_metadata(tmp_path)

    run_experiment_pipeline(query_set, session, metadata)
    row = _read_jsonl(metadata.get_file_path())[0]

    assert row["answer_f1"] == 0.0


def _read_jsonl(path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_answer_exact_match_lands_in_output_row_and_metadata_metrics(session, tmp_path):
    gold_answers = ["Cleveland Cavaliers"]
    query_set = [_make_query(gold_answers=gold_answers)]
    metadata = _make_metadata(tmp_path)

    run_experiment_pipeline(query_set, session, metadata)
    row = _read_jsonl(metadata.get_file_path())[0]

    assert row["answer_exact_match"] == pytest.approx(exact_match_score(gold_answers, DOC_TEXT))
    assert metadata.metrics["answer_exact_match"] == pytest.approx(exact_match_score(gold_answers, DOC_TEXT))


def test_gold_answers_land_in_output_row(session, tmp_path):
    query_set = [_make_query(gold_answers=["Cleveland Cavaliers"])]
    metadata = _make_metadata(tmp_path)

    run_experiment_pipeline(query_set, session, metadata)
    row = _read_jsonl(metadata.get_file_path())[0]

    assert row["gold_answers"] == ["Cleveland Cavaliers"]

