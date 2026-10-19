from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Callable
from sentence_transformers import SentenceTransformer





from experiment.classes.class_query import EvalQuery, ExperimentRunMetadata
from experiment.eval.metadata import METADATA_RECORD, METADATA_WRITE_PATH, get_suite_query_source, load_manifest_suite
from experiment.eval.metrics import compute_metrics, eval_metrics
from experiment.preprocessing.build_corpus import resolve_corpus_output_path
from experiment.preprocessing.corpus_encoding import EncodedCorpus, apply_corpus_encoding
from experiment.preprocessing.build_queries import resolve_queries_output_path
from experiment.preprocessing.phrasing import mutate_question, require_phrasing_cache
from experiment.preprocessing.question_form import load_question_forms, render_question
from experiment.retrievers.registry import RETRIEVER_REGISTRY
from experiment.util.consts import (
    CORPUS_ENCODING_COVERAGE_MIN, DEVICE, EVAL_QUERY_K, PHRASING_COVERAGE_MIN,
    QUERIES_OUTPUT_PATH, QUESTION_FORM_COVERAGE_MIN,
)
from experiment.util.run_guards import warn_if_heavy
from experiment.util.storage import (
    _tracking_path_for, init_chroma, load_data, refresh_embedding_db, save_rows_to_disk,
)
from experiment.util.temporal import rerank_by_span


def load_all_queries(path: Path = QUERIES_OUTPUT_PATH) -> list[EvalQuery]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows_by_query_id = {row["query_id"]: row for row in rows if row["gold_doc_ids"]}

    test_queries = []
    skipped = 0
    for row in rows_by_query_id.values():
        if row.get("relation_mode") is None:
            skipped += 1
            print(
                f"Warning: no relation_mode for relation_code {row['relation_code']!r} "
                f"(absent from relations.json at build time), skipping query_id {row['query_id']!r}"
            )
            continue

        test_queries.append(EvalQuery.row_to_structured_query(row))

    if skipped:
        print(f"Skipped {skipped} row(s) with no relation_mode - rebuild queries if this is unexpected.")

    return test_queries


@dataclass
class RetrievalSession:
    collection: object
    embedding_model: SentenceTransformer
    index: object
    doc_ids: list[str]
    query_fn: Callable
    build_fn: Callable
    entity_index_cache: dict[str, tuple] = field(default_factory=dict)
    temporal_index_cache: dict[tuple[int, int], tuple] = field(default_factory=dict)
    combined_index_cache: dict[tuple, tuple] = field(default_factory=dict)


_embedding_models: dict[str, SentenceTransformer] = {}


def _load_embedding_model(model_name: str) -> SentenceTransformer:
    if model_name not in _embedding_models:
        print(f"Loading {model_name!r} on device={DEVICE!r}")
        _embedding_models[model_name] = SentenceTransformer(model_name, cache_folder='./models', device=DEVICE)
    return _embedding_models[model_name]


def build_session(model_name: str, retriever_key: str, docs: list[dict], query_source: str | None = None,
                  corpus_encoding: str | None = None) -> RetrievalSession:
    retriever = RETRIEVER_REGISTRY[retriever_key]

    client, collection = init_chroma(model_name, source=query_source, encoding=corpus_encoding)
    embedding_model = _load_embedding_model(model_name)
    refresh_embedding_db(docs, embedding_model, collection, model_name, source=query_source, encoding=corpus_encoding)

    stored = collection.count()
    if stored != len(docs):
        raise RuntimeError(
            f"Collection {collection.name!r} holds {stored} document(s) but {len(docs)} were supplied - "
            f"the tracking file and the collection disagree. Delete "
            f"{_tracking_path_for(model_name, query_source, corpus_encoding)} and re-run, or use -c."
        )

    index, doc_ids = retriever["build"](collection)

    return RetrievalSession(
        collection=collection,
        embedding_model=embedding_model,
        index=index,
        doc_ids=doc_ids,
        query_fn=retriever["query"],
        build_fn=retriever["build"],
    )


def _form_row(test_case: EvalQuery) -> dict:
    return {
        "question_text": test_case.question_text,
        "link": test_case.link,
        "relation_code": test_case.relation_code,
        "relation_mode": test_case.relation_mode,
        "relation_meaning": test_case.relation_meaning,
        "time_start_raw": test_case.time_start_raw,
        "time_end_raw": test_case.time_end_raw,
    }


def _query_year_window(test_case: EvalQuery) -> tuple[int, int] | None:
    if not test_case.time_start or not test_case.time_end:
        return None
    try:
        return int(test_case.time_start[:4]), int(test_case.time_end[:4])
    except (TypeError, ValueError):
        return None


def _temporal_where(start_year: int, end_year: int) -> dict:
    return {"$and": [
        {"date_start": {"$lte": end_year}},
        {"date_end": {"$gte": start_year}},
    ]}


def _result_span(result: dict) -> tuple[int, int] | None:
    metadata = result.get("metadata") or {}
    if "date_start" not in metadata:
        return None
    return metadata["date_start"], metadata["date_end"]


@dataclass
class ResolvedIndex:
    index: object
    doc_ids: list[str]
    cache_hit: bool
    build_seconds: float


def _resolve_index(session: RetrievalSession, test_case: EvalQuery, entity_filter: bool, temporal_mode: str = "none") -> ResolvedIndex:
    use_entity = bool(entity_filter and test_case.entity)
    use_temporal = temporal_mode == "exclude"
    time_window = _query_year_window(test_case) if use_temporal else None
    use_temporal = bool(use_temporal and time_window is not None)

    if not use_entity and not use_temporal:
        return ResolvedIndex(session.index, session.doc_ids, cache_hit=True, build_seconds=0.0)

    if use_entity and use_temporal:
        start_year, end_year = time_window
        cache_key = (test_case.entity, start_year, end_year)
        cached = session.combined_index_cache.get(cache_key)
        if cached is not None:
            return ResolvedIndex(cached[0], cached[1], cache_hit=True, build_seconds=0.0)

        where = {"$and": [
            {"entity": test_case.entity},
            {"date_start": {"$lte": end_year}},
            {"date_end": {"$gte": start_year}},
        ]}
        build_start = time.perf_counter()
        filtered_index, filtered_doc_ids = session.build_fn(session.collection, where=where)
        build_seconds = time.perf_counter() - build_start
        if filtered_index is None:
            print(
                f"Warning: combined entity+temporal filter matched no docs for entity={test_case.entity!r}, "
                f"window=({start_year}, {end_year}), falling back to unfiltered index"
            )
            filtered_index, filtered_doc_ids = session.index, session.doc_ids

        session.combined_index_cache[cache_key] = (filtered_index, filtered_doc_ids)
        return ResolvedIndex(filtered_index, filtered_doc_ids, cache_hit=False, build_seconds=build_seconds)

    if use_entity:
        cached = session.entity_index_cache.get(test_case.entity)
        if cached is not None:
            return ResolvedIndex(cached[0], cached[1], cache_hit=True, build_seconds=0.0)

        build_start = time.perf_counter()
        filtered_index, filtered_doc_ids = session.build_fn(session.collection, where={"entity": test_case.entity})
        build_seconds = time.perf_counter() - build_start
        if filtered_index is None:
            print(f"Warning: entity filter matched no docs for entity={test_case.entity!r}, falling back to unfiltered index")
            filtered_index, filtered_doc_ids = session.index, session.doc_ids

        session.entity_index_cache[test_case.entity] = (filtered_index, filtered_doc_ids)
        return ResolvedIndex(filtered_index, filtered_doc_ids, cache_hit=False, build_seconds=build_seconds)

    start_year, end_year = time_window
    cache_key = (start_year, end_year)
    cached = session.temporal_index_cache.get(cache_key)
    if cached is not None:
        return ResolvedIndex(cached[0], cached[1], cache_hit=True, build_seconds=0.0)

    build_start = time.perf_counter()
    filtered_index, filtered_doc_ids = session.build_fn(session.collection, where=_temporal_where(start_year, end_year))
    build_seconds = time.perf_counter() - build_start
    if filtered_index is None:
        print(f"Warning: temporal filter matched no docs for window=({start_year}, {end_year}), falling back to unfiltered index")
        filtered_index, filtered_doc_ids = session.index, session.doc_ids

    session.temporal_index_cache[cache_key] = (filtered_index, filtered_doc_ids)
    return ResolvedIndex(filtered_index, filtered_doc_ids, cache_hit=False, build_seconds=build_seconds)


def run_experiment_pipeline(query_set: list[EvalQuery], session: RetrievalSession, metadata: ExperimentRunMetadata = None) -> ExperimentRunMetadata:
    if metadata is None:
        print("metadata missing, using undefined metadata record")
        metadata = METADATA_RECORD["undefined"]

    question_forms = load_question_forms()

    output_rows = []
    for test_case in query_set:

        resolved = _resolve_index(session, test_case, metadata.entity_filter, metadata.temporal_mode)
        index, doc_ids = resolved.index, resolved.doc_ids

        mutate_start_time = time.perf_counter()
        rendered = render_question(_form_row(test_case), metadata.question_form, question_forms)
        mutated_question_text = mutate_question(
            rendered.text,
            metadata.phrasing,
            test_case.time_start_raw,
            test_case.time_end_raw,
        )
        mutate_latency = time.perf_counter() - mutate_start_time

        query_start_time = time.perf_counter()
        k_results = session.query_fn(
            mutated_question_text,
            index,
            doc_ids,
            session.collection,
            session.embedding_model,
            k=metadata.k or EVAL_QUERY_K,
        )
        retrieval_latency = time.perf_counter() - query_start_time
        total_latency = resolved.build_seconds + mutate_latency + retrieval_latency

        reranked = metadata.temporal_mode == "rerank"
        if reranked:
            k_results = rerank_by_span(k_results, _query_year_window(test_case), _result_span)
            for new_rank, result in enumerate(k_results, start=1):
                result["rank"] = new_rank

        retrieved_ids = [result["id"] for result in k_results]
        predicted_answer = k_results[0]["text"] if k_results else ""
        test_case = compute_metrics(
            test_case,
            retrieved_ids,
            predicted_answer=predicted_answer,
            latency=retrieval_latency,
            index_resolve_latency=resolved.build_seconds,
            mutate_latency=mutate_latency,
            retrieval_latency=retrieval_latency,
            total_latency=total_latency,
            index_cache_hit=resolved.cache_hit,
        )
        eval_metrics(test_case, metadata.k or EVAL_QUERY_K)

        output_rows.append({
            "query_id": test_case.query_id,
            "test_query": mutated_question_text,
            "gold_doc_ids": test_case.gold_doc_ids,
            "hit_at_k": test_case.hit_at_k,
            "precision_at_k": test_case.precision_at_k,
            "recall_at_k": test_case.recall_at_k,
            "mrr": test_case.mrr,
            "answer_f1": test_case.answer_f1,
            "answer_exact_match": test_case.answer_exact_match,
            "chrf": test_case.chrf,
            "gold_answers": test_case.gold_answers,
            "predicted_answer": predicted_answer,
            "latency": test_case.latency,
            "index_resolve_latency": test_case.index_resolve_latency,
            "mutate_latency": test_case.mutate_latency,
            "retrieval_latency": test_case.retrieval_latency,
            "total_latency": test_case.total_latency,
            "index_cache_hit": test_case.index_cache_hit,
            "phrasing_applied": mutated_question_text != rendered.text,
            "question_form_applied": rendered.applied,
            "question_form_applicable": rendered.applicable,
            "relation_mode": test_case.relation_mode,
            "reranked": reranked,
            "k_results": k_results,
        })

    save_rows_to_disk(output_rows, metadata.get_file_path())

    metric_sums = {}
    metric_counts = {}
    for row in output_rows:
        for metric_name in (
            "hit_at_k", "precision_at_k", "recall_at_k", "mrr", "answer_f1", "answer_exact_match", "chrf", "latency",
            "index_resolve_latency", "mutate_latency", "retrieval_latency", "total_latency",
        ):
            value = row[metric_name]
            if value is not None:
                metric_sums[metric_name] = metric_sums.get(metric_name, 0) + value
                metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1

    for metric_name, total in metric_sums.items():
        metadata.metrics[metric_name] = total / metric_counts[metric_name]

    metadata.metrics["phrasing_coverage"] = (
        sum(1 for row in output_rows if row["phrasing_applied"]) / len(output_rows) if output_rows else 0.0
    )
    form_applicable = sum(1 for row in output_rows if row["question_form_applicable"])
    form_applied = sum(1 for row in output_rows if row["question_form_applied"])
    metadata.metrics["question_form_applicable"] = form_applicable / len(output_rows) if output_rows else 0.0
    metadata.metrics["question_form_coverage"] = form_applied / form_applicable if form_applicable else 0.0
    if metadata.question_form != "timeqa_source" and metadata.metrics["question_form_coverage"] < QUESTION_FORM_COVERAGE_MIN:
        reason = (
            f"question_form_coverage {metadata.metrics['question_form_coverage']:.3f} below "
            f"{QUESTION_FORM_COVERAGE_MIN} for question_form={metadata.question_form!r} - some rows "
            f"are unmutated duplicates of the control"
        )
        metadata.invalid_reason = f"{metadata.invalid_reason}; {reason}" if metadata.invalid_reason else reason
        print(f"Warning: run {metadata.file_stem!r} marked invalid - {reason}")

    if metadata.phrasing != "source" and metadata.metrics["phrasing_coverage"] < PHRASING_COVERAGE_MIN:
        metadata.invalid_reason = (
            f"phrasing_coverage {metadata.metrics['phrasing_coverage']:.3f} below {PHRASING_COVERAGE_MIN} "
            f"for phrasing={metadata.phrasing!r} - some rows are unmutated duplicates of the control"
        )
        print(f"Warning: run {metadata.file_stem!r} marked invalid - {metadata.invalid_reason}")

    cache_misses = [row for row in output_rows if row["index_cache_hit"] is False]
    metadata.metrics["index_build_count"] = len(cache_misses)
    metadata.metrics["index_build_total_seconds"] = sum(row["index_resolve_latency"] for row in cache_misses)
    metadata.metrics["index_build_mean_seconds"] = (
        metadata.metrics["index_build_total_seconds"] / len(cache_misses) if cache_misses else 0.0
    )
    metadata.metrics["index_cache_hit_rate"] = (
        sum(1 for row in output_rows if row["index_cache_hit"]) / len(output_rows) if output_rows else 1.0
    )

    return metadata


def run_experiment_suite(suite_name: str = "baseline", skip_pause: bool = False):
    if suite_name == "baseline":
        print("No experiment suite defined, running baseline experiment suite.")

    query_source = get_suite_query_source(suite_name)
    queries_path = resolve_queries_output_path(query_source)
    corpus_path = resolve_corpus_output_path(query_source)
    print(f"Suite {suite_name!r} query_source: {query_source!r} (queries file: {queries_path}, corpus file: {corpus_path})")

    if not queries_path.exists():
        raise FileNotFoundError(
            f"No built queries file for suite {suite_name!r} (query_source={query_source!r}) at "
            f"{queries_path} - build it first, e.g. via build_queries.main({query_source!r})"
        )
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"No built corpus file for suite {suite_name!r} (query_source={query_source!r}) at "
            f"{corpus_path} - build it first, e.g. via build_corpus.main({query_source!r})"
        )
    query_set = load_all_queries(queries_path)
    run_metadata_list = load_manifest_suite(suite_name)

    if any(metadata.phrasing == "language" for metadata in run_metadata_list):
        require_phrasing_cache()

    docs = load_data(corpus_path)

    encoded_cache: dict[str, EncodedCorpus] = {}

    def _encoded_for(encoding: str) -> EncodedCorpus:
        if encoding not in encoded_cache:
            encoded_cache[encoding] = apply_corpus_encoding(docs, encoding)
        return encoded_cache[encoding]

    warn_if_heavy(suite_name, run_metadata_list, query_set, docs, query_source, skip_pause=skip_pause,
                  encoded_docs_for=lambda encoding: _encoded_for(encoding).docs)

    session_cache: dict[tuple[str, str, str, str], RetrievalSession] = {}

    successful_run_metadata = []
    for metadata in run_metadata_list:
        session_key = (metadata.model_name, metadata.retriever, query_source, metadata.corpus_encoding)
        if session_key not in session_cache:
            encoded = _encoded_for(metadata.corpus_encoding)
            print(
                f"Corpus encoding {metadata.corpus_encoding!r}: {encoded.changed_count}/{encoded.dated_count} "
                f"dated paragraph(s) rewritten (coverage {encoded.coverage:.4f})"
            )
            session_cache[session_key] = build_session(
                metadata.model_name, metadata.retriever, encoded.docs, query_source,
                corpus_encoding=metadata.corpus_encoding,
            )

        modified_metadata = run_experiment_pipeline(query_set, session_cache[session_key], metadata)

        encoded = _encoded_for(metadata.corpus_encoding)
        modified_metadata.metrics["corpus_encoding_coverage"] = encoded.coverage
        if metadata.corpus_encoding != "source" and encoded.coverage < CORPUS_ENCODING_COVERAGE_MIN:
            reason = (
                f"corpus_encoding_coverage {encoded.coverage:.3f} below {CORPUS_ENCODING_COVERAGE_MIN} "
                f"for corpus_encoding={metadata.corpus_encoding!r} - most dated paragraphs were left "
                f"unchanged, so this arm is a near-duplicate of the source encoding"
            )
            modified_metadata.invalid_reason = (
                f"{modified_metadata.invalid_reason}; {reason}" if modified_metadata.invalid_reason else reason
            )
            print(f"Warning: run {metadata.file_stem!r} marked invalid - {reason}")

        successful_run_metadata.append(modified_metadata.get_printable())

    save_rows_to_disk(successful_run_metadata, METADATA_WRITE_PATH)

