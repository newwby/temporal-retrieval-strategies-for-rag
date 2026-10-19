import time

from experiment.classes.class_query import ExperimentRunMetadata
from experiment.util.consts import (
    DEVICE, EMBED_DOCS_PER_SECOND, HEAVY_RUN_PAUSE_SECONDS, HEAVY_RUN_QUERY_THRESHOLD,
    HEAVY_RUN_REEMBED_THRESHOLD, SMALL_DEFAULT_MODEL,
)
from experiment.util.storage import find_changed_docs


def assess_suite(
    suite_name: str,
    run_metadata_list: list[ExperimentRunMetadata],
    query_set: list,
    docs: list[dict],
    query_source: str,
    encoded_docs_for=None,
) -> dict | None:
    encoded_docs_for = encoded_docs_for or (lambda encoding: docs)
    num_runs = len(run_metadata_list)
    num_queries = len(query_set)
    models = sorted({m.model_name for m in run_metadata_list})
    large_models = [m for m in models if m != SMALL_DEFAULT_MODEL]
    needs_llm = any(m.phrasing == "language" for m in run_metadata_list)
    encodings = sorted({m.corpus_encoding for m in run_metadata_list})

    docs_to_embed_by_pair = {
        (model, encoding): len(find_changed_docs(
            encoded_docs_for(encoding), model_name=model, source=query_source, encoding=encoding))
        for encoding in encodings for model in models
    }
    docs_to_embed = sum(docs_to_embed_by_pair.values())

    slow_combo = DEVICE == "cpu" and bool(large_models)

    heavy = (
        num_runs * num_queries > HEAVY_RUN_QUERY_THRESHOLD
        or large_models
        or docs_to_embed > HEAVY_RUN_REEMBED_THRESHOLD
        or len(encodings) > 1
        or needs_llm
        or slow_combo
    )
    if not heavy:
        return None

    return {
        "suite_name": suite_name,
        "num_runs": num_runs,
        "num_queries": num_queries,
        "models": models,
        "device": DEVICE,
        "corpus_size": len(docs),
        "docs_to_embed": docs_to_embed,
        "docs_to_embed_by_pair": docs_to_embed_by_pair,
        "encodings": encodings,
        "embed_seconds_estimate": docs_to_embed / EMBED_DOCS_PER_SECOND.get(DEVICE, 60),
        "needs_llm": needs_llm,
        "slow_combo": slow_combo,
    }


def print_banner(summary: dict) -> None:
    print("=" * 72)
    print(f"HEAVY RUN: suite {summary['suite_name']!r} - {summary['num_runs']} run(s), "
          f"{summary['num_queries']} queries each, corpus {summary['corpus_size']} docs")
    print(f"  models: {', '.join(summary['models'])} | device: {summary['device']}")
    encodings = summary.get("encodings") or ["source"]
    if len(encodings) > 1:
        print(f"  corpus encodings: {', '.join(encodings)} ({len(encodings)})")
    if summary["docs_to_embed"]:
        print(f"  {summary['docs_to_embed']} document embedding(s) across {len(encodings)} corpus encoding(s):")
        for (model, encoding), count in sorted(summary.get("docs_to_embed_by_pair", {}).items()):
            note = "  (cached)" if count == 0 else ""
            print(f"      {model} / {encoding:9s}: {count:6d} / {summary['corpus_size']}{note}")
        seconds = summary.get("embed_seconds_estimate", 0)
        print(f"    rough encode estimate: ~{int(seconds // 3600)}h {int(seconds % 3600 // 60)}m on {summary['device']}")
    if len(encodings) > 1:
        print("  each encoding is a separate Chroma collection - disk grows by roughly one index per encoding")
        print("  'language' CORPUS encoding uses the deterministic rule (util/number_words.py), not the LLM cache")
    if summary["needs_llm"]:
        print("  'language' phrasing reads the precomputed LLM-backend cache (see util/consts.py PHRASING_CACHE_DIR)")
    if summary["slow_combo"]:
        print("  non-default model on CPU - the slowest realistic combination")
    print("  Ctrl+C now if you weren't expecting this.")
    print("=" * 72)


def warn_if_heavy(
    suite_name: str,
    run_metadata_list: list[ExperimentRunMetadata],
    query_set: list,
    docs: list[dict],
    query_source: str,
    skip_pause: bool = False,
    encoded_docs_for=None,
) -> None:
    summary = assess_suite(suite_name, run_metadata_list, query_set, docs, query_source, encoded_docs_for)
    if summary is None:
        return
    print_banner(summary)
    if not skip_pause:
        time.sleep(HEAVY_RUN_PAUSE_SECONDS)
