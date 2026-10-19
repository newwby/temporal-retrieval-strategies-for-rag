import json
import random

from experiment.preprocessing.build_corpus import resolve_corpus_output_path
from experiment.preprocessing.build_manifest import DEFAULT_BUILD_CONFIG
from experiment.preprocessing.build_queries import resolve_queries_output_path
from experiment.preprocessing.temporal_coverage import (
    compute_gold_survival_stats, resolve_temporal_coverage_path, write_temporal_coverage_stats,
)
from experiment.util.consts import (
    EXPECTED_MULTI_EVIDENCE,
    EXPECTED_TOTAL_QUESTIONS,
    EXPECTED_WITHOUT_EVIDENCE,
    EXPECTED_WITH_EVIDENCE,
)

"""
sanity check for build_corpus.py / build_queries.py output, reads the files on disk so it
catches write bugs as well as logic bugs

takes a build config name, "dev" (default) asserts against EXPECTED_*, other configs run the
same checks but only print counts

run per config (e.g. main('train')), not part of the -b/-nv flow
"""

__all__ = ["main"]


def _load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(config_name: str = DEFAULT_BUILD_CONFIG) -> None:
    corpus_path = resolve_corpus_output_path(config_name)
    queries_path = resolve_queries_output_path(config_name)
    corpus_rows = _load_jsonl(corpus_path)
    query_rows = _load_jsonl(queries_path)
    corpus_ids = {row["id"] for row in corpus_rows}

    print(f"Build config: {config_name!r}")
    print(f"Loaded {len(corpus_rows)} corpus rows, {len(query_rows)} query rows")

    with_evidence = [r for r in query_rows if r["has_evidence"]]
    without_evidence = [r for r in query_rows if not r["has_evidence"]]
    multi_evidence = [r for r in query_rows if len(r["gold_doc_ids"]) > 1]

    if config_name == DEFAULT_BUILD_CONFIG:
        assert len(query_rows) == EXPECTED_TOTAL_QUESTIONS
        assert len(with_evidence) == EXPECTED_WITH_EVIDENCE
        assert len(without_evidence) == EXPECTED_WITHOUT_EVIDENCE
        assert len(multi_evidence) == EXPECTED_MULTI_EVIDENCE
        print("Count assertions on written files: OK")
    else:
        print(f"With evidence: {len(with_evidence)} / Without evidence: {len(without_evidence)}")
        print(f"Multi-evidence (>1 gold paragraph): {len(multi_evidence)}")

    prefix = f"{corpus_path.name}_"
    unresolved = []
    for row in with_evidence:
        for gold_id in row["gold_doc_ids"]:
            raw_id = gold_id[len(prefix):] if gold_id.startswith(prefix) else gold_id
            if raw_id not in corpus_ids:
                unresolved.append((row["query_id"], gold_id))
    if unresolved:
        print(f"UNRESOLVED gold_doc_ids: {len(unresolved)} (showing up to 5)")
        for query_id, gold_id in unresolved[:5]:
            print(f"  {query_id} -> {gold_id}")
    else:
        print("All gold_doc_ids resolve to a real corpus row: OK")

    corpus_by_id = {row["id"]: row for row in corpus_rows}
    print("\n--- spot checks (answerable) ---")
    sample = random.sample(with_evidence, min(5, len(with_evidence)))
    for row in sample:
        print(f"[{row['query_id']}] {row['question_text']}")
        for gold_id in row["gold_doc_ids"]:
            raw_id = gold_id[len(prefix):] if gold_id.startswith(prefix) else gold_id
            text = corpus_by_id.get(raw_id, {}).get("text", "<missing>")
            print(f"  -> {raw_id}: {text[:150]!r}")
        print()

    print("--- spot check (no evidence) ---")
    if without_evidence:
        empty_example = without_evidence[0]
        print(json.dumps(empty_example, indent=2, ensure_ascii=False))
    else:
        print("(no without-evidence rows in this config's queries)")

    print("\n--- temporal filter coverage (exclude mode) ---")
    survival_stats = compute_gold_survival_stats(corpus_rows, query_rows, corpus_path.name)
    print(
        f"Gold docs checked: {survival_stats['gold_docs_checked']} "
        f"(scored queries: {survival_stats['scored_queries']})"
    )
    print(f"  survive filter: {survival_stats['survive']} ({survival_stats['survive_pct']:.1%})")
    print(
        f"  no date metadata: {survival_stats['no_date_metadata']} "
        f"({survival_stats['no_date_metadata_pct']:.1%})"
    )
    print(
        f"  dated but non-overlapping: {survival_stats['non_overlapping']} "
        f"({survival_stats['non_overlapping_pct']:.1%})"
    )
    print(f"  missing from corpus: {survival_stats['missing_from_corpus']}")
    print(
        f"Queries with EVERY gold doc excluded: {survival_stats['fully_excluded_queries']} "
        f"({survival_stats['fully_excluded_queries_pct']:.1%})"
    )
    write_temporal_coverage_stats(config_name, "gold_survival", survival_stats)
    print(f"Temporal coverage stats written to: {resolve_temporal_coverage_path(config_name)}")


if __name__ == "__main__":
    main()
