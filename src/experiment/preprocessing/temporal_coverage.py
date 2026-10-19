import json
import statistics
from datetime import datetime
from pathlib import Path

from experiment.preprocessing.build_manifest import DEFAULT_BUILD_CONFIG
from experiment.util.consts import DATA_DIR

"""
temporal coverage stats for one build config, used by build_corpus.py (paragraph date span
coverage) and verify.py (gold docs kept by an exclude mode temporal filter, needs the built
queries too). both write to one json file per config so the numbers can be quoted without rerunning

added after regex bugs in extract_dates.py made both numbers look fine while losing recall
"""

__all__ = [
    "compute_corpus_date_stats", "compute_gold_survival_stats",
    "resolve_temporal_coverage_path", "write_temporal_coverage_stats",
]

WIDE_SPAN_THRESHOLD_YEARS = 100


def compute_corpus_date_stats(rows: list[dict]) -> dict:
    total = len(rows)
    dated = [row for row in rows if "date_start" in row]
    undated = total - len(dated)
    widths = [row["date_end"] - row["date_start"] + 1 for row in dated]
    wide = [w for w in widths if w > WIDE_SPAN_THRESHOLD_YEARS]

    return {
        "total_paragraphs": total,
        "dated": len(dated),
        "dated_pct": (len(dated) / total) if total else 0.0,
        "undated": undated,
        "undated_pct": (undated / total) if total else 0.0,
        "span_over_100y": len(wide),
        "span_over_100y_pct_of_dated": (len(wide) / len(dated)) if dated else 0.0,
        "span_over_100y_pct_of_all": (len(wide) / total) if total else 0.0,
        "span_width_min": min(widths) if widths else None,
        "span_width_max": max(widths) if widths else None,
        "span_width_median": statistics.median(widths) if widths else None,
    }


def compute_gold_survival_stats(corpus_rows: list[dict], query_rows: list[dict], corpus_filename: str) -> dict:
    corpus_by_id = {row["id"]: row for row in corpus_rows}
    prefix = f"{corpus_filename}_"

    survive = no_date_meta = non_overlap = missing = 0
    scored_queries = fully_excluded_queries = 0

    for query in query_rows:
        golds = query.get("gold_doc_ids") or []
        if not golds:
            continue
        try:
            query_start, query_end = int(query["start_iso_value"][:4]), int(query["end_iso_value"][:4])
        except (TypeError, ValueError, KeyError):
            continue

        scored_queries += 1
        any_survive = False
        for gold_id in golds:
            raw_id = gold_id[len(prefix):] if gold_id.startswith(prefix) else gold_id
            row = corpus_by_id.get(raw_id)
            if row is None:
                missing += 1
            elif "date_start" not in row:
                no_date_meta += 1
            elif row["date_start"] <= query_end and row["date_end"] >= query_start:
                survive += 1
                any_survive = True
            else:
                non_overlap += 1
        if not any_survive:
            fully_excluded_queries += 1

    gold_checked = survive + no_date_meta + non_overlap + missing
    return {
        "scored_queries": scored_queries,
        "gold_docs_checked": gold_checked,
        "survive": survive,
        "survive_pct": (survive / gold_checked) if gold_checked else 0.0,
        "no_date_metadata": no_date_meta,
        "no_date_metadata_pct": (no_date_meta / gold_checked) if gold_checked else 0.0,
        "non_overlapping": non_overlap,
        "non_overlapping_pct": (non_overlap / gold_checked) if gold_checked else 0.0,
        "missing_from_corpus": missing,
        "fully_excluded_queries": fully_excluded_queries,
        "fully_excluded_queries_pct": (fully_excluded_queries / scored_queries) if scored_queries else 0.0,
    }


def resolve_temporal_coverage_path(config_name: str) -> Path:
    meta_dir = DATA_DIR / "meta"
    if config_name == DEFAULT_BUILD_CONFIG:
        return meta_dir / "temporal_coverage.json"
    return meta_dir / f"temporal_coverage_{config_name}.json"


def write_temporal_coverage_stats(config_name: str, section: str, data: dict) -> None:
    path = resolve_temporal_coverage_path(config_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            existing = json.load(f)

    existing["config_name"] = config_name
    existing[f"{section}_generated_at"] = datetime.now().isoformat()
    existing[section] = data

    with path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
