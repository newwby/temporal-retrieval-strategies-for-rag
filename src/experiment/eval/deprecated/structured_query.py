import json
from pathlib import Path

from experiment.util.consts import QUERIES_OUTPUT_PATH
from experiment.classes.class_query import EvalQuery as StructuredQuery

"""
query loader with two hand picked gold annotated rows from timeqa_queries.jsonl,
one for each return shape:
    "accumulate" - date range
    "pointwise" - single year

rows are already normalised by build_queries.py, loading them avoids typos
"""

# exports for import *
__all__ = ["StructuredQuery", "SEED_QUERY_IDS", "load_seed_queries"]


# two real query_ids picked from the 3021 rows for:
#   - a short readable question_text
#   - one gold_doc_id (keeps the paragraph check clear)
#   - a relation mode that matches the shape it stands for
SEED_QUERY_IDS = {
    # "Who owned Atari from Jul 1984 to 1996?" - a real range
    "accumulate": "/wiki/Atari#P127#1",
    # "Michael Batty received which award in 1982?" - a single point in time
    "pointwise": "/wiki/Michael_Batty#P166#2",
}




"""
reads a jsonl file line by line into dicts
"""
def _load_query_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            # skip blank lines, json.loads("") would raise
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


"""
converts a timeqa_queries.jsonl row into a StructuredQuery
time_kind comes from relation_mode not the row's dates, pointwise is one year and accumulate is a range
(holds in the data too, 100% of pointwise rows have start == end, 97.9% of accumulate rows don't)
"""
def _row_to_structured_query(row: dict, relation_mode: str) -> StructuredQuery:
    time_kind = "year" if relation_mode == "pointwise" else "range"

    return StructuredQuery(
        query_id=row["query_id"],
        relation_code=row["relation_code"],
        relation_mode=relation_mode,
        question_text=row["question_text"],
        time_kind=time_kind,
        time_start=row["start_iso_value"],
        time_end=row["end_iso_value"],
        # entity left as None for now
        gold_doc_ids=list(row["gold_doc_ids"]),
    )


"""
loads the two SEED_QUERY_IDS rows as StructuredQuery objects, accumulate first
"""
def load_seed_queries(path: Path = QUERIES_OUTPUT_PATH) -> list[StructuredQuery]:
    # step 1: read all rows
    rows = _load_query_rows(path)

    # step 2: index by query_id
    rows_by_query_id = {row["query_id"]: row for row in rows}

    # step 3: pull the two seed rows in order
    structured_queries = []
    for relation_mode, query_id in SEED_QUERY_IDS.items():
        # fail if a seed id is missing rather than quietly testing fewer
        if query_id not in rows_by_query_id:
            raise ValueError(
                f"Seed query_id {query_id!r} (mode={relation_mode!r}) not found in {path} - "
                "has timeqa_queries.jsonl been regenerated with different contents?"
            )
        row = rows_by_query_id[query_id]
        structured_queries.append(_row_to_structured_query(row, relation_mode))

    return structured_queries
