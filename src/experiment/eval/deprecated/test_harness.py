from datetime import datetime
import json
from pathlib import Path

from experiment.util.consts import QUERIES_OUTPUT_PATH
from experiment.classes.class_query import EvalQuery, ExperimentRunMetadata
from experiment.eval.eval_harness import build_session, run_experiment_pipeline
from experiment.eval.metadata import METADATA_WRITE_PATH, load_manifest_suite
from experiment.util.storage import load_data, save_rows_to_disk

# test queries for the test harness
PILOT_TEST_QUERY_IDS = {
    # "Who owned Atari from Jul 1984 to 1996?" - a real range
    "accumulate": "/wiki/Atari#P127#1",
    # "Michael Batty received which award in 1982?" - a single point in time
    "pointwise": "/wiki/Michael_Batty#P166#2",
}


def _load_fixed_test_queries(path: Path = QUERIES_OUTPUT_PATH) -> list[EvalQuery]:
    # read then index by id, avoids a scan per query
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            # skip blank lines, e.g. a trailing one at eof
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows_by_query_id = {row["query_id"]: row for row in rows}
    
    test_queries = []
    for relation_mode, query_id in PILOT_TEST_QUERY_IDS.items():
        # fail if the data has changed
        if query_id not in rows_by_query_id:
            raise ValueError(
                f"Test query_id {query_id!r} (mode={relation_mode!r}) not found in {path} - "
                "has timeqa_queries.jsonl been regenerated with different contents?"
            )
        row = rows_by_query_id[query_id]
        test_queries.append(EvalQuery.row_to_structured_query(row, relation_mode))

    return test_queries


def run_pilot_test_pipeline() -> None:
    test_queries = _load_fixed_test_queries()
    metadata = load_manifest_suite("pilot_test")[0]
    docs = load_data()
    session = build_session(metadata.model_name, metadata.retriever, docs)
    modified_metadata = run_experiment_pipeline(test_queries, session, metadata)
    # save the pilot metadata row to a timestamped results file
    save_rows_to_disk([modified_metadata.get_printable()], METADATA_WRITE_PATH)

