import json

import pytest

from experiment.classes.class_query import EvalQuery
from experiment.eval.eval_harness import load_all_queries
from experiment.preprocessing.build_queries import validate_relations


def _row(**overrides):
    row = {
        "query_id": "/wiki/Test#P39#0",
        "relation_code": "P39",
        "relation_meaning": "position held",
        "relation_mode": "accumulate",
        "question_text": "What was the position of Test from 1997 to 2001?",
        "start_iso_value": "1997",
        "end_iso_value": "2001",
        "time_start_raw": "1997",
        "time_end_raw": "2001",
        "entity": "Test",
        "gold_doc_ids": ["corpus.jsonl_Test__para1"],
    }
    row.update(overrides)
    return row


def _write_rows(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_loaded_relation_mode_is_a_string_not_a_dict(tmp_path):
    path = tmp_path / "queries.jsonl"
    _write_rows(path, [_row()])

    queries = load_all_queries(path)

    assert len(queries) == 1
    assert isinstance(queries[0].relation_mode, str)
    assert queries[0].relation_mode == "accumulate"


def test_pointwise_mode_survives_the_load(tmp_path):
    path = tmp_path / "queries.jsonl"
    _write_rows(path, [_row(relation_code="P166", relation_mode="pointwise")])

    assert load_all_queries(path)[0].relation_mode == "pointwise"


def test_constructing_with_the_whole_relations_entry_raises():
    with pytest.raises(TypeError, match="must be a str"):
        EvalQuery.row_to_structured_query(
            _row(relation_mode={"meaning": "position held", "mode": "accumulate"})
        )


def test_row_without_relation_mode_is_skipped(tmp_path):
    path = tmp_path / "queries.jsonl"
    _write_rows(path, [_row(), _row(query_id="/wiki/Other#P999#0", relation_mode=None)])

    queries = load_all_queries(path)

    assert [q.query_id for q in queries] == ["/wiki/Test#P39#0"]


def test_vendored_relations_satisfy_the_template_invariants():
    from pathlib import Path

    from experiment.preprocessing.build_queries import load_relations
    from experiment.util.consts import RELATIONS_PATH

    relations = load_relations(Path(RELATIONS_PATH))

    assert len(relations) == 102
    assert all("template" in entry for entry in relations.values())
    validate_relations(relations, Path(RELATIONS_PATH))


@pytest.mark.parametrize("mode,slot,other", [("accumulate", "$4", "$2"), ("pointwise", "$2", "$4")])
def test_validate_relations_rejects_the_wrong_time_slot(mode, slot, other):
    from pathlib import Path

    bad = {"P1": {"meaning": "x", "mode": mode, "template": [f"Who is $1 {other}?"]}}
    with pytest.raises(ValueError, match="must not use|needs exactly one"):
        validate_relations(bad, Path("test"))
