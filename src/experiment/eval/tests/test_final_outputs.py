import json

from experiment.analysis import final_outputs


def _write_run(run_dir, stems_to_rows):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "_metadata.jsonl").open("w", encoding="utf-8") as f:
        for stem in stems_to_rows:
            f.write(json.dumps({"id": stem, "metrics": {}}) + "\n")
    for stem, rows in stems_to_rows.items():
        with (run_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


def _hit_row(query_id, gold, ranked):
    return {"query_id": query_id, "gold_doc_ids": [gold],
            "k_results": [{"id": doc_id, "rank": i + 1} for i, doc_id in enumerate(ranked)]}


def test_score_suite_flags_the_control_arm_by_token(tmp_path):
    run_dir = tmp_path / "2026_09_03" / "10-00-00"
    _write_run(run_dir, {
        "phrasing_test_k30_minilm_phrasing-source": [_hit_row("q0", "d0", ["d0"])],
        "phrasing_test_k30_minilm_phrasing-delete": [_hit_row("q0", "d0", ["x", "d0"])],
    })
    rows = final_outputs.score_suite(run_dir, "phrasing_test")
    control = {r["arm"] for r in rows if r["is_control"]}
    assert control == {"phrasing-source"}
    assert {r["arm"] for r in rows} == {"phrasing-source", "phrasing-delete"}


def test_multi_gold_subset_is_never_treated_as_a_control(tmp_path):
    run_dir = tmp_path / "2026_09_03" / "10-00-00"
    _write_run(run_dir, {"multi_gold_train_test_results": [_hit_row("q0", "d0", ["x", "x", "d0"])]})
    rows = final_outputs.score_suite(run_dir, "multi_gold_train_test")
    assert rows and not any(r["is_control"] for r in rows)
    assert not final_outputs.check_gate(rows)


def test_check_gate_fails_when_control_moves_past_tolerance(tmp_path):
    run_dir = tmp_path / "2026_09_03" / "10-00-00"
    hits = [_hit_row(f"q{i}", "d0", ["d0"]) for i in range(10)]
    misses = [_hit_row(f"q{i}", "d0", ["x"]) for i in range(10, 20)]
    _write_run(run_dir, {"baseline_results": hits + misses})
    rows = final_outputs.score_suite(run_dir, "baseline")
    assert any(r["is_control"] for r in rows)
    failures = final_outputs.check_gate(rows)
    assert failures and "baseline" in failures[0]


def test_discover_runs_picks_the_latest_run_per_suite(tmp_path):
    old = tmp_path / "2026_08_20" / "13-00-00"
    new = tmp_path / "2026_08_25" / "09-00-00"
    _write_run(old, {"phrasing_test_k30_minilm_phrasing-source": [_hit_row("q0", "d0", ["d0"])]})
    _write_run(new, {"phrasing_test_k30_minilm_phrasing-source": [_hit_row("q0", "d0", ["d0"])]})
    found = final_outputs.discover_runs(tmp_path)
    assert found["phrasing_test"] == new


def test_arm_label_strips_suite_prefix_and_run_metadata(tmp_path):
    phrasing = tmp_path / "phrasing_test_k30_all-minilm-l6-v2_phrasing-source.jsonl"
    grid = tmp_path / "encoding_grid_k30_all-minilm-l6-v2_encoding-fuzzy_phrasing-delete.jsonl"
    single = tmp_path / "baseline_results.jsonl"
    assert final_outputs._arm_label(phrasing, "phrasing_test") == "phrasing-source"
    assert final_outputs._arm_label(grid, "encoding_grid") == "encoding-fuzzy_phrasing-delete"
    assert final_outputs._arm_label(single, "baseline") == "baseline"
