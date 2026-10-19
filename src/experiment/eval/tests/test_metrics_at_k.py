import json

from experiment.visualisation.metrics_at_k import curve, main

FIXTURE_ROWS = [
    {"query_id": "q0", "phrasing_applied": True, "gold_doc_ids": ["docA"],
     "k_results": [{"id": "docA", "rank": 1}]},
    {"query_id": "q1", "phrasing_applied": True, "gold_doc_ids": ["docB"],
     "k_results": [{"id": "docB", "rank": 1}]},
    {"query_id": "q2", "phrasing_applied": False, "gold_doc_ids": ["docC"],
     "k_results": [{"id": "docX", "rank": 1}]},
    {"query_id": "q3", "phrasing_applied": False, "gold_doc_ids": ["docD"],
     "k_results": [{"id": "docX", "rank": 1}]},
]


def test_applied_subset_scores_differently_from_whole_arm():
    whole = curve(FIXTURE_ROWS, (1,))
    applied_only = [row for row in FIXTURE_ROWS if row["phrasing_applied"]]
    subset = curve(applied_only, (1,))

    assert whole[1]["n"] == 4
    assert whole[1]["hit_at_k"] == 0.5
    assert subset[1]["n"] == 2
    assert subset[1]["hit_at_k"] == 1.0


def test_cli_subset_applied_flag_prints_labelled_subset_table(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fixture_path = run_dir / "numeric_k1.jsonl"
    with fixture_path.open("w", encoding="utf-8") as f:
        for row in FIXTURE_ROWS:
            f.write(json.dumps(row) + "\n")

    monkeypatch.setattr("sys.argv", ["metrics_at_k.py", str(run_dir), "--k", "1", "--subset-applied"])
    main()
    out = capsys.readouterr().out

    assert "[applied-subset]" in out
    assert "(n=2)" in out
    assert "(n=4)" in out


def test_cli_without_flag_omits_subset_table(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fixture_path = run_dir / "numeric_k1.jsonl"
    with fixture_path.open("w", encoding="utf-8") as f:
        for row in FIXTURE_ROWS:
            f.write(json.dumps(row) + "\n")

    monkeypatch.setattr("sys.argv", ["metrics_at_k.py", str(run_dir), "--k", "1"])
    main()
    out = capsys.readouterr().out

    assert "[applied-subset]" not in out
