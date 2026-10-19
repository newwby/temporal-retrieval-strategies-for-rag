import argparse
import csv
import json
from pathlib import Path

from experiment.analysis.encoding_grid_summary import build_table as build_grid_table
from experiment.util.consts import RESULTS_OUTPUT_DIR
from experiment.visualisation.metrics_at_k import curve, load_rows

HEADLINE_KS = (1, 3, 5)
GATE_K = 30
REPORT_KS = HEADLINE_KS + (GATE_K,)

CONTROL_HIT_AT_GATE_K = 0.7139
CONTROL_MRR_AT_GATE_K = 0.193
GATE_HIT_TOLERANCE_PP = 0.15
GATE_MRR_TOLERANCE = 0.002

REPORTABLE_SUITES = {
    "baseline": {"role": "headline-control", "control_token": "*single*", "oracle": False},
    "phrasing_test": {"role": "headline", "control_token": "phrasing-source", "oracle": False},
    "temporal_filtering_test": {"role": "headline", "control_token": "temporalmode-none", "oracle": False},
    "retriever_test": {"role": "secondary", "control_token": "faiss", "oracle": False},
    "entity_upper_bound": {"role": "secondary-oracle", "control_token": "entityfilter-off", "oracle": True},
    "question_form_test": {"role": "headline-support", "control_token": "qform-timeqa_source", "oracle": False},
    "multi_gold_train_test": {"role": "secondary-pr", "control_token": "", "oracle": False},
    "encoding_grid": {"role": "secondary-grid", "control_token": "encoding-source_phrasing-source", "oracle": False},
}

def _iter_run_dirs(root: Path):
    for metadata_path in sorted(root.glob("*/*/_metadata.jsonl")):
        yield metadata_path.parent


def _run_sort_key(run_dir: Path) -> tuple[str, str]:
    return (run_dir.parent.name, run_dir.name)


def discover_runs(root: Path) -> dict[str, Path]:
    latest: dict[str, Path] = {}
    for run_dir in _iter_run_dirs(root):
        stems = {json.loads(line)["id"] for line in (run_dir / "_metadata.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()}
        for suite in REPORTABLE_SUITES:
            if any(stem == f"{suite}_results" or stem.startswith(f"{suite}_k") for stem in stems):
                if suite not in latest or _run_sort_key(run_dir) > _run_sort_key(latest[suite]):
                    latest[suite] = run_dir
    return latest


def _arm_files(run_dir: Path, suite: str) -> list[Path]:
    files = [
        path for path in sorted(run_dir.glob("*.jsonl"))
        if path.name != "_metadata.jsonl"
        and (path.stem == f"{suite}_results" or path.stem.startswith(f"{suite}_k"))
    ]
    return files


def _arm_label(path: Path, suite: str) -> str:
    stem = path.stem
    if stem == f"{suite}_results":
        return suite
    rest = stem.split(f"{suite}_k", 1)[-1]
    parts = rest.split("_")
    return "_".join(parts[2:]) if len(parts) > 2 else rest


def score_suite(run_dir: Path, suite: str) -> list[dict]:
    meta = REPORTABLE_SUITES[suite]
    rows_out: list[dict] = []
    for path in _arm_files(run_dir, suite):
        rows = [row for row in load_rows(path) if row.get("gold_doc_ids") and row.get("k_results")]
        if not rows:
            print(f"  [skip] {path.name}: no scorable rows (missing gold_doc_ids / k_results)")
            continue
        label = _arm_label(path, suite)
        token = meta["control_token"]
        if token == "*single*":
            is_control = path.stem == f"{suite}_results"
        elif token:
            is_control = token in path.stem
        else:
            is_control = False
        data = curve(rows, REPORT_KS)
        for k in REPORT_KS:
            rows_out.append({
                "suite": suite,
                "role": meta["role"],
                "arm": label,
                "is_control": is_control,
                "is_oracle": meta["oracle"],
                "k": k,
                "n": data[k]["n"],
                "hit_at_k": round(data[k]["hit_at_k"], 6),
                "mrr": round(data[k]["mrr"], 6),
                "recall_at_k": round(data[k]["recall_at_k"], 6),
                "precision_at_k": round(data[k]["precision_at_k"], 6),
            })
    return rows_out


def check_gate(rows: list[dict]) -> list[str]:
    failures = []
    for row in rows:
        if not row["is_control"] or row["k"] != GATE_K:
            continue
        hit_pp = 100 * (row["hit_at_k"] - CONTROL_HIT_AT_GATE_K)
        mrr_delta = row["mrr"] - CONTROL_MRR_AT_GATE_K
        if abs(hit_pp) > GATE_HIT_TOLERANCE_PP or abs(mrr_delta) > GATE_MRR_TOLERANCE:
            failures.append(
                f"{row['suite']}/{row['arm']}: Hit@{GATE_K}={row['hit_at_k']:.4f} "
                f"(dpp {hit_pp:+.2f}), MRR@{GATE_K}={row['mrr']:.4f} (d {mrr_delta:+.4f}) "
                f"- expected {CONTROL_HIT_AT_GATE_K:.4f} / {CONTROL_MRR_AT_GATE_K:.3f}"
            )
    return failures


def print_tables(rows: list[dict]) -> None:
    by_suite: dict[str, list[dict]] = {}
    for row in rows:
        by_suite.setdefault(row["suite"], []).append(row)
    for suite, suite_rows in by_suite.items():
        oracle = suite_rows[0]["is_oracle"]
        tag = "  [ORACLE - not a comparable retrieval arm]" if oracle else ""
        print(f"\n{suite}  ({suite_rows[0]['role']}){tag}")
        arms = []
        for row in suite_rows:
            if row["arm"] not in arms:
                arms.append(row["arm"])
        header = "  " + "arm".ljust(36) + "".join(f"k={k}".ljust(22) for k in REPORT_KS)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for arm in arms:
            cells = ""
            for k in REPORT_KS:
                match = next(r for r in suite_rows if r["arm"] == arm and r["k"] == k)
                cells += f"{match['hit_at_k']:.1%} / {match['mrr']:.3f}".ljust(22)
            marker = " *" if next(r for r in suite_rows if r["arm"] == arm)["is_control"] else ""
            print("  " + (arm + marker).ljust(36) + cells)
    print("\n  (cells: Hit@k / MRR@k   * = control arm, must reproduce 71.39% / 0.193 at k=30)")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate the paper's final-output tables from saved runs "
                    "(headline suites scored at k=1,3,5 by truncation; k=30 kept as the regression gate).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-root", type=Path, default=RESULTS_OUTPUT_DIR,
                        help=f"tree of <date>/<time>/ run dirs to scan (default: {RESULTS_OUTPUT_DIR})")
    parser.add_argument("--run-dir", type=Path, action="append", default=[],
                        help="an explicit run dir to include (repeatable); skips discovery when given")
    parser.add_argument("--csv", type=Path, default=None, help="write the consolidated long-form csv here")
    parser.add_argument("--grid-csv", type=Path, default=None,
                        help="also write the encoding_grid long-form table here (via encoding_grid_summary)")
    args = parser.parse_args()

    if args.run_dir:
        run_dirs = list(args.run_dir)
        runs: dict[str, Path] = {}
        for run_dir in run_dirs:
            for suite in REPORTABLE_SUITES:
                if _arm_files(run_dir, suite):
                    runs[suite] = run_dir
    else:
        runs = discover_runs(args.results_root)

    if not runs:
        raise SystemExit(f"no reportable suite runs found under {args.results_root}")

    missing = [s for s in REPORTABLE_SUITES if s not in runs]
    if missing:
        print(f"[warn] no saved run found for: {', '.join(missing)} - re-run these in the suite pass")

    all_rows: list[dict] = []
    for suite in REPORTABLE_SUITES:
        if suite not in runs:
            continue
        print(f"\n=== {suite}  <-  {runs[suite]}")
        all_rows.extend(score_suite(runs[suite], suite))

    if not all_rows:
        raise SystemExit("no scorable arms found across discovered runs")

    print_tables(all_rows)

    failures = check_gate(all_rows)
    print("\n" + "=" * 72)
    if failures:
        print("REGRESSION GATE FAILED - control arm(s) do not reproduce 71.39% / 0.193 at k=30:")
        for line in failures:
            print(f"  ! {line}")
    else:
        print("regression gate OK - every control arm reproduces 71.39% / 0.193 at k=30")

    if args.csv:
        write_csv(all_rows, args.csv)

    if args.grid_csv and "encoding_grid" in runs:
        grid_rows = build_grid_table(runs["encoding_grid"])
        write_csv(grid_rows, args.grid_csv)


if __name__ == "__main__":
    main()
