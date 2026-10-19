import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / "src" / "experiment" / "data" / "results"
TRACKED_RESULTS = REPO_ROOT / "results"

SUITES = [
    ("phrasing_test", "phrasing"),
    ("temporal_filtering_test", "temporal_filtering_k1-3-5"),
    ("temporal_filtering_k30", "temporal_filtering_k30"),
    ("model_sweep", "model_sweep"),
    ("bm25", "bm25"),
    ("entity_upper_bound", "entity_upper_bound"),
    ("multi_gold_train_test", "multi_gold_train"),
]

COPY_SUFFIXES = (".png", ".csv", ".md", ".json")


def existing_run_dirs() -> set[Path]:
    return {p for p in RESULTS_ROOT.glob("*/*") if p.is_dir()}


def newest_new_run_dir(before: set[Path]) -> Path | None:
    created = [p for p in existing_run_dirs() if p not in before]
    if not created:
        return None
    return max(created, key=lambda p: p.stat().st_mtime)


def run_suite(suite: str) -> tuple[bool, Path | None, float]:
    before = existing_run_dirs()
    start = time.perf_counter()
    print("\n" + "=" * 78, flush=True)
    print(f"=== SUITE {suite}", flush=True)
    print("=" * 78, flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "experiment.main", "-s", suite, "-y"],
        cwd=REPO_ROOT,
    )
    elapsed = time.perf_counter() - start
    run_dir = newest_new_run_dir(before)
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED rc={result.returncode}"
    print(f"--- {suite}: {status} in {elapsed / 60:.1f} min -> {run_dir}", flush=True)
    return ok, run_dir, elapsed


def collect(run_dir: Path, label: str) -> list[Path]:
    """copies a run's figures, tables and csvs into the tracked results dir"""
    target = TRACKED_RESULTS / label
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and (path.suffix in COPY_SUFFIXES or path.name == "_metadata.jsonl"):
            shutil.copy2(path, target / path.name)
            copied.append(target / path.name)
    return copied


def run_analysis(name: str, argv: list[str]) -> bool:
    print(f"\n--- analysis: {name}", flush=True)
    print("    " + " ".join(str(a) for a in argv), flush=True)
    result = subprocess.run([sys.executable, *argv], cwd=REPO_ROOT)
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED rc={result.returncode}"
    print(f"--- analysis {name}: {status}", flush=True)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the final paper suites, then the post-hoc analyses, collecting every "
                    "figure/CSV into the tracked results/ directory."
    )
    parser.add_argument("-s", "--suite", action="append", default=None,
                        help="run only this suite (repeatable); default is all of them.")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="run the suites only, skip the post-hoc analysis scripts.")
    parser.add_argument("--analysis-only", action="store_true",
                        help="skip the suites, run the post-hoc analyses against the recorded runs.")
    args = parser.parse_args()

    TRACKED_RESULTS.mkdir(parents=True, exist_ok=True)
    selected = [(s, label) for s, label in SUITES if not args.suite or s in args.suite]
    manifest_path = TRACKED_RESULTS / "run_dirs.json"

    summary = []
    run_dirs: dict[str, Path] = {}
    if manifest_path.exists():
        run_dirs = {k: Path(v) for k, v in json.loads(manifest_path.read_text()).items()}

    if not args.analysis_only:
        for suite, label in selected:
            ok, run_dir, elapsed = run_suite(suite)
            if ok and run_dir is not None:
                files = collect(run_dir, label)
                run_dirs[suite] = run_dir
                summary.append((suite, label, "OK", f"{elapsed / 60:.1f} min", len(files)))
            else:
                summary.append((suite, label, "FAILED", f"{elapsed / 60:.1f} min", 0))
            manifest_path.write_text(json.dumps({k: str(v) for k, v in run_dirs.items()}, indent=2))

    if not args.skip_analysis:
        # chrf divergence and the timeqa em/f1 table use the phrasing run, its source arm is the control
        phrasing_dir = run_dirs.get("phrasing_test")
        if phrasing_dir is not None and phrasing_dir.is_dir():
            run_analysis("chrf_divergence", [
                "src/experiment/analysis/chrf_divergence.py", str(phrasing_dir),
                "--csv", str(phrasing_dir / "chrf_divergence.csv"),
                "--json", str(phrasing_dir / "chrf_divergence.json"),
            ])
            run_analysis("timeqa_metric_comparison", [
                "src/experiment/analysis/timeqa_metric_comparison.py", str(phrasing_dir),
            ])
            collect(phrasing_dir, "phrasing")

        # bertscore loads a model per run file, so use the 3 arm k=30 temporal run not the 15 arm
        # phrasing run, em and f1 come from the same script
        bert_dir = run_dirs.get("temporal_filtering_k30")
        if bert_dir is not None and bert_dir.is_dir():
            if run_analysis("answer_bertscore", [
                "-m", "experiment.analysis.answer_bertscore", str(bert_dir),
            ]):
                collect(bert_dir, "temporal_filtering_k30")

    print("\n" + "=" * 78)
    print("=== SUMMARY")
    print("=" * 78)
    for suite, label, status, elapsed, n in summary:
        print(f"  {status:7s} {suite:26s} -> results/{label:26s} {elapsed:>10s}  {n} file(s)")
    print(f"\nTracked outputs under: {TRACKED_RESULTS}")


if __name__ == "__main__":
    main()
