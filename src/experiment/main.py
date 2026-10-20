
from pathlib import Path
import sys
import argparse

from experiment.preprocessing.build_corpus import main as build_corpus_main, resolve_corpus_output_path
from experiment.preprocessing.build_manifest import DEFAULT_BUILD_CONFIG
from experiment.preprocessing.build_queries import main as build_queries_main, resolve_queries_output_path
from experiment.preprocessing.clean import clean_built_data
from experiment.preprocessing.verify import main as verify_main

from experiment.eval.eval_harness import run_experiment_suite
from experiment.eval.deprecated.test_harness import run_pilot_test_pipeline
from experiment.visualisation.visualise import main as run_visualise
from experiment.eval.metadata import METADATA_WRITE_PATH, get_suite_query_source


def _require_data_exists(suite_name: str) -> None:
    query_source = get_suite_query_source(suite_name)
    queries_path = resolve_queries_output_path(query_source)
    corpus_path = resolve_corpus_output_path(query_source)
    missing = [p for p in (queries_path, corpus_path) if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        print(
            f"No built data found for suite {suite_name!r} (query_source={query_source!r}): "
            f"({missing_str}) - run 'python main.py -b {query_source}' first."
        )
        sys.exit(1)


def run_build_pipeline(run_verify: bool = False, config_name: str = DEFAULT_BUILD_CONFIG):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Step 1: building corpus (config={config_name!r})...")
    build_corpus_main(config_name)

    print(f"\nStep 2: building queries (config={config_name!r})...")
    build_queries_main(config_name)

    if run_verify:
        print("\nStep 3: verifying outputs...")
        verify_main(config_name)


def run_test_pipeline():
    run_pilot_test_pipeline()


def run_visualise_pipeline(input_path: str | None = None):
    try:
        if input_path:
            run_visualise(input_path=Path(input_path))
        else:
            run_visualise()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=r"Run build (-b/--build) at least once before running -t/--test.")
    parser.add_argument("-b", "--build", nargs="?", const=DEFAULT_BUILD_CONFIG, default=None, metavar="CONFIG",
                         help=f"build CONFIG from build_manifest.json (default: {DEFAULT_BUILD_CONFIG!r}) and verify data.")
    parser.add_argument("-nv", "--no-verify", action="store_true", help="execute the build pipeline without verification.")
    parser.add_argument("-c", "--clean", action="store_true", help="remove previously built data (corpus/queries/chroma_db/data-indexing cache) for a clean rebuild - never touches data/timeqa/ source datasets.")
    parser.add_argument("-s", "--suite", default="baseline", help="name of the suite to run from manifest.json (used with -e, default: baseline).")
    parser.add_argument("-t", "--test", action="store_true", help="execute the pilot experiment test.")
    parser.add_argument("-v", "--visualise", action="store_true", help="re-visualise an existing run - -t/-s already visualise their own output automatically, so -v is only needed standalone with input_path pointing at an old run's _metadata.jsonl (defaults to visualisation/_metadata.jsonl if omitted).")
    parser.add_argument("-y", "--yes", action="store_true", help="skip the heavy-run banner's pause (banner still prints) - for unattended runs.")

    parser.add_argument("input_path", type=str, nargs="?", default=None, help="Path to your input (used for -v visualisation)")

    args = parser.parse_args()

    suite_explicitly_passed = any(
        arg == "-s" or arg == "--suite" or arg.startswith("--suite=") for arg in sys.argv[1:]
    )

    if args.clean:
        clean_built_data()
    elif args.build is not None and not args.no_verify:
        run_build_pipeline(True, args.build)
    elif args.build is not None and args.no_verify:
        run_build_pipeline(False, args.build)
    elif args.test:
        _require_data_exists("pilot_test")
        run_test_pipeline()
        run_visualise_pipeline(str(METADATA_WRITE_PATH))
    elif args.visualise and not suite_explicitly_passed:
        input_path = args.input_path if args.input_path else None
        run_visualise_pipeline(input_path)
    else:
        _require_data_exists(args.suite)
        run_experiment_suite(args.suite, skip_pause=args.yes)
        run_visualise_pipeline(str(METADATA_WRITE_PATH))


if __name__ == "__main__":
    main()

