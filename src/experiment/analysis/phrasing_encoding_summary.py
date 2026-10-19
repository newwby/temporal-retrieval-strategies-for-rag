import argparse
import csv
import json
from pathlib import Path

from experiment.preprocessing.phrasing import PHRASING_CONDITIONS
from experiment.visualisation.metrics_at_k import curve, load_rows

REPORT_KS = (1, 3, 5, 10, 30)


def load_metadata(run_dir: Path) -> dict[str, dict]:
    by_condition = {}
    with (run_dir / "_metadata.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_condition[row["phrasing"]] = row
    return by_condition


def _require_single_encoding(run_dir: Path) -> None:
    encodings = set()
    with (run_dir / "_metadata.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                encodings.add(json.loads(line).get("corpus_encoding", "source"))
    if len(encodings) > 1:
        raise ValueError(
            f"{run_dir} is a corpus_encoding grid run ({len(encodings)} encodings: "
            f"{sorted(encodings)}). This tool keys results by phrasing alone and would read an "
            f"arbitrary cell per condition. Use analysis/encoding_grid_summary.py instead."
        )


def find_condition_file(run_dir: Path, condition: str) -> Path:
    matches = sorted(run_dir.glob(f"*_phrasing-{condition}.jsonl"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly 1 result file matching '*_phrasing-{condition}.jsonl' in {run_dir}, "
            f"found {len(matches)}: {[m.name for m in matches]}"
        )
    return matches[0]


def build_table(run_dir: Path, ks=REPORT_KS) -> list[dict]:
    _require_single_encoding(run_dir)
    metadata = load_metadata(run_dir)
    source_curve = None
    rows_out = []

    for condition in PHRASING_CONDITIONS:
        path = find_condition_file(run_dir, condition)
        rows = load_rows(path)
        scored = [r for r in rows if r.get("gold_doc_ids")]
        applied = [r for r in scored if r.get("phrasing_applied")]
        noop = len(scored) - len(applied)

        whole = curve(scored, ks)
        subset = curve(applied, ks) if applied else None
        if condition == "source":
            source_curve = whole

        coverage = metadata[condition]["metrics"]["phrasing_coverage"]
        for k in ks:
            hit_delta = whole[k]["hit_at_k"] - source_curve[k]["hit_at_k"]
            mrr_delta = whole[k]["mrr"] - source_curve[k]["mrr"]
            rows_out.append({
                "condition": condition,
                "k": k,
                "n": whole[k]["n"],
                "hit_at_k": whole[k]["hit_at_k"],
                "mrr": whole[k]["mrr"],
                "hit_at_k_applied_subset": subset[k]["hit_at_k"] if subset else "",
                "mrr_applied_subset": subset[k]["mrr"] if subset else "",
                "phrasing_coverage": coverage,
                "noop_count": noop,
                "hit_delta_vs_source_pp": 100 * hit_delta,
                "mrr_delta_vs_source": mrr_delta,
            })
    return rows_out


def write_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict]) -> None:
    k30 = [r for r in rows if r["k"] == 30]
    print(f"{'condition':<10}{'coverage':>10}{'noop':>7}{'Hit@30':>9}{'MRR':>8}{'dHit(pp)':>10}{'dMRR':>8}")
    for r in k30:
        print(f"{r['condition']:<10}{r['phrasing_coverage']:>10.4f}{r['noop_count']:>7d}"
              f"{r['hit_at_k']:>9.2%}{r['mrr']:>8.3f}{r['hit_delta_vs_source_pp']:>10.2f}{r['mrr_delta_vs_source']:>8.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="a phrasing_test run directory (contains _metadata.jsonl)")
    parser.add_argument("--csv", type=Path, default=None, help="also write a tidy csv")
    parser.add_argument("--k", default=",".join(str(k) for k in REPORT_KS), help="comma-separated k values")
    args = parser.parse_args()

    ks = tuple(int(part) for part in args.k.split(","))
    rows = build_table(args.run_dir, ks)
    print_table(rows)

    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nwrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
