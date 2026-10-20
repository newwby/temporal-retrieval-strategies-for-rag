import argparse
import csv
import json
from pathlib import Path

from experiment.preprocessing.corpus_encoding import CORPUS_ENCODINGS
from experiment.preprocessing.phrasing import PHRASING_CONDITIONS
from experiment.visualisation.metrics_at_k import curve, load_rows

REPORT_KS = (1, 3, 5, 10, 30)


def load_metadata(run_dir: Path) -> dict[tuple[str, str], dict]:
    by_cell = {}
    with (run_dir / "_metadata.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            by_cell[(row.get("corpus_encoding", "source"), row["phrasing"])] = row
    return by_cell


def find_cell_file(run_dir: Path, encoding: str, phrasing: str) -> Path:
    matches = sorted(run_dir.glob(f"*_encoding-{encoding}_phrasing-{phrasing}.jsonl"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly 1 result file for cell (encoding={encoding!r}, phrasing={phrasing!r}) "
            f"in {run_dir}, found {len(matches)}: {[m.name for m in matches]}"
        )
    return matches[0]


def build_table(run_dir: Path, ks=REPORT_KS) -> list[dict]:
    metadata = load_metadata(run_dir)
    encodings = [e for e in CORPUS_ENCODINGS if any(key[0] == e for key in metadata)]
    phrasings = [p for p in PHRASING_CONDITIONS if any(key[1] == p for key in metadata)]

    curves: dict[tuple[str, str], dict] = {}
    for encoding in encodings:
        for phrasing in phrasings:
            rows = load_rows(find_cell_file(run_dir, encoding, phrasing))
            curves[(encoding, phrasing)] = curve([r for r in rows if r.get("gold_doc_ids")], ks)

    global_source = curves[("source", "source")]
    rows_out = []
    for encoding in encodings:
        row_source = curves[(encoding, "source")]
        for phrasing in phrasings:
            cell = curves[(encoding, phrasing)]
            cell_metadata = metadata[(encoding, phrasing)]
            for k in ks:
                rows_out.append({
                    "corpus_encoding": encoding,
                    "phrasing": phrasing,
                    "k": k,
                    "n": cell[k]["n"],
                    "hit_at_k": cell[k]["hit_at_k"],
                    "mrr": cell[k]["mrr"],
                    "corpus_encoding_coverage": cell_metadata["metrics"].get("corpus_encoding_coverage", ""),
                    "phrasing_coverage": cell_metadata["metrics"].get("phrasing_coverage", ""),
                    "hit_delta_vs_global_source_pp": 100 * (cell[k]["hit_at_k"] - global_source[k]["hit_at_k"]),
                    "mrr_delta_vs_global_source": cell[k]["mrr"] - global_source[k]["mrr"],
                    "hit_delta_vs_row_source_pp": 100 * (cell[k]["hit_at_k"] - row_source[k]["hit_at_k"]),
                    "mrr_delta_vs_row_source": cell[k]["mrr"] - row_source[k]["mrr"],
                    "invalid_reason": cell_metadata.get("invalid_reason") or "",
                })
    return rows_out


def _matrix(rows: list[dict], k: int, metric: str) -> tuple[list[str], list[str], dict]:
    encodings, phrasings = [], []
    values = {}
    for row in rows:
        if row["k"] != k:
            continue
        if row["corpus_encoding"] not in encodings:
            encodings.append(row["corpus_encoding"])
        if row["phrasing"] not in phrasings:
            phrasings.append(row["phrasing"])
        values[(row["corpus_encoding"], row["phrasing"])] = row[metric]
    return encodings, phrasings, values


def print_table(rows: list[dict], k: int = 30) -> None:
    for metric, label, fmt in (("hit_at_k", f"Hit@{k}", lambda v: f"{100 * v:6.2f}%"),
                               ("mrr", f"MRR@{k}", lambda v: f"{v:7.4f}")):
        encodings, phrasings, values = _matrix(rows, k, metric)
        print(f"\n{label}  (rows = corpus_encoding, cols = query phrasing)")
        print(f"{'':10s}" + "".join(f"{p:>10s}" for p in phrasings))
        for encoding in encodings:
            cells = "".join(f"{fmt(values[(encoding, p)]):>10s}" for p in phrasings)
            print(f"{encoding:10s}{cells}")

    print(f"\nCost of query-side phrasing WITHIN each corpus encoding "
          f"(Hit@{k} pp vs that row's own source phrasing)")
    encodings, phrasings, values = _matrix(rows, k, "hit_delta_vs_row_source_pp")
    print(f"{'':10s}" + "".join(f"{p:>10s}" for p in phrasings))
    for encoding in encodings:
        cells = "".join(f"{values[(encoding, p)]:>+9.2f} " for p in phrasings)
        print(f"{encoding:10s}{cells}")

    flagged = {(r["corpus_encoding"], r["phrasing"]): r["invalid_reason"] for r in rows if r["invalid_reason"]}
    if flagged:
        print("\nFlagged cells:")
        for cell, reason in flagged.items():
            print(f"  {cell}: {reason}")


def write_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} row(s) to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="a run directory containing _metadata.jsonl and the cell .jsonl files")
    parser.add_argument("--csv", type=Path, default=None, help="write the long-form table here")
    parser.add_argument("--k", type=int, default=30, help="k for the printed matrices (default 30)")
    args = parser.parse_args()

    rows = build_table(args.run_dir)
    print_table(rows, k=args.k)
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
