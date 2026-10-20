import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

from experiment.eval.metrics import chrf_score
from experiment.util.consts import QUERIES_OUTPUT_PATH

CLASSES = ("numeric", "textual")


def load_gold_answers(queries_path: Path) -> dict[str, list[str]]:
    gold_by_id = {}
    with queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gold_by_id[row["query_id"]] = [answer for answer in row.get("gold_answers", []) if answer]
    return gold_by_id


def is_numeric_answer(gold_answers: list[str]) -> bool:
    if not gold_answers:
        return False
    return all(
        any(ch.isdigit() for ch in answer) and not any(ch.isalpha() for ch in answer)
        for answer in gold_answers
    )


def iter_result_files(run_dir: Path):
    for path in sorted(run_dir.glob("*.jsonl")):
        if path.name == "_metadata.jsonl":
            continue
        yield path


def top1_text(row: dict) -> str:
    k_results = row.get("k_results") or []
    if not k_results:
        return ""
    return k_results[0].get("text") or ""


def build_records(run_dir: Path, gold_by_id: dict[str, list[str]], beta: float) -> list[dict]:
    records = []
    for path in iter_result_files(run_dir):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                gold_answers = gold_by_id.get(row["query_id"], [])
                if not gold_answers:
                    continue
                records.append({
                    "file": path.name,
                    "query_id": row["query_id"],
                    "hit_at_k": bool(row.get("hit_at_k")),
                    "chrf": chrf_score(gold_answers, top1_text(row), beta=beta),
                    "answer_class": "numeric" if is_numeric_answer(gold_answers) else "textual",
                })
    return records


def pick_threshold(records: list[dict]) -> float:
    hit_scores = [r["chrf"] for r in records if r["hit_at_k"]]
    if hit_scores:
        return fmean(hit_scores)
    all_scores = [r["chrf"] for r in records]
    return fmean(all_scores) if all_scores else 0.0


def _class_summary(records: list[dict], threshold: float) -> dict:
    n = len(records)
    misses = [r for r in records if not r["hit_at_k"]]
    false_close = [r for r in misses if r["chrf"] >= threshold]
    hits = [r for r in records if r["hit_at_k"]]
    return {
        "n": n,
        "hit_rate": (len(hits) / n) if n else 0.0,
        "mean_chrf": fmean(r["chrf"] for r in records) if n else 0.0,
        "mean_chrf_hit": fmean(r["chrf"] for r in hits) if hits else 0.0,
        "mean_chrf_miss": fmean(r["chrf"] for r in misses) if misses else 0.0,
        "miss_count": len(misses),
        "false_close_count": len(false_close),
        "false_close_rate": (len(false_close) / len(misses)) if misses else 0.0,
    }


def summarise(records: list[dict], threshold: float) -> dict:
    summary = {"threshold": threshold, "overall": _class_summary(records, threshold)}
    for answer_class in CLASSES:
        subset = [r for r in records if r["answer_class"] == answer_class]
        summary[answer_class] = _class_summary(subset, threshold)
    summary["overstatement_pp"] = 100 * (
        summary["numeric"]["false_close_rate"] - summary["textual"]["false_close_rate"]
    )
    return summary


def print_report(summary: dict) -> None:
    print(f"chrF vs hit_at_k divergence (threshold = mean chrF of hit rows = {summary['threshold']:.4f})")
    print()
    header = f"{'class':<9}{'n':>7}{'hit rate':>10}{'mean chrF':>11}{'chrF|hit':>10}{'chrF|miss':>11}{'misses':>8}{'false-close':>13}"
    print(header)
    for key in ("overall", "numeric", "textual"):
        s = summary[key]
        print(
            f"{key:<9}{s['n']:>7}{s['hit_rate']:>10.2%}{s['mean_chrf']:>11.4f}"
            f"{s['mean_chrf_hit']:>10.4f}{s['mean_chrf_miss']:>11.4f}{s['miss_count']:>8}"
            f"{s['false_close_rate']:>12.2%}"
        )
    print()
    print(
        f"Numeric answers are called 'close despite a retrieval miss' "
        f"{summary['overstatement_pp']:+.1f} pp more often than textual answers."
    )


def write_csv(summary: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "class", "n", "hit_rate", "mean_chrf", "mean_chrf_hit", "mean_chrf_miss",
        "miss_count", "false_close_count", "false_close_rate",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in ("overall", "numeric", "textual"):
            writer.writerow({"class": key, **summary[key]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="a run directory holding *.jsonl result files")
    parser.add_argument("--queries", type=Path, default=QUERIES_OUTPUT_PATH, help="built queries file with gold_answers")
    parser.add_argument("--beta", type=float, default=1.0, help="chrF precision/recall weighting (1.0 = true F1)")
    parser.add_argument("--threshold", type=float, default=None, help="override the auto 'close enough' chrF threshold")
    parser.add_argument("--csv", type=Path, default=None, help="also write the per-class table as csv")
    parser.add_argument("--json", type=Path, default=None, help="also write the full summary as json")
    args = parser.parse_args()

    gold_by_id = load_gold_answers(args.queries)
    records = build_records(args.run_dir, gold_by_id, args.beta)
    if not records:
        raise SystemExit(f"No result rows with gold answers found under {args.run_dir}")

    threshold = args.threshold if args.threshold is not None else pick_threshold(records)
    summary = summarise(records, threshold)
    print_report(summary)

    if args.csv:
        write_csv(summary, args.csv)
        print(f"\nwrote {args.csv}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
