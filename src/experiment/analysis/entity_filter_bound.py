import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median

from experiment.util.consts import CORPUS_OUTPUT_PATH, QUERIES_OUTPUT_PATH, RESULTS_OUTPUT_DIR

DEFAULT_K_VALUES = (1, 3, 5, 10, 20, 30)


def load_article_sizes(corpus_path: Path = CORPUS_OUTPUT_PATH) -> Counter:
    sizes = Counter()
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sizes[row["entity"]] += 1
    return sizes


def load_scored_queries(queries_path: Path = QUERIES_OUTPUT_PATH) -> list[dict]:
    queries = []
    with queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["gold_doc_ids"]:
                queries.append(row)
    return queries


def structural_hit_probability(article_size: int, gold_count: int, k: int) -> float:
    if gold_count <= 0 or article_size <= k:
        return 1.0
    non_gold = max(article_size - gold_count, 0)
    if k > non_gold:
        return 1.0
    return 1.0 - (math.comb(non_gold, k) / math.comb(article_size, k))


def compute_summary(article_sizes: Counter) -> dict:
    sizes = list(article_sizes.values())
    return {
        "distinct_entities": len(article_sizes),
        "total_paragraphs": sum(sizes),
        "median_paragraphs_per_article": median(sizes),
    }


def compute_k_curve(article_sizes: Counter, queries: list[dict], k_values=DEFAULT_K_VALUES) -> list[dict]:
    n = len(queries)
    rows = []
    for k in k_values:
        le_k = 0
        length_sum = 0
        hit_floor_sum = 0.0
        for q in queries:
            size = article_sizes.get(q["entity"], 0)
            if size <= k:
                le_k += 1
            length_sum += min(size, k)
            hit_floor_sum += structural_hit_probability(size, len(q["gold_doc_ids"]), k)
        rows.append({
            "k": k,
            "pct_queries_article_le_k": 100 * le_k / n,
            "mean_filtered_length": length_sum / n,
            "structural_hit_floor": hit_floor_sum / n,
        })
    return rows


def print_report(summary: dict, k_curve: list[dict]) -> None:
    print("Entity filter upper-bound statistics")
    print(f"  Distinct entities:           {summary['distinct_entities']}")
    print(f"  Total paragraphs:            {summary['total_paragraphs']}")
    print(f"  Median paragraphs/article:   {summary['median_paragraphs_per_article']}")
    print()
    print(f"{'K':>4}  {'% article<=K':>13}  {'mean list len':>13}  {'hit floor':>9}")
    for row in k_curve:
        print(f"{row['k']:>4}  {row['pct_queries_article_le_k']:>12.1f}%  {row['mean_filtered_length']:>13.1f}  {row['structural_hit_floor']:>9.1%}")


def write_outputs(summary: dict, k_curve: list[dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "entity_filter_bound.json"
    csv_path = output_dir / "entity_filter_bound.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "k_curve": k_curve}, f, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "pct_queries_article_le_k", "mean_filtered_length", "structural_hit_floor"])
        writer.writeheader()
        writer.writerows(k_curve)

    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Entity-filter oracle upper-bound statistics (no retrieval required).")
    parser.add_argument("-k", "--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES), help=f"K values to report (default: {DEFAULT_K_VALUES}).")
    args = parser.parse_args()

    article_sizes = load_article_sizes()
    queries = load_scored_queries()
    summary = compute_summary(article_sizes)
    k_curve = compute_k_curve(article_sizes, queries, args.k_values)
    print_report(summary, k_curve)

    now = datetime.now()
    output_dir = RESULTS_OUTPUT_DIR / now.strftime("%Y_%m_%d") / now.strftime("%H-%M-%S")
    json_path, csv_path = write_outputs(summary, k_curve, output_dir)
    print(f"\nWritten: {json_path}")
    print(f"Written: {csv_path}")


if __name__ == "__main__":
    main()
