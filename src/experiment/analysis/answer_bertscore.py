"""
bertscore (and em/f1) for a saved experiment run, after the fact

bertscore compares the top 1 passage (predicted_answer) to the gold answers with contextual
embeddings, so paraphrases get credit that token f1/em miss

run separately rather than in the harness so the model only loads when asked, and a whole run
is scored in one batch. no re-retrieval, predicted_answer and gold_answers are on each saved
row (older runs without gold_answers are joined from timeqa_queries.jsonl by query_id)

caveats:
  - predicted_answer is a whole passage not a span, so precision style scores are capped,
    bertscore is another view of the same retriever only setup, not better than answer_f1
  - em is almost always 0 with this predicted_answer, only reported for completeness

usage:
    python -m experiment.analysis.answer_bertscore <run_dir> [--model-type MODEL]
"""

import argparse
import json
from pathlib import Path
from statistics import mean

from experiment.eval.metrics import exact_match_score, f1_score
from experiment.util.consts import QUERIES_OUTPUT_PATH
from experiment.visualisation.csv_export import write_csv

METADATA_STEM = "_metadata"


def load_gold_answers_index(queries_path: Path = QUERIES_OUTPUT_PATH) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    with queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            index[row["query_id"]] = list(row.get("gold_answers", []))
    return index


def load_run_rows(run_dir: Path) -> dict[Path, list[dict]]:
    per_file: dict[Path, list[dict]] = {}
    for path in sorted(run_dir.glob("*.jsonl")):
        if path.stem == METADATA_STEM:
            continue
        with path.open("r", encoding="utf-8") as f:
            per_file[path] = [json.loads(line) for line in f if line.strip()]
    return per_file


def resolve_gold(row: dict, gold_index: dict[str, list[str]]) -> list[str]:
    if row.get("gold_answers"):
        return list(row["gold_answers"])
    return gold_index.get(row["query_id"], [])


def score_rows(rows: list[dict], gold_index: dict[str, list[str]], model_type: str | None) -> list[dict]:
    """scores every row in one batch, returns per query bertscore + em/f1"""
    from bert_score import score as bert_score

    preds: list[str] = []
    refs: list[list[str]] = []
    resolved_golds: list[list[str]] = []
    for row in rows:
        gold = resolve_gold(row, gold_index)
        resolved_golds.append(gold)
        preds.append(row.get("predicted_answer") or "")
        # bert_score takes a list of references per prediction and keeps the best, empty gold -> [""]
        refs.append(gold or [""])

    kwargs = {"lang": "en"} if model_type is None else {"model_type": model_type}
    p, r, f1 = bert_score(preds, refs, verbose=False, **kwargs)

    scored = []
    for row, gold, pi, ri, fi in zip(rows, resolved_golds, p.tolist(), r.tolist(), f1.tolist()):
        pred = row.get("predicted_answer") or ""
        scored.append({
            "query_id": row["query_id"],
            "answer_exact_match": exact_match_score(gold, pred),
            "answer_f1": f1_score(gold, pred),
            "answer_bertscore_p": pi,
            "answer_bertscore_r": ri,
            "answer_bertscore_f1": fi,
        })
    return scored


def run_means(scored: list[dict]) -> dict[str, float]:
    keys = ("answer_exact_match", "answer_f1", "answer_bertscore_p", "answer_bertscore_r", "answer_bertscore_f1")
    return {k: mean(s[k] for s in scored) for k in keys} if scored else {k: 0.0 for k in keys}


def write_outputs(run_file: Path, scored: list[dict], means: dict[str, float]) -> tuple[Path, Path]:
    csv_path = run_file.with_name(f"{run_file.stem}_bertscore.csv")
    md_path = run_file.with_name(f"{run_file.stem}_bertscore.md")
    write_csv(csv_path, scored)

    header = "| metric | mean |\n| --- | --- |\n"
    body = "".join(f"| {k} | {v:.4f} |\n" for k, v in means.items())
    md_path.write_text(f"# BERTScore - {run_file.stem}\n\n{header}{body}", encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc BERTScore/EM/F1 for a saved run directory.")
    parser.add_argument("run_dir", type=Path, help="Run directory containing per-query *.jsonl files.")
    parser.add_argument(
        "--model-type", default=None,
        help="bert_score model_type (default: bert_score's English default, roberta-large). "
             "Use a lighter model for iteration; reserve the default for reported numbers.",
    )
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        parser.error(f"not a directory: {args.run_dir}")

    gold_index = load_gold_answers_index()
    per_file = load_run_rows(args.run_dir)
    if not per_file:
        parser.error(f"no per-query *.jsonl files found in {args.run_dir}")

    for run_file, rows in per_file.items():
        scored = score_rows(rows, gold_index, args.model_type)
        means = run_means(scored)
        csv_path, md_path = write_outputs(run_file, scored, means)
        print(f"\n{run_file.name}  ({len(scored)} queries)")
        for k, v in means.items():
            print(f"  {k:22s} {v:.4f}")
        print(f"  written: {csv_path.name}, {md_path.name}")


if __name__ == "__main__":
    main()
