from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment.visualisation.visualise import _get_title, _slugify_title

METADATA_FILENAME = "_metadata.jsonl"

TIMEQA_REFERENCE_ROWS = (
    {"system": "TimeQA FiD baseline (easy)", "exact_match": 60.5, "f1": 67.9},
    {"system": "TimeQA human performance (easy)", "exact_match": 89.0, "f1": 93.3},
)

MEASUREMENT_NOTE = (
    "predicted_answer here is the full top-1 retrieved passage text, not a generated short "
    "answer. Precision, and therefore F1 and exact match, are structurally suppressed "
    "relative to TimeQA's own answer-generation pipeline, which scores a short generated "
    "string against the gold answer. The gap between this project and the TimeQA reference "
    "rows is a measurement-methodology difference, not evidence of worse retrieval."
)


def resolve_metadata_path(target: Path) -> Path:
    metadata_path = target / METADATA_FILENAME if target.is_dir() else target
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No {METADATA_FILENAME} found at {metadata_path}. Pass a run directory or the "
            f"path to its {METADATA_FILENAME}."
        )
    return metadata_path


def load_metadata_rows(metadata_path: Path) -> list[dict]:
    rows = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{metadata_path} exists but contains no rows.")
    return rows


def project_scores(row: dict) -> dict:
    metrics = row.get("metrics", {})
    exact_match = metrics.get("answer_exact_match")
    f1 = metrics.get("answer_f1")
    if exact_match is None or f1 is None:
        label = row.get("id") or row.get("file_stem") or "run"
        raise ValueError(
            f"run {label!r} has no answer_exact_match/answer_f1 in metrics - run the suite "
            f"with predicted_answer populated before comparing."
        )
    return {"system": row.get("id") or row.get("file_stem") or "this project",
            "exact_match": exact_match, "f1": f1}


def build_comparison_table(rows: list[dict]) -> str:
    lines = ["| System | EM | F1 |", "| --- | --- | --- |"]
    for row in rows:
        scores = project_scores(row)
        lines.append(f"| {scores['system']} | {scores['exact_match'] * 100:.1f} | {scores['f1'] * 100:.1f} |")
    for reference in TIMEQA_REFERENCE_ROWS:
        lines.append(f"| {reference['system']} | {reference['exact_match']:.1f} | {reference['f1']:.1f} |")
    return "\n".join(lines)


def build_markdown(rows: list[dict], chart_title: str) -> str:
    return "\n".join([
        f"# {chart_title} - EM / F1 vs TimeQA (easy split)",
        "",
        build_comparison_table(rows),
        "",
        MEASUREMENT_NOTE,
        "",
    ])


def write_comparison(metadata_path: Path, run_dir: Path | None = None) -> Path:
    rows = load_metadata_rows(metadata_path)
    run_dir = run_dir or metadata_path.parent
    chart_title = _get_title(rows[0].get("id") or rows[0].get("file_stem"))
    slug = _slugify_title(chart_title)
    out_path = run_dir / f"{slug}_timeqa_comparison.md"
    out_path.write_text(build_markdown(rows, chart_title), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare this project's answer EM/F1 against TimeQA's published easy-split numbers (offline)."
    )
    parser.add_argument("target", type=Path, help=f"a run directory or a {METADATA_FILENAME} path")
    args = parser.parse_args()

    metadata_path = resolve_metadata_path(args.target)
    out_path = write_comparison(metadata_path)
    print(out_path.read_text(encoding="utf-8"))
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
