from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from consts import  RAW_DATA_DIRECTORY, INVALID_RAW_DATA_DIRECTORY, MONTHS
import argparse
import calendar
import csv
import json
import re

"""
extracts explicit date ranges from timeqa question text
use -h for help

input: jsonl with at least {"idx": ..., "question": ...}
output (in raw_data):
    <input stem>.jsonl - one row per range found
    <input stem>.csv - optional, see OUTPUT_CSV
    no_explicit_range/<input stem>_no_explicit_range.jsonl - questions without ranges, for review

usage (from repo root):
    python3 .archived/w4_fuzzydistance/strip_questions.py data/timeqa/dataset/human_test_easy.json
    (same for human_test_hard, test_easy, test_hard)

keeps raw date precision (year, month or day) plus min/max calendar bounds
e.g. 1935 -> 1935-01-01 to 1935-12-31, nov 1935 -> 1935-11-01 to 1935-11-30
"""

#TODO remove broad bounds, maybe merge with strip_range_data.py to avoid duplicate work

# entry point when called from main.py
__all__ = ['strip_questions']

# set True for csv output too (off by default, jsonl suits processing)
OUTPUT_CSV = False

MONTH_RE = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)"
DATE_RE = rf"(?:(?:\d{{1,2}}\s+)?{MONTH_RE}\s+)?\d{{3,4}}"
RANGE_RE = re.compile(
    rf"(?P<text>(?P<start>{DATE_RE})\s*to\s*(?P<end>{DATE_RE}))",
    re.IGNORECASE,
)

@dataclass
class NormalizedDate:
    raw: str
    precision: str
    iso_value: str
    min_date: str
    max_date: str

@dataclass
class ExtractedRange:
    idx: str
    entity_slug: str
    relation_code: str
    question: str
    match_index: int
    range_numeric: str
    start_text: str
    end_text: str
    start_precision: str
    end_precision: str
    start_iso_value: str
    end_iso_value: str
    interval_start_min: str
    interval_start_max: str
    interval_end_min: str
    interval_end_max: str
    interval_broad_min: str
    interval_broad_max: str


def split_idx(idx: str) -> tuple[str, str]:
    # e.g. /wiki/Sabine_Hossenfelder#P937#0
    parts = idx.split("#")
    entity_slug = parts[0].replace("/wiki/", "") if parts else ""
    relation_code = parts[1] if len(parts) > 1 else ""
    return entity_slug, relation_code


def normalize_date(raw: str) -> NormalizedDate:
    raw = " ".join(raw.strip().split())
    parts = raw.split()

    if len(parts) == 1 and parts[0].isdigit():
        year = int(parts[0])
        return NormalizedDate(
            raw=raw,
            precision="year",
            iso_value=f"{year:04d}",
            min_date=f"{year:04d}-01-01",
            max_date=f"{year:04d}-12-31",
        )

    if len(parts) == 2 and parts[1].isdigit():
        month = MONTHS[parts[0].lower()]
        year = int(parts[1])
        last_day = calendar.monthrange(year, month)[1]
        return NormalizedDate(
            raw=raw,
            precision="month",
            iso_value=f"{year:04d}-{month:02d}",
            min_date=f"{year:04d}-{month:02d}-01",
            max_date=f"{year:04d}-{month:02d}-{last_day:02d}",
        )

    if len(parts) == 3 and parts[0].isdigit() and parts[2].isdigit():
        day = int(parts[0])
        month = MONTHS[parts[1].lower()]
        year = int(parts[2])
        return NormalizedDate(
            raw=raw,
            precision="day",
            iso_value=f"{year:04d}-{month:02d}-{day:02d}",
            min_date=f"{year:04d}-{month:02d}-{day:02d}",
            max_date=f"{year:04d}-{month:02d}-{day:02d}",
        )

    raise ValueError(f"Unsupported date token: {raw!r}")


def extract_ranges(input_path: Path) -> tuple[list[ExtractedRange], list[dict]]:
    rows: list[ExtractedRange] = []
    no_range: list[dict] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            obj = json.loads(line)
            idx = obj.get("idx", "")
            question = obj.get("question", "")
            entity_slug, relation_code = split_idx(idx)
            matches = list(RANGE_RE.finditer(question))

            if not matches:
                no_range.append({
                    "line_number": line_number,
                    "idx": idx,
                    "entity_slug": entity_slug,
                    "relation_code": relation_code,
                    "question": question,
                })
                continue

            for match_index, match in enumerate(matches, start=1):
                start = normalize_date(match.group("start"))
                end = normalize_date(match.group("end"))
                rows.append(ExtractedRange(
                    idx=idx,
                    entity_slug=entity_slug,
                    relation_code=relation_code,
                    question=question,
                    match_index=match_index,
                    range_numeric=match.group("text"),
                    start_text=start.raw,
                    end_text=end.raw,
                    start_precision=start.precision,
                    end_precision=end.precision,
                    start_iso_value=start.iso_value,
                    end_iso_value=end.iso_value,
                    interval_start_min=start.min_date,
                    interval_start_max=start.max_date,
                    interval_end_min=end.min_date,
                    interval_end_max=end.max_date,
                    interval_broad_min=start.min_date,
                    interval_broad_max=end.max_date,
                ))

    return rows, no_range


def write_outputs(rows: list[ExtractedRange], no_range: list[dict], output_file_path: Path) -> None:
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_file_path.with_suffix(".csv")
    output_jsonl_path = output_file_path.with_suffix(".jsonl")
    
    no_range_directory = output_file_path.parent / INVALID_RAW_DATA_DIRECTORY
    no_range_directory.mkdir(parents=True, exist_ok=True)
    
    no_range_path = no_range_directory / f"{output_file_path.stem}_no_explicit_range.jsonl"

    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    # no rows, exit early & flag (was a regex issue early on)
    if not fieldnames:
        print(f"No rows to write for output path: {output_file_path}")
        return

    if OUTPUT_CSV:
        with output_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

    with output_jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    if no_range:
        with no_range_path.open("w", encoding="utf-8") as f:
            for row in no_range:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


"""
strip questions and extract date ranges
takes a path instead of cli args so other scripts can call it
"""
def strip_questions(input_file: Path) -> None:
    script_base_directory = Path(__file__).resolve().parent
    output_file_path = script_base_directory / RAW_DATA_DIRECTORY / input_file.stem

    rows, no_range = extract_ranges(input_file)
    # csv only if OUTPUT_CSV
    write_outputs(rows, no_range, output_file_path)

    questions_with_ranges = len({row.idx for row in rows})
    print(f"Output path: {output_file_path}")
    print(f"Range occurrences: {len(rows)}")
    print(f"Questions with ranges: {questions_with_ranges}")
    print(f"Questions without explicit ranges: {len(no_range)}")

    if len(rows) == 0:
        print("Warning: No ranges were extracted. Please check the input file and regex patterns.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    strip_questions(args.input)


if __name__ == "__main__":
    main()

