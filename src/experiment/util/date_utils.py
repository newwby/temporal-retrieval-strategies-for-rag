from __future__ import annotations
from dataclasses import dataclass
import calendar
import re
from experiment.util.consts import MONTHS

"""
normalises time range tokens ("May 1997", "2001") into one shape
ported from w4 strip_questions.py normalize_date(), without the regex step as
annotated_dev.json already gives clean start/end tokens

also the one home for the free text date regexes used by extract_dates.py (corpus
paragraphs) and metrics_at_k.py (offline rerank), each used to keep its own copy
and they drifted apart
"""

__all__ = [
    "NormalizedDate", "normalize_date", "MONTH_RE", "DATE_RE", "RANGE_RE",
    "SINGLE_DATE_RE", "YEAR_RE", "plausible_year",
]

MONTH_RE = (
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|"
    r"April|May|June|July|August|September|October|November|December)"
)
DATE_RE = rf"(?:(?:\d{{1,2}}\s+)?{MONTH_RE}\s+)?\b(1[0-9]{{3}}|20[0-2][0-9])\b"
RANGE_RE = re.compile(rf"(?P<start>{DATE_RE})\s*to\s*(?P<end>{DATE_RE})", re.IGNORECASE)
SINGLE_DATE_RE = re.compile(DATE_RE, re.IGNORECASE)
YEAR_RE = SINGLE_DATE_RE


def plausible_year(year: int) -> bool:
    return 1000 <= year <= 2030


@dataclass
class NormalizedDate:
    raw: str
    precision: str
    iso_value: str
    min_date: str
    max_date: str


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
