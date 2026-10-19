from __future__ import annotations

import json
import re
from pathlib import Path

from experiment.util.consts import LLM_MODEL_NAME, PHRASING_CACHE_DIR
from experiment.util.date_utils import YEAR_RE, normalize_date

"""
changes query question text before it goes to the retriever
"source" is the unchanged control
"fuzzy" buckets by decade
"language" swaps each date for words, read from a committed cache
"numeric" swaps each date for its iso form, a year only token becomes its edge month
(start "-01", end "-12") so it isn't left the same, see numeric_iso()
"delete" removes the time expression

"numeric", "fuzzy" and "language" swap the start and end tokens separately. "language" used to
need a literal "<start> to <end>" and left the question alone without it, so 10.3% of that arm
were copies of the control

all of them find dates with _resolve_span(), timeqa annotations are often more precise than
the question, so the raw token isn't always in the text
"""

__all__ = [
    "PHRASING_CONDITIONS", "RANGE_JOIN", "RANGE_LEAD", "TOKEN_LEAD", "fuzzy_phrase",
    "load_phrasing_cache", "mutate_question", "numeric_iso", "require_phrasing_cache",
    "resolve_phrasing_cache_path", "set_phrasing_cache",
]

PHRASING_CONDITIONS = ("source", "numeric", "fuzzy", "language", "delete")

RANGE_LEAD = r"(?:from|between)"
RANGE_JOIN = r"(?:to|and|until|through)"
TOKEN_LEAD = r"(?:from|in|on|at|during|since|by|until|to)"

_phrasing_cache: dict[str, str] | None = None


def resolve_phrasing_cache_path(model_name: str = LLM_MODEL_NAME) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name)
    return PHRASING_CACHE_DIR / f"phrasing_cache__{safe_name}.json"


def set_phrasing_cache(cache: dict[str, str] | None) -> None:
    global _phrasing_cache
    _phrasing_cache = cache


def load_phrasing_cache(path: Path | None = None) -> dict[str, str]:
    global _phrasing_cache
    if _phrasing_cache is not None and path is None:
        return _phrasing_cache

    cache_path = path or resolve_phrasing_cache_path()
    if not cache_path.exists():
        available = sorted(p.name for p in PHRASING_CACHE_DIR.glob("phrasing_cache__*.json")) if PHRASING_CACHE_DIR.exists() else []
        raise FileNotFoundError(
            f"No phrasing cache at {cache_path} - build it with "
            f"'python src/experiment/preprocessing/build_phrasing_cache.py'. Available caches: {available}"
        )

    with cache_path.open("r", encoding="utf-8") as f:
        cache = json.load(f)
    if not cache:
        raise ValueError(f"Phrasing cache {cache_path} is empty - rebuild before running the 'language' condition.")

    _phrasing_cache = cache
    return cache


def require_phrasing_cache() -> dict[str, str]:
    return load_phrasing_cache()


def mutate_question(question_text: str, phrasing: str, time_start_raw: str, time_end_raw: str) -> str:
    if phrasing == "source":
        return question_text
    if phrasing == "numeric":
        return _mutate_numeric(question_text, time_start_raw, time_end_raw)
    if phrasing == "fuzzy":
        return _mutate_fuzzy(question_text, time_start_raw, time_end_raw)
    if phrasing == "language":
        return _mutate_language(question_text, time_start_raw, time_end_raw)
    if phrasing == "delete":
        return _mutate_delete(question_text, time_start_raw, time_end_raw)

    raise ValueError(f"Unknown phrasing condition {phrasing!r}, expected one of {PHRASING_CONDITIONS}")


def _resolve_span(question_text: str, raw_token: str) -> str | None:
    if raw_token in question_text:
        return raw_token
    match = YEAR_RE.search(raw_token)
    if match and match.group(1) in question_text:
        return match.group(1)
    return None


def _substitute_spans(question_text: str, time_start_raw: str, start_text: str, time_end_raw: str, end_text: str) -> str:
    start_span = _resolve_span(question_text, time_start_raw)
    end_span = _resolve_span(question_text, time_end_raw)

    mutated = question_text
    search_from = 0
    if start_span is not None and end_span is not None:
        anchor = re.search(
            rf"\b{RANGE_LEAD}\s+{re.escape(start_span)}\s+{RANGE_JOIN}\s+{re.escape(end_span)}\b",
            question_text,
            re.IGNORECASE,
        )
        if anchor:
            search_from = anchor.start()

    if start_span is not None:
        index = mutated.find(start_span, search_from)
        mutated = mutated[:index] + start_text + mutated[index + len(start_span):]
        search_from = index + len(start_text)

    if end_span is not None:
        index = mutated.find(end_span, search_from)
        if index != -1:
            mutated = mutated[:index] + end_text + mutated[index + len(end_span):]

    return mutated


def _mutate_numeric(question_text: str, time_start_raw: str, time_end_raw: str) -> str:
    start_iso = numeric_iso(time_start_raw, "min_date")
    end_iso = numeric_iso(time_end_raw, "max_date")
    return _substitute_spans(question_text, time_start_raw, start_iso, time_end_raw, end_iso)


def numeric_iso(raw: str, bound_field: str) -> str:
    normalized = normalize_date(raw)
    if normalized.precision != "year":
        return normalized.iso_value
    return getattr(normalized, bound_field)[:7]


def _mutate_fuzzy(question_text: str, time_start_raw: str, time_end_raw: str) -> str:
    start_year, start_month = _year_month(time_start_raw)
    end_year, end_month = _year_month(time_end_raw)

    grouped = _group_ranges({
        "start_month": start_month,
        "start_year": start_year,
        "end_month": end_month,
        "end_year": end_year,
    })
    start_phrase = f"{grouped['fuzzy_year_start']} {grouped['start_decade']}s"
    end_phrase = f"{grouped['fuzzy_year_end']} {grouped['end_decade']}s"

    return _substitute_spans(question_text, time_start_raw, start_phrase, time_end_raw, end_phrase)


def _mutate_language(question_text: str, time_start_raw: str, time_end_raw: str) -> str:
    start_words = _get_natural_language_equivalent(time_start_raw)
    end_words = _get_natural_language_equivalent(time_end_raw)
    return _substitute_spans(question_text, time_start_raw, start_words, time_end_raw, end_words)


def _mutate_delete(question_text: str, time_start_raw: str, time_end_raw: str) -> str:
    start_span = _resolve_span(question_text, time_start_raw)
    end_span = _resolve_span(question_text, time_end_raw)

    mutated = question_text
    removed = 0
    if start_span is not None and end_span is not None:
        range_pattern = rf"\s*\b{RANGE_LEAD}\s+{re.escape(start_span)}\s+{RANGE_JOIN}\s+{re.escape(end_span)}\b"
        mutated, removed = re.subn(range_pattern, "", mutated, count=1, flags=re.IGNORECASE)

    if not removed:
        for span in (start_span, end_span):
            if span is None:
                continue
            token_pattern = rf"\s*\b{TOKEN_LEAD}\s+{re.escape(span)}\b"
            mutated, hit = re.subn(token_pattern, "", mutated, count=1, flags=re.IGNORECASE)
            if not hit:
                mutated = mutated.replace(span, "", 1)

    mutated = re.sub(r"\s{2,}", " ", mutated)
    return re.sub(r"\s+([?.,;:])", r"\1", mutated).strip()


def _year_month(raw: str) -> tuple[int, int | None]:
    normalized = normalize_date(raw)
    if normalized.precision == "year":
        return int(normalized.iso_value), None
    year_str, month_str = normalized.iso_value[:4], normalized.iso_value[5:7]
    return int(year_str), int(month_str)


"""
groups ranges into deterministic fuzzy categories based on start/end years, calculating distance & midpoint
"""
def _group_ranges(entry):
    start_month = entry["start_month"] if entry["start_month"] is not None else 1
    end_month = entry["end_month"] if entry["end_month"] is not None else 12
    start_year = int(entry["start_year"])
    end_year = int(entry["end_year"])

    start_decade = start_year // 10 * 10
    end_decade = end_year // 10 * 10
    fuzzy_year_start = "early" if start_year % 10 <= 3 else "mid" if start_year % 10 <= 6 else "late"
    fuzzy_year_end = "early" if end_year % 10 <= 3 else "mid" if end_year % 10 <= 6 else "late"

    range_fuzzy = f"{fuzzy_year_start} {start_decade}s to {fuzzy_year_end} {end_decade}s"

    return {
        "fuzzy_year_start": fuzzy_year_start,
        "fuzzy_year_end": fuzzy_year_end,
        "start_decade": start_decade,
        "end_decade": end_decade,
        "range_fuzzy": range_fuzzy
    }


def fuzzy_phrase(year: int) -> str:
    decade = year // 10 * 10
    band = "early" if year % 10 <= 3 else "mid" if year % 10 <= 6 else "late"
    return f"{band} {decade}s"


def _get_natural_language_equivalent(date_token: str) -> str:
    cache = load_phrasing_cache()
    if date_token not in cache:
        raise KeyError(
            f"Date token {date_token!r} missing from phrasing cache {resolve_phrasing_cache_path()} - "
            f"rebuild it with 'python src/experiment/preprocessing/build_phrasing_cache.py'."
        )
    return cache[date_token]
