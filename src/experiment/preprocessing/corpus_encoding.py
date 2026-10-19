from __future__ import annotations

from dataclasses import dataclass
import re

from experiment.preprocessing.extract_dates import DateMention, find_date_mentions
from experiment.preprocessing.phrasing import RANGE_LEAD, TOKEN_LEAD, fuzzy_phrase, numeric_iso
from experiment.util.date_utils import normalize_date
from experiment.util.number_words import day_to_words, year_to_words

"""
rewrites how the corpus writes its dates, before paragraphs are embedded

phrasing.py rewrites the query and keeps the corpus fixed, which only shows how far a rewritten
query drifts from the corpus. changing both gives the encoding x phrasing grid, the diagonal shows
if matching the corpus date form to the query gets the score back

1. metadata is never re-extracted. date_start/date_end come from the original text in
   build_corpus.py and are what the temporal filter uses. re-extracting here would tie the temporal
   arm to the encoding arm, so this doesn't import extract_paragraph_date_span

2. rewriting is never applied twice. "numeric" turns 1997 into 1997-01, a second pass would give
   1997-01-01. apply_corpus_encoding() always takes freshly loaded docs and returns new dicts

spans come from the original string and are walked in order, so replaced text is never re-scanned
and the "1" -> "0001" -> "0000001" bug (see _substitute_spans in phrasing.py) can't happen
"""

__all__ = [
    "CORPUS_ENCODINGS", "EncodedCorpus", "apply_corpus_encoding", "mutate_paragraph",
]

CORPUS_ENCODINGS = ("source", "numeric", "fuzzy", "language", "delete")

_RANGE_PREP = re.compile(rf"\b{RANGE_LEAD}\s+$", re.IGNORECASE)
_TOKEN_PREP = re.compile(rf"\b{TOKEN_LEAD}\s+$", re.IGNORECASE)
_PREP_LOOKBEHIND = 12

_MONTH_NAME = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|May|"
    r"June|July|August|September|October|November|December)\b", re.IGNORECASE
)
_DAY = re.compile(r"^\s*(\d{1,2})\s+")
_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")


def _year_of(token: str) -> int | None:
    match = _YEAR.search(token)
    return int(match.group(1)) if match else None



def _numeric_token(token: str, bound_field: str) -> str:
    return numeric_iso(token, bound_field)


def _fuzzy_token(token: str) -> str:
    year = _year_of(token)
    if year is None:
        raise ValueError(f"no year in {token!r}")
    return fuzzy_phrase(year)


def _language_token(token: str) -> str:
    year = _year_of(token)
    if year is None:
        raise ValueError(f"no year in {token!r}")

    words = year_to_words(year)
    month_match = _MONTH_NAME.search(token)
    if month_match:
        words = f"{month_match.group(1)} {words}"
        day_match = _DAY.match(token)
        if day_match:
            words = f"{day_to_words(int(day_match.group(1)))} {words}"
    return words


def _rewrite_mention(mention: DateMention, encoding: str) -> str:
    if encoding == "numeric":
        if mention.is_range:
            return f"{_numeric_token(mention.start_token, 'min_date')} to {_numeric_token(mention.end_token, 'max_date')}"
        return _numeric_token(mention.raw, "min_date")

    if encoding == "fuzzy":
        if mention.is_range:
            return f"{_fuzzy_token(mention.start_token)} to {_fuzzy_token(mention.end_token)}"
        return _fuzzy_token(mention.raw)

    if encoding == "language":
        if mention.is_range:
            return f"{_language_token(mention.start_token)} to {_language_token(mention.end_token)}"
        return _language_token(mention.raw)

    raise ValueError(f"no token rewrite for encoding {encoding!r}")



def _rewrite(text: str, mentions: list[DateMention], encoding: str) -> str:
    parts: list[str] = []
    cursor = 0
    for mention in mentions:
        try:
            replacement = _rewrite_mention(mention, encoding)
        except (ValueError, KeyError):
            continue
        parts.append(text[cursor:mention.start])
        parts.append(replacement)
        cursor = mention.end
    parts.append(text[cursor:])
    return "".join(parts)


def _delete(text: str, mentions: list[DateMention]) -> str:
    parts: list[str] = []
    cursor = 0
    for mention in mentions:
        start = mention.start
        prefix_re = _RANGE_PREP if mention.is_range else _TOKEN_PREP
        window = text[max(cursor, start - _PREP_LOOKBEHIND):start]
        match = prefix_re.search(window)
        if match:
            start = max(cursor, start - (len(window) - match.start()))
        parts.append(text[cursor:start])
        cursor = mention.end
    parts.append(text[cursor:])

    out = "".join(parts)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([?.,;:])", r"\1", out)
    return out.strip()


def mutate_paragraph(text: str, encoding: str) -> str:
    if encoding == "source":
        return text
    if encoding not in CORPUS_ENCODINGS:
        raise ValueError(f"Unknown corpus encoding {encoding!r}, expected one of {CORPUS_ENCODINGS}")

    mentions = find_date_mentions(text)
    if not mentions:
        return text
    if encoding == "delete":
        return _delete(text, mentions)
    return _rewrite(text, mentions, encoding)



@dataclass(frozen=True)
class EncodedCorpus:
    encoding: str
    docs: list[dict]
    dated_count: int
    changed_count: int

    @property
    def coverage(self) -> float:
        return self.changed_count / self.dated_count if self.dated_count else 1.0


def apply_corpus_encoding(docs: list[dict], encoding: str) -> EncodedCorpus:
    if encoding not in CORPUS_ENCODINGS:
        raise ValueError(f"Unknown corpus encoding {encoding!r}, expected one of {CORPUS_ENCODINGS}")

    encoded: list[dict] = []
    dated = changed = 0
    for doc in docs:
        metadata = doc.get("metadata", {})
        text = doc["text"] if encoding == "source" else mutate_paragraph(doc["text"], encoding)
        encoded.append({**doc, "text": text, "metadata": dict(metadata)})
        if "date_start" in metadata:
            dated += 1
            if text != doc["text"]:
                changed += 1

    return EncodedCorpus(encoding=encoding, docs=encoded, dated_count=dated, changed_count=changed)
