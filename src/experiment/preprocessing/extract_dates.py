from dataclasses import dataclass

from experiment.util.date_utils import (
    DATE_RE, MONTH_RE, RANGE_RE, SINGLE_DATE_RE, normalize_date, plausible_year,
)

"""
finds dates in corpus paragraphs for a (date_start, date_end) chroma metadata span per paragraph,
like entity/link/para_index in storage.py _row_metadata() (phrasing.py does this for questions)

1. years only, free text is too noisy for month/day. years before 1000 or after 2029 aren't
   matched (DATE_RE in date_utils.py), timeqa has none
2. no date -> (None, None) and the field is left out. unlike entity_filter (falls back to the
   full index for an unknown entity), a paragraph with no span is dropped by any chroma $lte/$gte
   filter, see _resolve_index in eval_harness.py. the "rerank" mode avoids losing recall to this.
   several dates -> min/max across all, leans to including too much over excluding, better recall
3. two passes: (a) "x to y" ranges (RANGE_RE, from strip_questions.py), (b) single dates not
   inside an (a) match, as free text rarely uses "to". normalize_date() parses both
"""

__all__ = [
    "MONTH_RE", "DATE_RE", "RANGE_RE", "SINGLE_DATE_RE",
    "DateMention", "extract_paragraph_date_span", "find_date_mentions",
]


def _year_span(normalized) -> tuple[int, int]:
    return int(normalized.min_date[:4]), int(normalized.max_date[:4])


@dataclass(frozen=True)
class DateMention:
    start: int
    end: int
    raw: str
    is_range: bool
    start_token: str
    end_token: str


"""
every date in a paragraph, in order and not overlapping
the same two passes as above, kept separate so corpus_encoding.py can reuse it to rewrite dates

returns spans so rewrites use offsets from the original string, replaced text is never
re-scanned so the "1" -> "0001" -> "0000001" bug (see _substitute_spans in phrasing.py) can't happen
"""
def find_date_mentions(text: str) -> list[DateMention]:
    mentions: list[DateMention] = []
    consumed_spans: list[tuple[int, int]] = []

    for match in RANGE_RE.finditer(text):
        consumed_spans.append(match.span())
        mentions.append(DateMention(
            start=match.start(),
            end=match.end(),
            raw=match.group(0),
            is_range=True,
            start_token=match.group("start"),
            end_token=match.group("end"),
        ))

    def _inside_consumed(pos: int) -> bool:
        return any(span_start <= pos < span_end for span_start, span_end in consumed_spans)

    for match in SINGLE_DATE_RE.finditer(text):
        if _inside_consumed(match.start()):
            continue
        token = match.group(0)
        mentions.append(DateMention(
            start=match.start(),
            end=match.end(),
            raw=token,
            is_range=False,
            start_token=token,
            end_token=token,
        ))

    mentions.sort(key=lambda mention: mention.start)
    return mentions


def extract_paragraph_date_span(text: str) -> tuple[int, int] | tuple[None, None]:
    years: list[int] = []

    for mention in find_date_mentions(text):
        try:
            if mention.is_range:
                start = normalize_date(mention.start_token)
                end = normalize_date(mention.end_token)
            else:
                start = end = normalize_date(mention.raw)
        except (ValueError, KeyError):
            continue
        start_min, start_max = _year_span(start)
        end_min, end_max = _year_span(end)
        years.extend([start_min, start_max, end_min, end_max])

    years = [year for year in years if plausible_year(year)]

    if not years:
        return None, None

    return min(years), max(years)
