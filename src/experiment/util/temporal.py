
__all__ = ["bucket_for_span", "rerank_by_span"]


def bucket_for_span(span: tuple[int, int] | None, window: tuple[int, int]) -> int:
    if span is None:
        return 1
    span_start, span_end = span
    window_start, window_end = window
    return 2 if (span_start <= window_end and span_end >= window_start) else 0


def rerank_by_span(items: list, window: tuple[int, int] | None, get_span) -> list:
    if window is None:
        return items
    indexed = sorted(enumerate(items), key=lambda pair: (-bucket_for_span(get_span(pair[1]), window), pair[0]))
    return [item for _, item in indexed]
