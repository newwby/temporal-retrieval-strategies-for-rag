from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import urllib.parse

from experiment.preprocessing.build_queries import load_relations
from experiment.util.consts import QUESTION_FORMS_PATH, RELATIONS_PATH

"""
renders a query's question in a chosen form, the question_form axis
question_form sets how the question is built, phrasing then rewrites the date inside it,
eval_harness.py applies them in that order

timeqa's questions come from templates, each relation in relations.json has a template list and
its mode picks the time slot ($4 range for "accumulate", $2 point for "pointwise")

over all 3021 dev rows (see test_question_form.py) render_from_template() matches timeqa's
question for 2974 (98.44%). the other 47 are ranges timeqa wrote as a point, e.g.
("Feb 2001", "Apr 2001") as "in Feb 2001". timeqa picks this at random (same year ranges split
67 "from x to y" / 37 "in x", 10 more across years) so no rule can match it

so the timeqa_source control is dataset_verbatim, it returns the row's own question_text
rather than re-rendering, the 98.44% just shows the slot logic is right
"""

__all__ = [
    "QUESTION_FORMS", "RenderedQuestion", "load_question_forms", "point_expression",
    "range_expression", "render_from_template", "render_question", "subject_from_link",
]

QUESTION_FORMS = ("timeqa_source", "timeqa_alt", "declarative", "keyword", "time_first", "no_relation")

_JAN_YEAR = re.compile(r"^Jan\s+(\d{4})$")

_FUNCTION_WORDS = re.compile(
    r"\b(?:the|of|a|an|for|to|in|on|at|from|and|was|were|is|are|did|do|does)\b", re.IGNORECASE
)

_MODE_FALLBACK = "accumulate"

_forms_cache: dict | None = None
_relations_cache: dict | None = None


def load_question_forms(path: Path | None = None) -> dict:
    global _forms_cache
    if _forms_cache is not None and path is None:
        return _forms_cache
    forms_path = path or Path(QUESTION_FORMS_PATH)
    with forms_path.open("r", encoding="utf-8") as f:
        forms = {name: entry for name, entry in json.load(f).items() if not name.startswith("_")}
    if path is None:
        _forms_cache = forms
    return forms


def _relations() -> dict:
    global _relations_cache
    if _relations_cache is None:
        _relations_cache = load_relations(Path(RELATIONS_PATH))
    return _relations_cache


def subject_from_link(link: str) -> str:
    slug = link[len("/wiki/"):] if link.startswith("/wiki/") else link
    return urllib.parse.unquote(slug).replace("_", " ")


def _bare(token: str) -> str:
    return _JAN_YEAR.sub(r"\1", token)


def range_expression(time_start_raw: str, time_end_raw: str) -> str:
    start, end = _bare(time_start_raw), _bare(time_end_raw)
    if start == end:
        return f"in {start}"
    return f"from {start} to {end}"


def point_expression(time_start_raw: str) -> str:
    return f"in {_bare(time_start_raw)}"


def _time_expression(row: dict, mode: str) -> str:
    if mode == "pointwise":
        return point_expression(row["time_start_raw"])
    return range_expression(row["time_start_raw"], row["time_end_raw"])


_SLOT = re.compile(r"\$(?:1|2|4|m)")


def _substitute(template: str, values: dict[str, str]) -> str:
    return _SLOT.sub(lambda match: values.get(match.group(0), match.group(0)), template)


def _tidy(text: str) -> str:
    text = re.sub(r"\s{2,}", " ", text)
    return re.sub(r"\s+([?.,;:])", r"\1", text).strip()


@dataclass(frozen=True)
class RenderedQuestion:
    text: str
    applied: bool
    applicable: bool


def render_from_template(row: dict, index: int, relations: dict | None = None) -> str | None:
    relations = relations if relations is not None else _relations()
    entry = relations.get(row["relation_code"])
    if not entry:
        return None
    templates = entry.get("template") or []
    if index >= len(templates):
        return None

    mode = entry.get("mode") or _MODE_FALLBACK
    time_expression = _time_expression(row, mode)
    return _substitute(templates[index], {
        "$1": subject_from_link(row["link"]),
        "$2": time_expression,
        "$4": time_expression,
        "$m": entry.get("meaning") or "",
    })


def _first_differing_template(row: dict, original: str, relations: dict | None = None) -> str | None:
    relations = relations if relations is not None else _relations()
    entry = relations.get(row["relation_code"]) or {}
    for index in range(len(entry.get("template") or [])):
        candidate = render_from_template(row, index, relations)
        if candidate is not None and _tidy(candidate) != original:
            return candidate
    return None


def render_question(
    row: dict,
    form: str,
    forms: dict | None = None,
    relations: dict | None = None,
) -> RenderedQuestion:
    forms = forms if forms is not None else load_question_forms()
    if form not in forms:
        raise ValueError(f"Unknown question_form {form!r}, expected one of {sorted(forms)}")

    entry = forms[form]
    kind = entry["kind"]
    original = row["question_text"]

    if kind == "dataset_verbatim":
        return RenderedQuestion(original, applied=False, applicable=True)

    if kind == "dataset_template":
        index = entry["template_index"]
        if index == "first_differing":
            text = _first_differing_template(row, original, relations)
        else:
            text = render_from_template(row, index, relations)
        if text is None:
            return RenderedQuestion(original, applied=False, applicable=False)
        text = _tidy(text)
        return RenderedQuestion(text, applied=text != original, applicable=True)

    if kind == "authored":
        mode = row.get("relation_mode") or _MODE_FALLBACK
        template = entry.get(mode) or entry[_MODE_FALLBACK]
        text = _substitute(template, {
            "$1": subject_from_link(row["link"]),
            "$2": point_expression(row["time_start_raw"]),
            "$4": range_expression(row["time_start_raw"], row["time_end_raw"]),
            "$m": row.get("relation_meaning") or "",
        })
        if entry.get("strip_function_words"):
            text = _FUNCTION_WORDS.sub(" ", text)
            text = re.sub(r"[^\w\s-]", " ", text)
        return RenderedQuestion(_tidy(text), applied=_tidy(text) != original, applicable=True)

    raise ValueError(f"Unknown question_form kind {kind!r} for form {form!r}")
