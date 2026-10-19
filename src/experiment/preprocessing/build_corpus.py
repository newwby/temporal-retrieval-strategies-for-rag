import json
from pathlib import Path

from experiment.preprocessing.build_manifest import DEFAULT_BUILD_CONFIG, load_build_manifest, resolve_build_sources
from experiment.preprocessing.extract_dates import extract_paragraph_date_span
from experiment.preprocessing.temporal_coverage import (
    WIDE_SPAN_THRESHOLD_YEARS, compute_corpus_date_stats, resolve_temporal_coverage_path,
    write_temporal_coverage_stats,
)
from experiment.util.consts import CORPUS_OUTPUT_DIR, CORPUS_OUTPUT_PATH, DEBUG_MODE

"""
turns annotated_dev.json's article paragraphs into a flat corpus, one row per paragraph,
that storage.py's load_data() can read directly

"text" is embedded as is, entity/link/para_index/date_start/date_end go into chroma metadata
(dates from extract_dates.py)

a build config can combine several annotated_<split>.json files into one corpus
"""

__all__ = [
    "build_corpus_rows",
    "load_and_validate_entities",
    "load_and_validate_sources",
    "load_annotated_entities",
    "main",
    "resolve_corpus_output_path",
    "validate_annotated_entities",
]


def load_annotated_entities(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse {path} as a single JSON document ({e}). If this file looks like "
                "JSONL (one JSON object per line, e.g. a *.easy.json question-join file), it was "
                "likely pointed at as an annotated_path by mistake."
            ) from e


"""
checks one annotated_<split>.json before it's combined into a build, returns the valid entities
1 - wrong top level shape (wrong file), reject the file
2 - some entries malformed, skip just those
"""
def validate_annotated_entities(entities, source_path: Path) -> list[dict]:
    if not isinstance(entities, list):
        raise ValueError(
            f"Annotated entities file {source_path} is not a JSON list at the top level "
            f"(got {type(entities).__name__}) - check this is an annotated_<split>.json file, "
            "not e.g. a *.easy.json question-join file or some other JSON document."
        )

    required_entity_keys = {"index", "link", "paras", "questions", "type"}

    def _entity_has_required_shape(entity) -> bool:
        if not isinstance(entity, dict) or not required_entity_keys.issubset(entity.keys()):
            return False
        for question in entity["questions"]:
            if not isinstance(question, (list, tuple)) or len(question) != 2:
                return False
            time_range, evidence_list = question
            if not isinstance(time_range, (list, tuple)) or len(time_range) != 2:
                return False
            if not isinstance(evidence_list, list):
                return False
            for evidence in evidence_list:
                if not isinstance(evidence, dict) or "para" not in evidence or "answer" not in evidence:
                    return False
        return True

    valid_entities = []
    skipped = 0
    for i, entity in enumerate(entities):
        if _entity_has_required_shape(entity):
            valid_entities.append(entity)
        else:
            skipped += 1
            entity_label = entity.get("index") if isinstance(entity, dict) else f"item #{i}"
            print(
                f"Warning: skipping malformed entity ({entity_label!r}) in {source_path} - "
                "missing required keys or a 'questions' entry that isn't a "
                "[[start, end], evidence_list] pair with 'para'/'answer' evidence items"
            )

    if entities and not valid_entities:
        raise ValueError(
            f"Annotated entities file {source_path} loaded as a list of {len(entities)} item(s) but "
            "NONE had the required schema (index/link/paras/questions/type keys, with 'questions' "
            "entries shaped as [[start, end], evidence_list] pairs) - check this is an "
            "annotated_<split>.json file, not e.g. a human_annotated_*.json file (different, "
            "incompatible schema) or some other JSON file."
        )

    if skipped:
        print(
            f"Validation: skipped {skipped} malformed entit{'y' if skipped == 1 else 'ies'} out of "
            f"{len(entities)} in {source_path} ({len(valid_entities)} valid entities kept)"
        )

    return valid_entities


def load_and_validate_entities(source: dict) -> list[dict]:
    annotated_path = Path(source["annotated_path"])
    raw_entities = load_annotated_entities(annotated_path)
    return validate_annotated_entities(raw_entities, annotated_path)


def load_and_validate_sources(sources: list[dict]) -> list[dict]:
    combined_entities = []
    for source in sources:
        combined_entities.extend(load_and_validate_entities(source))
    return combined_entities


def resolve_corpus_output_path(config_name: str) -> Path:
    if config_name == DEFAULT_BUILD_CONFIG:
        return CORPUS_OUTPUT_PATH
    return CORPUS_OUTPUT_DIR / f"{config_name}_corpus.jsonl"


def slugify_link(link: str) -> str:
    return link.replace("/wiki/", "")


def corpus_id(entity_slug: str, para_index: int) -> str:
    return f"{entity_slug}__para{para_index}"


"""
a few entities show up twice in annotated_dev.json under different relation codes (e.g. one
link tagged P127 and P1448 with the same paras), dedupe by link so no paragraph id is written twice
"""
def build_corpus_rows(entities: list[dict]) -> list[dict]:
    seen_links: dict[str, list] = {}
    for entry in entities:
        link = entry["link"]
        if link in seen_links and seen_links[link] != entry["paras"]:
            raise ValueError(
                f"Duplicate link {link} has differing paras across entries - "
                "dedupe assumption no longer holds, needs investigation."
            )
        seen_links[link] = entry["paras"]

    rows = []
    for link, paras in seen_links.items():
        slug = slugify_link(link)
        for para_index, para_text in enumerate(paras):
            row = {
                "id": corpus_id(slug, para_index),
                "text": para_text,
                "entity": slug,
                "link": link,
                "para_index": para_index,
            }
            date_start, date_end = extract_paragraph_date_span(para_text)
            if date_start is not None:
                row["date_start"] = date_start
                row["date_end"] = date_end
            rows.append(row)
    return rows


def write_corpus(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(config_name: str = DEFAULT_BUILD_CONFIG) -> None:
    manifest = load_build_manifest()
    sources = resolve_build_sources(config_name, manifest)
    entities = load_and_validate_sources(sources)

    rows = build_corpus_rows(entities)
    output_path = resolve_corpus_output_path(config_name)
    write_corpus(rows, output_path)

    unique_entities = len({row["entity"] for row in rows})
    print(f"Build config: {config_name!r} ({len(sources)} source file(s))")
    print(f"Entities read: {len(entities)} (unique: {unique_entities})")
    print(f"Corpus paragraphs written: {len(rows)}")
    print(f"Output path: {output_path}")

    date_stats = compute_corpus_date_stats(rows)
    print(
        f"Dated: {date_stats['dated']} ({date_stats['dated_pct']:.1%}) / "
        f"Undated: {date_stats['undated']} ({date_stats['undated_pct']:.1%})"
    )
    print(
        f"Span > {WIDE_SPAN_THRESHOLD_YEARS}y: {date_stats['span_over_100y']} "
        f"({date_stats['span_over_100y_pct_of_dated']:.1%} of dated, "
        f"{date_stats['span_over_100y_pct_of_all']:.1%} of all)"
    )
    print(
        f"Span width (dated only) min/max/median: {date_stats['span_width_min']}/"
        f"{date_stats['span_width_max']}/{date_stats['span_width_median']}"
    )
    write_temporal_coverage_stats(config_name, "corpus", date_stats)
    print(f"Temporal coverage stats written to: {resolve_temporal_coverage_path(config_name)}")


if __name__ == "__main__":
    main()
