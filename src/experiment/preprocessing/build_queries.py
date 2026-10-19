import gzip
import json
from pathlib import Path
from typing import Iterator

from experiment.preprocessing.build_corpus import (
    corpus_id,
    load_and_validate_entities,
    resolve_corpus_output_path,
    slugify_link,
)
from experiment.preprocessing.build_manifest import DEFAULT_BUILD_CONFIG, load_build_manifest, resolve_build_sources
from experiment.util.date_utils import normalize_date
from experiment.util.consts import (
    CORPUS_FILENAME,
    EXPECTED_MULTI_EVIDENCE,
    EXPECTED_TOTAL_QUESTIONS,
    EXPECTED_WITHOUT_EVIDENCE,
    EXPECTED_WITH_EVIDENCE,
    QUERIES_OUTPUT_DIR,
    QUERIES_OUTPUT_PATH,
    RELATIONS_PATH,
)

"""
turns each time range question in annotated_dev.json into a query row: the normalised range,
the question text from dev.easy.json, and the gold corpus doc ids that answer it (used for
hit@k / mrr / em later)

main() takes a build config name (default "dev"), question indexes from each source are merged
and gold_doc_ids are prefixed with that config's corpus file name
"""

__all__ = ["build_query_rows", "load_dev_easy_questions", "main", "resolve_queries_output_path"]


def load_dev_easy_questions(path: Path) -> dict[str, str]:
    index = {}
    opener = gzip.open if path.suffix == ".gzip" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            index[obj["idx"]] = obj["question"]
    return index


def load_relations(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        relations = json.load(f)
    validate_relations(relations, path)
    return relations


"""
relations.json is timeqa's own metadata, its "template" lists made every question in
dev.easy.json / train.easy.json. "mode" picks the template, question_form.py relies on:

  - "accumulate" templates take a range, slot $4
  - "pointwise" templates take a point, slot $2
  - every template has one $1 (the subject) and one time slot
  - "n/a" relations have no templates (18 of 102, none in dev)

true for all 102 entries, checked at build time in case a newer timeqa copy breaks it
"""
TIME_SLOT_FOR_MODE = {"accumulate": "$4", "pointwise": "$2"}


def validate_relations(relations: dict, path: Path) -> None:
    problems = []
    for code, entry in relations.items():
        mode = entry.get("mode")
        templates = entry.get("template", [])

        if mode not in TIME_SLOT_FOR_MODE:
            if templates:
                problems.append(f"{code}: mode {mode!r} should have no templates, found {len(templates)}")
            continue

        if not templates:
            problems.append(f"{code}: mode {mode!r} has no templates")
            continue

        expected_slot = TIME_SLOT_FOR_MODE[mode]
        other_slot = "$2" if expected_slot == "$4" else "$4"
        for template in templates:
            if template.count("$1") != 1:
                problems.append(f"{code}: template {template!r} needs exactly one $1")
            if template.count(expected_slot) != 1:
                problems.append(f"{code}: mode {mode!r} template {template!r} needs exactly one {expected_slot}")
            if other_slot in template:
                problems.append(f"{code}: mode {mode!r} template {template!r} must not use {other_slot}")

    if problems:
        raise ValueError(
            f"relations.json at {path} violates the mode/template invariants "
            f"question_form.py depends on:\n  " + "\n  ".join(problems)
        )


def prefixed_corpus_id(raw_id: str, corpus_filename: str) -> str:
    return f"{corpus_filename}_{raw_id}"


def build_query_rows(entities: list[dict], dev_easy_index: dict[str, str], relations: dict, corpus_filename: str = CORPUS_FILENAME) -> Iterator[dict]:
    for entry in entities:
        slug = slugify_link(entry["link"])
        relation_code = entry["type"]
        relation_meaning = relations.get(relation_code, {}).get("meaning")
        relation_mode = relations.get(relation_code, {}).get("mode")

        for position, (time_range, evidence_list) in enumerate(entry["questions"]):
            query_id = f"{entry['index']}#{position}"
            start_text, end_text = time_range
            start = normalize_date(start_text)
            end = normalize_date(end_text)

            has_evidence = any(ev["answer"] != "" for ev in evidence_list)
            gold_doc_ids = [
                prefixed_corpus_id(corpus_id(slug, ev["para"]), corpus_filename)
                for ev in evidence_list if ev["answer"] != ""
            ]
            gold_answers = [ev["answer"] for ev in evidence_list if ev["answer"] != ""]

            question_text = dev_easy_index.get(query_id)

            yield {
                "query_id": query_id,
                "entity": slug,
                "link": entry["link"],
                "relation_code": relation_code,
                "relation_meaning": relation_meaning,
                "relation_mode": relation_mode,
                "question_text": question_text,
                "time_start_raw": start_text,
                "time_end_raw": end_text,
                "start_precision": start.precision,
                "start_iso_value": start.iso_value,
                "start_min_date": start.min_date,
                "start_max_date": start.max_date,
                "end_precision": end.precision,
                "end_iso_value": end.iso_value,
                "end_min_date": end.min_date,
                "end_max_date": end.max_date,
                "has_evidence": has_evidence,
                "gold_doc_ids": gold_doc_ids,
                "gold_answers": gold_answers,
                "gold_evidence": evidence_list,
            }


def write_queries(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_queries_output_path(config_name: str) -> Path:
    if config_name == DEFAULT_BUILD_CONFIG:
        return QUERIES_OUTPUT_PATH
    return QUERIES_OUTPUT_DIR / f"{config_name}_queries.jsonl"


def main(config_name: str = DEFAULT_BUILD_CONFIG) -> None:
    manifest = load_build_manifest()
    sources = resolve_build_sources(config_name, manifest)

    entities = []
    dev_easy_index: dict[str, str] = {}
    for source in sources:
        entities.extend(load_and_validate_entities(source))

        easy_path = Path(source["easy_path"])
        source_easy_index = load_dev_easy_questions(easy_path)
        overlap = dev_easy_index.keys() & source_easy_index.keys()
        if overlap:
            print(
                f"Warning: {len(overlap)} question id(s) already seen before merging {easy_path} "
                "into the combined dev_easy_index - the later source's question text wins"
            )
        dev_easy_index.update(source_easy_index)

    relations = load_relations(RELATIONS_PATH)
    corpus_filename = resolve_corpus_output_path(config_name).name

    rows = list(build_query_rows(entities, dev_easy_index, relations, corpus_filename))

    missing_question = [r for r in rows if r["question_text"] is None]
    if missing_question:
        print(f"Warning: {len(missing_question)} rows had no easy-question match for their query_id")

    output_path = resolve_queries_output_path(config_name)
    write_queries(rows, output_path)

    with_evidence = sum(1 for r in rows if r["has_evidence"])
    without_evidence = len(rows) - with_evidence
    multi_evidence = sum(1 for r in rows if len(r["gold_doc_ids"]) > 1)

    print(f"Build config: {config_name!r} ({len(sources)} source file(s))")
    print(f"Query rows written: {len(rows)}")
    print(f"With evidence: {with_evidence} / Without evidence: {without_evidence}")
    print(f"Multi-evidence (>1 gold paragraph): {multi_evidence}")
    print(f"Output path: {output_path}")

    if config_name == DEFAULT_BUILD_CONFIG:
        assert len(rows) == EXPECTED_TOTAL_QUESTIONS, f"expected {EXPECTED_TOTAL_QUESTIONS} rows, got {len(rows)}"
        assert with_evidence == EXPECTED_WITH_EVIDENCE, f"expected {EXPECTED_WITH_EVIDENCE} with evidence, got {with_evidence}"
        assert without_evidence == EXPECTED_WITHOUT_EVIDENCE, f"expected {EXPECTED_WITHOUT_EVIDENCE} without evidence, got {without_evidence}"
        assert multi_evidence == EXPECTED_MULTI_EVIDENCE, f"expected {EXPECTED_MULTI_EVIDENCE} multi-evidence, got {multi_evidence}"


if __name__ == "__main__":
    main()
