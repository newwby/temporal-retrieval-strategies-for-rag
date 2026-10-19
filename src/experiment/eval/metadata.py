from datetime import datetime
from itertools import product
import json
import re
from pathlib import Path
from experiment.classes.class_query import ExperimentRunMetadata
from experiment.preprocessing.corpus_encoding import CORPUS_ENCODINGS
from experiment.preprocessing.phrasing import PHRASING_CONDITIONS
from experiment.preprocessing.question_form import QUESTION_FORMS
from experiment.util.consts import RESULTS_OUTPUT_DIR
from experiment.util.storage import slugify

now = datetime.now()
formatted_date = now.strftime("%Y_%m_%d")
formatted_time = now.strftime("%H-%M-%S")

METADATA_WRITE_PATH = Path(RESULTS_OUTPUT_DIR / formatted_date / formatted_time / "_metadata.jsonl")

MANIFEST_PATH = Path(__file__).parent.parent / "manifest.json"

METADATA_RECORD = {
    "undefined": ExperimentRunMetadata(datestamp=formatted_date, timestamp=formatted_time, model_name="all-MiniLM-L6-v2", file_stem="undefined_results"),
}


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest is None:
        raise ValueError(f"Failed to load manifest.json from {path}, cannot proceed with test harness")
    return manifest


def _lookup_suite_entry(name: str, manifest: dict, path: Path) -> dict:
    if name not in manifest:
        available = sorted(k for k in manifest.keys() if not k.startswith("_"))
        raise KeyError(f"Unknown suite {name!r} in {path}. Available suites: {available}")
    return manifest[name]


VALID_TEMPORAL_MODES = ("none", "exclude", "rerank")


def _warn_if_entity_filter_combined(name: str, entity_filter: bool, temporal_mode: str, phrasing: str) -> None:
    if not entity_filter:
        return
    other_arm = None
    if temporal_mode != "none":
        other_arm = f"temporal_filter={temporal_mode!r}"
    elif phrasing != "source":
        other_arm = f"phrasing={phrasing!r}"
    if other_arm:
        print(
            f"Warning: suite {name!r} combines entity_filter with {other_arm}. Entity filtering is "
            "oracle-like and its results are an upper "
            "bound, not a retrieval condition. Do not report this as a comparable arm."
        )


def _normalize_temporal_mode(value: bool | str) -> str:
    if isinstance(value, bool):
        mode = "exclude" if value else "none"
        print(f"Warning: manifest 'temporal_filter': {value!r} is deprecated - use one of {VALID_TEMPORAL_MODES}. Interpreted as {mode!r}.")
        return mode
    if value not in VALID_TEMPORAL_MODES:
        raise ValueError(f"Invalid 'temporal_filter' value {value!r} - must be a bool (deprecated) or one of {VALID_TEMPORAL_MODES}.")
    return value


def load_manifest_suite(name: str, path: Path = MANIFEST_PATH) -> list[ExperimentRunMetadata]:
    manifest = _load_manifest(path)
    entry = _lookup_suite_entry(name, manifest, path)
    model_aliases = entry.get("model_aliases", {})
    declared_metrics = entry.get("metrics", [])
    gold_filter = entry.get("gold_filter")

    field_names = [key for key in ("model_name", "k", "retriever", "entity_filter", "temporal_filter", "corpus_encoding", "question_form", "phrasing") if key in entry]
    field_values = []
    for key in field_names:
        value = entry[key]
        field_values.append(value if isinstance(value, list) else [value])

    combinations = list(product(*field_values))
    multi_run = len(combinations) > 1
    multi_retriever = "retriever" in entry and isinstance(entry["retriever"], list) and len(entry["retriever"]) > 1
    multi_entity_filter = "entity_filter" in entry and isinstance(entry["entity_filter"], list) and len(entry["entity_filter"]) > 1
    multi_temporal_filter = "temporal_filter" in entry and isinstance(entry["temporal_filter"], list) and len(entry["temporal_filter"]) > 1
    multi_phrasing = "phrasing" in entry and isinstance(entry["phrasing"], list) and len(entry["phrasing"]) > 1
    multi_question_form = "question_form" in entry and isinstance(entry["question_form"], list) and len(entry["question_form"]) > 1
    multi_corpus_encoding = "corpus_encoding" in entry and isinstance(entry["corpus_encoding"], list) and len(entry["corpus_encoding"]) > 1

    runs = []
    for combination in combinations:
        values = dict(zip(field_names, combination))
        model_name = values.get("model_name", "all-MiniLM-L6-v2")
        k = values.get("k")
        retriever = values.get("retriever", "faiss")
        entity_filter = values.get("entity_filter", False)
        temporal_mode = _normalize_temporal_mode(values.get("temporal_filter", "none"))
        phrasing = values.get("phrasing", "source")
        question_form = values.get("question_form", "timeqa_source")
        corpus_encoding = values.get("corpus_encoding", "source")
        if phrasing not in PHRASING_CONDITIONS:
            raise ValueError(f"Invalid 'phrasing' value {phrasing!r} in suite {name!r} - must be one of {PHRASING_CONDITIONS}.")
        if question_form not in QUESTION_FORMS:
            raise ValueError(f"Invalid 'question_form' value {question_form!r} in suite {name!r} - must be one of {QUESTION_FORMS}.")
        if corpus_encoding not in CORPUS_ENCODINGS:
            raise ValueError(f"Invalid 'corpus_encoding' value {corpus_encoding!r} in suite {name!r} - must be one of {CORPUS_ENCODINGS}.")
        _warn_if_entity_filter_combined(name, entity_filter, temporal_mode, phrasing)

        if multi_run:
            model_label = model_aliases.get(model_name, slugify(model_name))
            file_stem = f"{name}_k{k}_{model_label}"
            if multi_retriever:
                file_stem = f"{file_stem}_{slugify(retriever)}"
            if multi_entity_filter:
                file_stem = f"{file_stem}_entityfilter-{'on' if entity_filter else 'off'}"
            if multi_temporal_filter:
                file_stem = f"{file_stem}_temporalmode-{temporal_mode}"
            if multi_corpus_encoding:
                file_stem = f"{file_stem}_encoding-{corpus_encoding}"
            if multi_question_form:
                file_stem = f"{file_stem}_qform-{question_form}"
            if multi_phrasing:
                file_stem = f"{file_stem}_phrasing-{phrasing}"
        else:
            file_stem = f"{name}_results"

        runs.append(ExperimentRunMetadata(
            datestamp=formatted_date,
            timestamp=formatted_time,
            model_name=model_name,
            file_stem=file_stem,
            k=k,
            retriever=retriever,
            entity_filter=entity_filter,
            temporal_mode=temporal_mode,
            corpus_encoding=corpus_encoding,
            question_form=question_form,
            phrasing=phrasing,
            declared_metrics=declared_metrics,
            gold_filter=gold_filter,
        ))

    return runs

def get_suite_query_source(name: str, path: Path = MANIFEST_PATH) -> str:
    manifest = _load_manifest(path)
    entry = _lookup_suite_entry(name, manifest, path)
    return entry.get("query_source", "dev")
