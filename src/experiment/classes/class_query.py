from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

from pypika import JSON

from experiment.util.consts import RESULTS_OUTPUT_DIR

@dataclass
class EvalQuery:
    query_id: str
    relation_code: str

    relation_mode: str

    time_start: str
    time_end: str

    time_start_raw: str
    time_end_raw: str

    question_text: str
    link: str | None = None
    relation_meaning: str | None = None
    entity: str | None = None

    gold_doc_ids: list[str] = field(default_factory=list)
    gold_answers: list[str] = field(default_factory=list)
    has_evidence: bool = False

    hit_at_k: bool | None = None
    precision_at_k: float | None = None
    recall_at_k: float | None = None
    mrr: float | None = None
    answer_f1: float | None = None
    answer_exact_match: float | None = None
    chrf: float | None = None
    answer_bertscore_p: float | None = None
    answer_bertscore_r: float | None = None
    answer_bertscore_f1: float | None = None
    latency: float | None = None
    index_resolve_latency: float | None = None
    mutate_latency: float | None = None
    retrieval_latency: float | None = None
    total_latency: float | None = None
    index_cache_hit: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.relation_mode, str):
            raise TypeError(
                f"EvalQuery.relation_mode must be a str (relations.json 'mode'), got "
                f"{type(self.relation_mode).__name__} for query_id {self.query_id!r} - "
                f"pass entry['mode'], not the whole relations.json entry."
            )

    @staticmethod
    def row_to_structured_query(row: dict, relation_mode: str | None = None) -> EvalQuery:
        if relation_mode is None:
            relation_mode = row["relation_mode"]

        return EvalQuery(
            query_id=row["query_id"],
            relation_code=row["relation_code"],
            relation_mode=relation_mode,
            question_text=row["question_text"],
            link=row.get("link"),
            relation_meaning=row.get("relation_meaning"),
            time_start=row["start_iso_value"],
            time_end=row["end_iso_value"],
            time_start_raw=row["time_start_raw"],
            time_end_raw=row["time_end_raw"],
            entity=row.get("entity"),
            gold_doc_ids=list(row["gold_doc_ids"]),
            gold_answers=list(row.get("gold_answers", [])),
            has_evidence=row.get("has_evidence", False),
        )


@dataclass
class ExperimentRunMetadata:
    datestamp: str
    timestamp: str
    model_name: str
    file_stem: str
    k: int | None = None
    retriever: str = "faiss"
    entity_filter: bool = False
    temporal_mode: str = "none"
    corpus_encoding: str = "source"
    question_form: str = "timeqa_source"
    phrasing: str = "source"
    invalid_reason: str | None = None
    declared_metrics: list[str] = field(default_factory=list)
    gold_filter: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


    def get_printable(self) -> JSON:
        return {
            "id": self.file_stem,
            "metrics": self.metrics,
            "datestamp": self.datestamp,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "k": self.k,
            "retriever": self.retriever,
            "entity_filter": self.entity_filter,
            "temporal_mode": self.temporal_mode,
            "corpus_encoding": self.corpus_encoding,
            "question_form": self.question_form,
            "phrasing": self.phrasing,
            "invalid_reason": self.invalid_reason,
            "declared_metrics": self.declared_metrics,
            "gold_filter": self.gold_filter,
        }


    def get_file_path(self) -> Path:
        output_path = RESULTS_OUTPUT_DIR / self.datestamp / self.timestamp / f"{self.file_stem}.jsonl"
        return output_path


    def output_to_json(self, output_dir: Path) -> None:
        output_path = self.get_file_path(output_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=4)   

