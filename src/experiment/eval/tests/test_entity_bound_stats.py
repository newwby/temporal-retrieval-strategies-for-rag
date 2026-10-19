import json

import pytest

from experiment.analysis.entity_filter_bound import (
    compute_k_curve,
    compute_summary,
    load_article_sizes,
    load_scored_queries,
    structural_hit_probability,
)
from experiment.eval.metadata import load_manifest_suite


def _write_corpus(path, sizes: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for entity, size in sizes.items():
            for i in range(size):
                f.write(json.dumps({"id": f"{entity}__para{i}", "entity": entity, "text": "x"}) + "\n")


def _write_queries(path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def small_corpus(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    sizes = {"small": 2, "medium": 5, "large": 50}
    _write_corpus(corpus_path, sizes)
    _write_queries(queries_path, [
        {"entity": "small", "gold_doc_ids": ["small__para0"]},
        {"entity": "medium", "gold_doc_ids": ["medium__para0"]},
        {"entity": "large", "gold_doc_ids": ["large__para0"]},
        {"entity": "large", "gold_doc_ids": []},
    ])
    return corpus_path, queries_path, sizes


def test_load_article_sizes(small_corpus):
    corpus_path, _, sizes = small_corpus
    assert load_article_sizes(corpus_path) == sizes


def test_load_scored_queries_excludes_unscored(small_corpus):
    _, queries_path, _ = small_corpus
    scored = load_scored_queries(queries_path)
    assert len(scored) == 3
    assert all(q["gold_doc_ids"] for q in scored)


def test_le_k_share_and_mean_length(small_corpus):
    corpus_path, queries_path, _ = small_corpus
    article_sizes = load_article_sizes(corpus_path)
    queries = load_scored_queries(queries_path)

    k_curve = compute_k_curve(article_sizes, queries, k_values=(5,))
    row = k_curve[0]

    assert row["pct_queries_article_le_k"] == pytest.approx(200 / 3)
    assert row["mean_filtered_length"] == pytest.approx(4.0)


def test_summary_stats(small_corpus):
    corpus_path, _, _ = small_corpus
    article_sizes = load_article_sizes(corpus_path)
    summary = compute_summary(article_sizes)
    assert summary["distinct_entities"] == 3
    assert summary["total_paragraphs"] == 57
    assert summary["median_paragraphs_per_article"] == 5


def test_structural_hit_probability():
    assert structural_hit_probability(article_size=5, gold_count=1, k=10) == 1.0
    assert structural_hit_probability(article_size=100, gold_count=1, k=10) == pytest.approx(0.10)
    single = structural_hit_probability(100, 1, 10)
    multi = structural_hit_probability(100, 3, 10)
    assert multi > single


def test_warning_fires_when_entity_filter_combined(tmp_path, capsys):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "mixed_suite": {
            "model_name": "all-MiniLM-L6-v2",
            "k": 30,
            "entity_filter": True,
            "temporal_filter": "exclude",
        }
    }), encoding="utf-8")

    load_manifest_suite("mixed_suite", path=manifest_path)
    captured = capsys.readouterr()
    assert "combines entity_filter" in captured.out


def test_warning_silent_for_dedicated_suite(tmp_path, capsys):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "entity_upper_bound": {
            "model_name": "all-MiniLM-L6-v2",
            "k": 30,
            "entity_filter": [True, False],
        }
    }), encoding="utf-8")

    load_manifest_suite("entity_upper_bound", path=manifest_path)
    captured = capsys.readouterr()
    assert "combines entity_filter" not in captured.out
