"""
compares slm vs regex fuzzy window resolution against the gold window (validation only)

scores a saved fuzzy phrasing run three ways, baseline (no rerank), regex resolver and a local
slm resolver, and reports mrr@5 / hit@1 / hit@5 for each on the same sample and rankings
standalone, no retrieval or model reload, not part of manifest.json or eval_harness.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

from experiment.preprocessing.build_phrasing_cache import _call_backend
from experiment.util.consts import (
    CORPUS_OUTPUT_PATH,
    LLM_BACKEND,
    LLM_BASE_URL,
    LLM_MODEL_NAME,
    OLLAMA_URL,
    QUERIES_OUTPUT_PATH,
    RESULTS_OUTPUT_DIR,
)
from experiment.util.temporal import rerank_by_span
from experiment.visualisation.metrics_at_k import build_rerank_context, load_rows, score_at_k

DEFAULT_RUN = (
    RESULTS_OUTPUT_DIR / "2026_08_25" / "16-53-24"
    / "phrasing_test_k30_all-minilm-l6-v2_phrasing-fuzzy.jsonl"
)

BAND_RE = re.compile(r"\b(early|mid|late)\s+(\d{3,4})s\b", re.IGNORECASE)
BAND_BOUNDS = {"early": (0, 3), "mid": (4, 6), "late": (7, 9)}
JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
YEAR_IN_TEXT_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")

PROMPT_HEADER = (
    "Convert a vague time phrase into an explicit range of four-digit years.\n"
    "'early <decade>s' is the first third of that decade, 'mid <decade>s' the middle third, "
    "'late <decade>s' the last third.\n"
    "Reply with ONLY a JSON object {\"start\": <year>, \"end\": <year>} and nothing else.\n"
    "Phrase: "
)


def extract_fuzzy_phrase(question_text: str) -> str | None:
    matches = list(BAND_RE.finditer(question_text))
    if not matches:
        return None
    return question_text[matches[0].start():matches[-1].end()]


def resolve_window_regex(phrase: str) -> tuple[int, int] | None:
    matches = BAND_RE.findall(phrase)
    if not matches:
        return None
    first_band, first_decade = matches[0]
    last_band, last_decade = matches[-1]
    start = int(first_decade) + BAND_BOUNDS[first_band.lower()][0]
    end = int(last_decade) + BAND_BOUNDS[last_band.lower()][1]
    if end < start:
        start, end = end, start
    return start, end


def _coerce_year(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1000 <= value <= 2030 else None
    if isinstance(value, str):
        match = YEAR_IN_TEXT_RE.search(value)
        if match:
            return int(match.group(1))
    return None


def parse_slm_window(response_text: str) -> tuple[int, int] | None:
    for candidate in JSON_OBJECT_RE.findall(response_text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        start = _coerce_year(payload.get("start"))
        end = _coerce_year(payload.get("end"))
        if start is not None and end is not None:
            return (start, end) if start <= end else (end, start)
    years = [int(match) for match in YEAR_IN_TEXT_RE.findall(response_text)]
    if len(years) >= 2:
        return min(years), max(years)
    return None


def resolve_window_slm(phrase: str, backend: str, base_url: str, model_name: str) -> tuple[tuple[int, int] | None, str]:
    response_text = _call_backend(PROMPT_HEADER + phrase, backend, base_url, model_name)
    return parse_slm_window(response_text), response_text


def _mrr5(rows: list[dict], windows: dict[str, tuple[int, int] | None], doc_span) -> dict:
    totals = {"mrr": 0.0, "hit_at_1": 0.0, "hit_at_5": 0.0}
    for row in rows:
        ids = [result["id"] for result in row["k_results"]]
        window = windows.get(row["query_id"]) if windows is not None else None
        if window is not None:
            ids = rerank_by_span(ids, window, doc_span.get)
        totals["mrr"] += score_at_k(ids, row["gold_doc_ids"], 5)["mrr"]
        totals["hit_at_1"] += score_at_k(ids, row["gold_doc_ids"], 1)["hit_at_k"]
        totals["hit_at_5"] += score_at_k(ids, row["gold_doc_ids"], 5)["hit_at_k"]
    n = len(rows)
    return {name: (value / n if n else 0.0) for name, value in totals.items()}


def run_validation(
    run_path: Path,
    corpus_path: Path,
    queries_path: Path,
    sample_size: int,
    seed: int,
    backend: str,
    base_url: str,
    model_name: str,
    max_seconds: float,
) -> dict:
    ctx = build_rerank_context(corpus_path, queries_path)
    gold_windows = ctx["query_window"]
    doc_span = ctx["doc_span"]

    all_rows = [row for row in load_rows(run_path) if row.get("gold_doc_ids") and row.get("k_results")]
    eligible = [
        row for row in all_rows
        if row["query_id"] in gold_windows and extract_fuzzy_phrase(row["test_query"]) is not None
    ]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    sample = eligible[:sample_size]

    regex_windows: dict[str, tuple[int, int] | None] = {}
    slm_windows: dict[str, tuple[int, int] | None] = {}
    slm_traces = []
    slm_latencies = []
    aborted_after = None

    for index, row in enumerate(sample):
        phrase = extract_fuzzy_phrase(row["test_query"])
        regex_windows[row["query_id"]] = resolve_window_regex(phrase)

        if aborted_after is not None:
            slm_windows[row["query_id"]] = None
            continue

        call_start = time.perf_counter()
        try:
            window, response_text = resolve_window_slm(phrase, backend, base_url, model_name)
        except RuntimeError as exc:
            slm_windows[row["query_id"]] = None
            slm_traces.append({"query_id": row["query_id"], "phrase": phrase, "error": str(exc)})
            aborted_after = index
            continue
        elapsed = time.perf_counter() - call_start
        slm_latencies.append(elapsed)
        slm_windows[row["query_id"]] = window
        slm_traces.append({
            "query_id": row["query_id"],
            "phrase": phrase,
            "gold": gold_windows.get(row["query_id"]),
            "regex": regex_windows[row["query_id"]],
            "slm": window,
            "response": response_text.strip()[:200],
        })

        if max_seconds and sum(slm_latencies) > max_seconds:
            aborted_after = index + 1

    resolved_both = [
        row for row in sample
        if regex_windows.get(row["query_id"]) is not None and slm_windows.get(row["query_id"]) is not None
    ]

    def arm(rows, windows):
        return _mrr5(rows, windows, doc_span)

    scored = {
        "baseline_fuzzy": arm(resolved_both, None),
        "regex_resolved": arm(resolved_both, regex_windows),
        "slm_resolved": arm(resolved_both, slm_windows),
        "gold_window": arm(resolved_both, gold_windows),
    }
    scored_full = {
        "baseline_fuzzy": arm(sample, None),
        "regex_resolved": arm(sample, regex_windows),
        "slm_resolved": arm(sample, slm_windows),
        "gold_window": arm(sample, gold_windows),
    }

    exact_regex = sum(
        1 for row in sample
        if regex_windows.get(row["query_id"]) == gold_windows.get(row["query_id"])
    )
    exact_slm = sum(
        1 for row in sample
        if slm_windows.get(row["query_id"]) is not None
        and slm_windows.get(row["query_id"]) == gold_windows.get(row["query_id"])
    )

    return {
        "run_path": str(run_path),
        "corpus_path": str(corpus_path),
        "queries_path": str(queries_path),
        "backend": backend,
        "base_url": base_url,
        "model_name": model_name,
        "seed": seed,
        "eligible_pool": len(eligible),
        "sample_size": len(sample),
        "slm_calls_made": len(slm_latencies),
        "slm_aborted_after": aborted_after,
        "slm_mean_latency_s": (sum(slm_latencies) / len(slm_latencies)) if slm_latencies else None,
        "regex_coverage": sum(1 for row in sample if regex_windows.get(row["query_id"]) is not None) / len(sample) if sample else 0.0,
        "slm_coverage": sum(1 for row in sample if slm_windows.get(row["query_id"]) is not None) / len(sample) if sample else 0.0,
        "regex_exact_gold_match": exact_regex / len(sample) if sample else 0.0,
        "slm_exact_gold_match": exact_slm / len(sample) if sample else 0.0,
        "resolved_both_n": len(resolved_both),
        "scored_resolved_both": scored,
        "scored_full_sample": scored_full,
        "slm_traces": slm_traces,
    }


def _decision(result: dict) -> str:
    scored = result["scored_resolved_both"]
    regex_mrr = scored["regex_resolved"]["mrr"]
    slm_mrr = scored["slm_resolved"]["mrr"]
    gold_mrr = scored["gold_window"]["mrr"]
    gap_before = gold_mrr - regex_mrr
    closed = slm_mrr - regex_mrr
    fraction = (closed / gap_before) if gap_before else 0.0
    verdict = (
        "BUILD the full SLM layer" if closed >= 0.02 and fraction >= 0.5
        else "DO NOT build the full SLM layer"
    )
    return (
        f"regex MRR@5 {regex_mrr:.4f} | slm MRR@5 {slm_mrr:.4f} | gold MRR@5 {gold_mrr:.4f}\n"
        f"gap left after regex: {gap_before:+.4f}   closed by SLM: {closed:+.4f} "
        f"({fraction:.0%} of the remaining gap)\n"
        f"decision: {verdict}"
    )


def print_report(result: dict) -> None:
    print(f"run:      {result['run_path']}")
    print(f"backend:  {result['backend']} / {result['model_name']} @ {result['base_url']}")
    print(f"sample:   {result['sample_size']} of {result['eligible_pool']} eligible (seed {result['seed']})")
    print(f"SLM:      {result['slm_calls_made']} call(s)"
          + (f", aborted after {result['slm_aborted_after']}" if result['slm_aborted_after'] is not None else "")
          + (f", mean {result['slm_mean_latency_s']:.2f}s/call" if result['slm_mean_latency_s'] else ""))
    print(f"coverage: regex {result['regex_coverage']:.1%} | slm {result['slm_coverage']:.1%}")
    print(f"exact gold-window match: regex {result['regex_exact_gold_match']:.1%} | slm {result['slm_exact_gold_match']:.1%}")
    print()
    for label, scored in (("resolved-by-both subset (n=%d)" % result["resolved_both_n"], result["scored_resolved_both"]),
                          ("full sample (n=%d, unresolved = no rerank)" % result["sample_size"], result["scored_full_sample"])):
        print(label)
        print(f"  {'arm':<18}{'MRR@5':>10}{'Hit@1':>10}{'Hit@5':>10}")
        for arm_name in ("baseline_fuzzy", "regex_resolved", "slm_resolved", "gold_window"):
            metrics = scored[arm_name]
            print(f"  {arm_name:<18}{metrics['mrr']:>10.4f}{metrics['hit_at_1']:>10.4f}{metrics['hit_at_5']:>10.4f}")
        print()
    print(_decision(result))


def write_outputs(result: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "normalisation_validation.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN, help="saved fuzzy-phrasing result .jsonl to score")
    parser.add_argument("--corpus", type=Path, default=CORPUS_OUTPUT_PATH)
    parser.add_argument("--queries", type=Path, default=QUERIES_OUTPUT_PATH)
    parser.add_argument("--sample", type=int, default=150)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--backend", default=LLM_BACKEND, choices=("openai", "ollama"))
    parser.add_argument("--base-url", default=None, help="default: consts.LLM_BASE_URL, or consts.OLLAMA_URL for --backend ollama")
    parser.add_argument("--model", default=LLM_MODEL_NAME)
    parser.add_argument("--max-seconds", type=float, default=900.0, help="SLM wall-clock budget; 0 disables the guard")
    parser.add_argument("--out", type=Path, default=None, help="directory for the json dump (default: a timestamped results dir)")
    args = parser.parse_args()

    base_url = args.base_url or (OLLAMA_URL if args.backend == "ollama" else LLM_BASE_URL)

    result = run_validation(
        run_path=args.run,
        corpus_path=args.corpus,
        queries_path=args.queries,
        sample_size=args.sample,
        seed=args.seed,
        backend=args.backend,
        base_url=base_url,
        model_name=args.model,
        max_seconds=args.max_seconds,
    )
    print_report(result)

    now = datetime.now()
    output_dir = args.out or (RESULTS_OUTPUT_DIR / now.strftime("%Y_%m_%d") / now.strftime("%H-%M-%S"))
    json_path = write_outputs(result, output_dir)
    print(f"\nWritten: {json_path}")


if __name__ == "__main__":
    main()
