from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from experiment.preprocessing.build_manifest import DEFAULT_BUILD_CONFIG
from experiment.preprocessing.build_queries import resolve_queries_output_path
from experiment.preprocessing.phrasing import resolve_phrasing_cache_path
from experiment.util.consts import (
    LLM_BACKEND,
    LLM_BASE_URL,
    LLM_BATCH_SIZE,
    LLM_MODEL_NAME,
    OLLAMA_URL,
)

"""
one off builder for the "language" phrasing token cache

converts each distinct time_start_raw/time_end_raw token in a built queries file to words
("May 1997" -> "May nineteen ninety-seven") and writes data/phrasing/phrasing_cache__<model>.json,
which is committed. the experiment never calls an llm, phrasing.py only reads this file

tokens go in numbered batches (LLM_BATCH_SIZE per request). a batch that doesn't line up is split
in half and retried, down to single tokens, otherwise a short or shuffled reply would pair tokens
with the wrong words

usage (from repo root, with a backend running):
    python src/experiment/preprocessing/build_phrasing_cache.py
    ... --backend ollama                      # old /api/generate
    ... --base-url http://127.0.0.1:11434/v1  # ollama's openai style endpoint
"""

__all__ = ["build_cache", "collect_tokens", "format_batch_prompt", "main", "parse_batch_response"]

PROMPT_HEADER = (
    "Convert each numbered date into words. Spell out every number, keep month names as words.\n"
    "Return exactly one numbered line per input, using the same numbers, and nothing else.\n"
    "Example input:\n"
    "1. Nov 1935\n"
    "2. 1937\n"
    "Example output:\n"
    "1. November nineteen thirty-five\n"
    "2. nineteen thirty-seven\n"
    "Now convert:\n"
)

NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\s*[.)]\s*(.+?)\s*$")


def collect_tokens(queries_path: Path) -> list[str]:
    tokens = set()
    with queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("gold_doc_ids"):
                continue
            tokens.add(row["time_start_raw"])
            tokens.add(row["time_end_raw"])
    return sorted(tokens)


def format_batch_prompt(tokens: list[str]) -> str:
    numbered = "\n".join(f"{position}. {token}" for position, token in enumerate(tokens, start=1))
    return PROMPT_HEADER + numbered


def _clean_conversion(text: str) -> str:
    text = text.strip().strip('"').strip("'").rstrip(".")
    for prefix in ("from ", "in ", "on "):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
    return text.strip()


def parse_batch_response(response_text: str, expected_count: int) -> list[str]:
    items = []
    for line in response_text.splitlines():
        match = NUMBERED_LINE_RE.match(line)
        if match:
            items.append((int(match.group(1)), match.group(2)))

    if len(items) != expected_count:
        raise ValueError(f"Batch response had {len(items)} numbered items, expected {expected_count}")

    for position, (number, _) in enumerate(items, start=1):
        if number != position:
            raise ValueError(f"Batch response index {number} out of order at position {position}")

    converted = [_clean_conversion(text) for _, text in items]
    for position, text in enumerate(converted, start=1):
        if not text:
            raise ValueError(f"Batch response item {position} was empty after cleaning")
    return converted


def _post_json(url: str, payload: dict, timeout: int = 300) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        raise RuntimeError(f"LLM backend timed out at {url}") from exc
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM backend HTTP error {exc.code} at {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to LLM backend at {url} ({exc.reason})") from exc


def _call_openai(prompt: str, base_url: str, model_name: str) -> str:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "stream": False,
    }
    result = _post_json(f"{base_url.rstrip('/')}/chat/completions", payload)
    return result["choices"][0]["message"]["content"]


def _call_ollama(prompt: str, base_url: str, model_name: str) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0},
    }
    return _post_json(base_url, payload)["response"]


def _call_backend(prompt: str, backend: str, base_url: str, model_name: str) -> str:
    if backend == "openai":
        return _call_openai(prompt, base_url, model_name)
    if backend == "ollama":
        return _call_ollama(prompt, base_url, model_name)
    raise ValueError(f"Unknown LLM backend {backend!r}, expected one of ('openai', 'ollama')")


def _convert_batch(batch: list[str], backend: str, base_url: str, model_name: str, call_counter: list[int]) -> list[str]:
    call_counter[0] += 1
    response_text = _call_backend(format_batch_prompt(batch), backend, base_url, model_name)
    try:
        return parse_batch_response(response_text, len(batch))
    except ValueError as exc:
        if len(batch) == 1:
            raise RuntimeError(
                f"Could not convert token {batch[0]!r} ({exc}). Last response:\n{response_text}"
            ) from exc
        print(f"  {exc} - splitting batch of {len(batch)}", flush=True)
        midpoint = len(batch) // 2
        return (
            _convert_batch(batch[:midpoint], backend, base_url, model_name, call_counter)
            + _convert_batch(batch[midpoint:], backend, base_url, model_name, call_counter)
        )


def build_cache(
    tokens: list[str],
    backend: str = LLM_BACKEND,
    base_url: str = LLM_BASE_URL,
    model_name: str = LLM_MODEL_NAME,
    batch_size: int = LLM_BATCH_SIZE,
) -> tuple[dict[str, str], int]:
    cache = {}
    call_counter = [0]
    batches = [tokens[start:start + batch_size] for start in range(0, len(tokens), batch_size)]

    for batch_number, batch in enumerate(batches, start=1):
        print(f"Batch {batch_number}/{len(batches)} ({len(batch)} tokens)", flush=True)
        cache.update(zip(batch, _convert_batch(batch, backend, base_url, model_name, call_counter)))

    return cache, call_counter[0]


def main(config_name: str = DEFAULT_BUILD_CONFIG, backend: str = LLM_BACKEND, base_url: str | None = None,
         model_name: str = LLM_MODEL_NAME, batch_size: int = LLM_BATCH_SIZE) -> Path:
    if base_url is None:
        base_url = OLLAMA_URL if backend == "ollama" else LLM_BASE_URL

    queries_path = resolve_queries_output_path(config_name)
    if not queries_path.exists():
        raise FileNotFoundError(
            f"No built queries file at {queries_path} - run 'python src/experiment/main.py -b {config_name}' first."
        )

    tokens = collect_tokens(queries_path)
    print(f"Converting {len(tokens)} distinct date tokens from {queries_path} via {backend} ({model_name})")

    cache, call_count = build_cache(tokens, backend, base_url, model_name, batch_size)

    output_path = resolve_phrasing_cache_path(model_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dict(sorted(cache.items())), f, indent=2, ensure_ascii=False)

    print(f"\nCoverage: {len(cache)}/{len(tokens)} tokens ({len(cache) / len(tokens):.1%}) in {call_count} model calls")
    print(f"Written to: {output_path}\n")
    print("Sample conversions:")
    sampled = sorted(cache)[::max(1, len(cache) // 20)][:20]
    for token in sampled:
        print(f"  {token!r} -> {cache[token]!r}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the committed token -> natural-language cache for the 'language' phrasing condition.")
    parser.add_argument("-b", "--build-config", default=DEFAULT_BUILD_CONFIG, help=f"build_manifest.json config whose queries file to read (default: {DEFAULT_BUILD_CONFIG!r}).")
    parser.add_argument("--backend", default=LLM_BACKEND, choices=("openai", "ollama"), help=f"LLM backend to call (default: {LLM_BACKEND!r}).")
    parser.add_argument("--base-url", default=None, help="override the backend endpoint (default: consts.LLM_BASE_URL, or consts.OLLAMA_URL for --backend ollama).")
    parser.add_argument("--model", default=LLM_MODEL_NAME, help=f"model name, also names the output cache file (default: {LLM_MODEL_NAME!r}).")
    parser.add_argument("--batch-size", type=int, default=LLM_BATCH_SIZE, help=f"tokens per request (default: {LLM_BATCH_SIZE}).")
    args = parser.parse_args()

    main(args.build_config, args.backend, args.base_url, args.model, args.batch_size)
