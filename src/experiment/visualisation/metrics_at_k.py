
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from experiment.util.date_utils import YEAR_RE, plausible_year
from experiment.util.temporal import rerank_by_span
from experiment.visualisation.style import INK_PRIMARY, SURFACE, style_axis

DEFAULT_KS = (1, 3, 5, 10, 20, 30)


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows




def score_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> dict:
    top = ranked_ids[:k]
    gold = set(gold_ids)
    correct = sum(1 for doc_id in top if doc_id in gold)

    rank = 0
    for position, doc_id in enumerate(top, start=1):
        if doc_id in gold:
            rank = position
            break

    return {
        "hit_at_k": 1.0 if rank else 0.0,
        "precision_at_k": (correct / len(top)) if top else 0.0,
        "recall_at_k": (correct / len(gold)) if gold else 0.0,
        "mrr": (1.0 / rank) if rank else 0.0,
    }


def curve(rows: list[dict], ks, rerank_ctx=None) -> dict[int, dict]:
    out = {}
    for k in ks:
        totals = {"hit_at_k": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}
        n = 0
        for row in rows:
            if not row.get("gold_doc_ids"):
                continue
            ids = [result["id"] for result in row["k_results"]]
            if rerank_ctx is not None:
                ids = temporal_rerank(ids, row["query_id"], rerank_ctx)
            scores = score_at_k(ids, row["gold_doc_ids"], k)
            for name, value in scores.items():
                totals[name] += value
            n += 1
        out[k] = {name: (total / n if n else 0.0) for name, total in totals.items()}
        out[k]["n"] = n
    return out




def build_rerank_context(corpus_path: Path, queries_path: Path) -> dict:
    doc_span: dict[str, tuple[int, int] | None] = {}
    corpus_prefix = corpus_path.name
    for row in load_rows(corpus_path):
        years = [int(match.group(1)) for match in YEAR_RE.finditer(row["text"])]
        years = [year for year in years if plausible_year(year)]
        doc_span[f"{corpus_prefix}_{row['id']}"] = (min(years), max(years)) if years else None

    query_window: dict[str, tuple[int, int]] = {}
    for row in load_rows(queries_path):
        if not row.get("gold_doc_ids"):
            continue
        try:
            query_window[row["query_id"]] = (
                int(row["start_iso_value"][:4]),
                int(row["end_iso_value"][:4]),
            )
        except (TypeError, ValueError):
            continue

    return {"doc_span": doc_span, "query_window": query_window}


def temporal_rerank(ids: list[str], query_id: str, ctx: dict) -> list[str]:
    window = ctx["query_window"].get(query_id)
    return rerank_by_span(ids, window, ctx["doc_span"].get)




def build_k_sweep_figure(curves: dict[str, dict[int, dict]], ks, chart_title: str) -> plt.Figure:
    fig, (ax_hit, ax_mrr) = plt.subplots(1, 2, figsize=(max(9.0, len(curves) * 1.2 + 7), 5.0), facecolor=SURFACE)

    for label, data in curves.items():
        usable_ks = [k for k in ks if k in data]
        ax_hit.plot(usable_ks, [data[k]["hit_at_k"] for k in usable_ks], marker="o", label=label)
        ax_mrr.plot(usable_ks, [data[k]["mrr"] for k in usable_ks], marker="o", label=label)

    ax_hit.set_title("Hit@k", loc="left", color=INK_PRIMARY, fontsize=11, fontweight="bold")
    ax_hit.set_ylabel("Hit@k")
    ax_hit.set_ylim(0, 1.05)
    ax_mrr.set_title("MRR@k", loc="left", color=INK_PRIMARY, fontsize=11, fontweight="bold")
    ax_mrr.set_ylabel("MRR@k")

    for ax in (ax_hit, ax_mrr):
        ax.set_xlabel("k")
        ax.set_xticks(sorted(ks))
        style_axis(ax)

    if len(curves) > 1:
        ax_mrr.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)

    fig.suptitle(f"{chart_title} — k sweep", color=INK_PRIMARY, fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def print_table(name: str, data: dict[int, dict], ks) -> None:
    n = data[ks[0]]["n"]
    print(f"\n{name}  (n={n})")
    header = "  " + "metric".ljust(16) + "".join(f"k={k:<8}" for k in ks)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for metric, fmt in (
        ("hit_at_k", "{:.1%}"),
        ("recall_at_k", "{:.1%}"),
        ("precision_at_k", "{:.2%}"),
        ("mrr", "{:.3f}"),
    ):
        cells = "".join(fmt.format(data[k][metric]).ljust(10) for k in ks)
        print("  " + metric.ljust(16) + cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="a run directory or a single result .jsonl")
    parser.add_argument("--k", default=",".join(str(k) for k in DEFAULT_KS), help="comma-separated k values")
    parser.add_argument("--csv", type=Path, default=None, help="also write a tidy csv")
    parser.add_argument("--rerank", action="store_true", help="additionally score a temporal rerank of each run")
    parser.add_argument("--corpus", type=Path, default=Path("src/experiment/data/corpus/timeqa_corpus.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("src/experiment/data/queries/timeqa_queries.jsonl"))
    parser.add_argument("--subset-applied", action="store_true",
                         help="also score only rows with phrasing_applied=true, alongside the whole-arm table")
    args = parser.parse_args()

    ks = tuple(int(part) for part in args.k.split(","))

    if args.target.is_dir():
        files = sorted(p for p in args.target.glob("*.jsonl") if p.name != "_metadata.jsonl")
    else:
        files = [args.target]
    if not files:
        raise SystemExit(f"no result .jsonl found under {args.target}")

    ctx = None
    if args.rerank:
        print(f"building rerank context from {args.corpus} / {args.queries} ...")
        ctx = build_rerank_context(args.corpus, args.queries)

    csv_rows = []
    for path in files:
        rows = load_rows(path)
        max_k = max(len(r["k_results"]) for r in rows)
        usable = [k for k in ks if k <= max_k]
        skipped = [k for k in ks if k > max_k]
        if skipped:
            print(f"\n[{path.name}] run only stored {max_k} results/query - skipping k={skipped}")

        applied_rows = [row for row in rows if row.get("phrasing_applied")]

        variants = [("", None)] + ([(" + temporal rerank", ctx)] if ctx else [])
        for suffix, rerank_ctx in variants:
            populations = [(rows, "")]
            if args.subset_applied:
                populations.append((applied_rows, " [applied-subset]"))
            for scored_rows, pop_suffix in populations:
                label = path.stem + suffix + pop_suffix
                data = curve(scored_rows, usable, rerank_ctx)
                print_table(label, data, usable)
                for k in usable:
                    csv_rows.append({"run": label, "k": k, **{m: data[k][m] for m in
                                     ("hit_at_k", "precision_at_k", "recall_at_k", "mrr")}, "n": data[k]["n"]})

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nwrote {len(csv_rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
