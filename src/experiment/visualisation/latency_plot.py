from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from experiment.visualisation.style import INK_MUTED, INK_PRIMARY, INK_SECONDARY, SURFACE, style_axis

RETRIEVAL_COLOUR = "#2a78d6"
BUILD_COLOUR = "#eb6834"

LATENCY_CSV_FIELDS = (
    "retrieval_latency", "index_resolve_latency", "mutate_latency", "total_latency",
    "index_build_count", "index_build_total_seconds", "index_build_mean_seconds", "index_cache_hit_rate",
)


def has_latency_data(rows: list[dict]) -> bool:
    return any(row.get("metrics", {}).get("retrieval_latency") is not None for row in rows)


def latency_csv_rows(rows: list[dict], labels: list[str]) -> list[dict]:
    attributed = attributed_build_values(rows)
    return [
        {
            "run": label,
            **{field: row.get("metrics", {}).get(field) for field in LATENCY_CSV_FIELDS},
            "index_build_effective_seconds": value,
            "index_build_attributed": was_attributed,
        }
        for label, row, (value, was_attributed) in zip(labels, rows, attributed)
    ]


def has_index_build_data(rows: list[dict]) -> bool:
    return any(value for value, _ in attributed_build_values(rows))


"""
whether this arm queries a filtered index that has to be built
rerank uses the full index and only reorders, so it costs nothing, exclude needs a temporal sub index
"""
def _needs_built_index(row: dict) -> bool:
    return row.get("temporal_mode") == "exclude" or bool(row.get("entity_filter"))


def _filter_key(row: dict) -> tuple:
    return (row.get("temporal_mode"), bool(row.get("entity_filter")), row.get("retriever"),
            row.get("corpus_encoding"))


"""
index build cost per arm, charged to every arm that needs the index
a filtered index is built once per session and cached, so the first arm takes the whole cost
and the rest show zero. each arm that needs one is shown the cost for its filter setup,
flagged as attributed rather than measured
"""
def attributed_build_values(rows: list[dict]) -> list[tuple[float, bool]]:
    measured = {}
    for row in rows:
        value = row.get("metrics", {}).get("index_build_mean_seconds") or 0.0
        if value:
            measured.setdefault(_filter_key(row), value)

    out = []
    for row in rows:
        value = row.get("metrics", {}).get("index_build_mean_seconds") or 0.0
        if value:
            out.append((value, False))
        elif _needs_built_index(row) and _filter_key(row) in measured:
            out.append((measured[_filter_key(row)], True))
        else:
            out.append((0.0, False))
    return out


def build_latency_figure(rows: list[dict], labels: list[str], chart_title: str) -> plt.Figure:
    x = np.arange(len(labels))
    attributed = attributed_build_values(rows)
    build_values = [value for value, _ in attributed]
    show_build = any(build_values)

    width = max(8.0, len(labels) * 2.2) if show_build else max(5.0, len(labels) * 1.3)
    fig, axes = plt.subplots(1, 2 if show_build else 1, figsize=(width, 5.0), facecolor=SURFACE, squeeze=False)
    ax_retrieval = axes[0][0]

    retrieval_values = [row.get("metrics", {}).get("retrieval_latency") or 0.0 for row in rows]
    ax_retrieval.bar(x, retrieval_values, color=RETRIEVAL_COLOUR, zorder=3)
    ax_retrieval.set_ylabel("Seconds/query", color=INK_PRIMARY)
    ax_retrieval.set_title("Retrieval latency", loc="left", color=INK_PRIMARY, fontsize=11, fontweight="bold")
    drawn = [ax_retrieval]

    if show_build:
        ax_build = axes[0][1]
        bars = ax_build.bar(x, build_values, color=BUILD_COLOUR, zorder=3)
        for bar, (value, was_attributed) in zip(bars, attributed):
            if value == 0.0:
                continue
            if was_attributed:
                bar.set_alpha(0.45)
                bar.set_hatch("//")
                bar.set_edgecolor(BUILD_COLOUR)
            ax_build.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                          "inherited" if was_attributed else "built",
                          ha="center", va="bottom", fontsize=7, color=INK_SECONDARY)
        ax_build.set_ylabel("Seconds (per index build)", color=INK_PRIMARY)
        ax_build.set_title("Index build cost", loc="left", color=INK_PRIMARY, fontsize=11, fontweight="bold")
        drawn.append(ax_build)

    for ax in drawn:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8, color=INK_PRIMARY)
        style_axis(ax)

    if show_build:
        note = ("A filtered index is built once per session and cached, so hatched bars show the "
                "cost that arm inherited rather than paid - it would build the index itself if run "
                "first. Arms with no bar need no filtered index.")
    else:
        note = "No index build cost on any run - every arm queried the unfiltered index."

    fig.suptitle(f"{chart_title} - latency", color=INK_PRIMARY, fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.945, note, fontsize=8, color=INK_MUTED, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    return fig
