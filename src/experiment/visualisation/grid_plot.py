from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from experiment.visualisation.style import INK_MUTED, INK_PRIMARY, INK_SECONDARY, SURFACE

_CMAP = "RdBu"

GRID_CSV_FIELDS = ("corpus_encoding", "phrasing", "hit_at_k", "mrr", "corpus_encoding_coverage")


def has_grid_data(rows: list[dict]) -> bool:
    encodings = {row.get("corpus_encoding", "source") for row in rows}
    phrasings = {row.get("phrasing", "source") for row in rows}
    return len(encodings) > 1 and len(phrasings) > 1


def grid_csv_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "corpus_encoding": row.get("corpus_encoding", "source"),
            "phrasing": row.get("phrasing", "source"),
            "hit_at_k": row.get("metrics", {}).get("hit_at_k"),
            "mrr": row.get("metrics", {}).get("mrr"),
            "corpus_encoding_coverage": row.get("metrics", {}).get("corpus_encoding_coverage"),
            "invalid_reason": row.get("invalid_reason") or "",
        }
        for row in rows
    ]


def _axes_order(rows: list[dict]) -> tuple[list[str], list[str]]:
    encodings, phrasings = [], []
    for row in rows:
        encoding = row.get("corpus_encoding", "source")
        phrasing = row.get("phrasing", "source")
        if encoding not in encodings:
            encodings.append(encoding)
        if phrasing not in phrasings:
            phrasings.append(phrasing)
    return encodings, phrasings


def _matrix(rows: list[dict], encodings: list[str], phrasings: list[str], metric: str) -> np.ndarray:
    lookup = {
        (row.get("corpus_encoding", "source"), row.get("phrasing", "source")): row.get("metrics", {}).get(metric)
        for row in rows
    }
    matrix = np.full((len(encodings), len(phrasings)), np.nan)
    for i, encoding in enumerate(encodings):
        for j, phrasing in enumerate(phrasings):
            value = lookup.get((encoding, phrasing))
            if value is not None:
                matrix[i, j] = value
    return matrix


def _draw_panel(ax, matrix, encodings, phrasings, title, fmt, baseline) -> None:
    delta = matrix - baseline
    limit = np.nanmax(np.abs(delta)) or 1.0
    ax.imshow(delta, cmap=_CMAP, vmin=-limit, vmax=limit, aspect="auto")

    ax.set_xticks(range(len(phrasings)))
    ax.set_xticklabels(phrasings, rotation=20, ha="right", color=INK_PRIMARY, fontsize=9)
    ax.set_yticks(range(len(encodings)))
    ax.set_yticklabels(encodings, color=INK_PRIMARY, fontsize=9)
    ax.set_xlabel("query phrasing", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("corpus encoding", color=INK_SECONDARY, fontsize=9)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, fontweight="bold", loc="left")

    for i in range(len(encodings)):
        for j in range(len(phrasings)):
            if np.isnan(matrix[i, j]):
                continue
            on_diagonal = encodings[i] == phrasings[j]
            ax.text(
                j, i, fmt(matrix[i, j]),
                ha="center", va="center", fontsize=8.5, color=INK_PRIMARY,
                fontweight="bold" if on_diagonal else "normal",
            )
            if on_diagonal:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=INK_PRIMARY, linewidth=1.6))

    ax.set_xticks(np.arange(-0.5, len(phrasings), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(encodings), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(colors=INK_MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_grid_figure(rows: list[dict], chart_title: str) -> plt.Figure:
    encodings, phrasings = _axes_order(rows)
    hit = _matrix(rows, encodings, phrasings, "hit_at_k")
    mrr = _matrix(rows, encodings, phrasings, "mrr")

    fig, axes = plt.subplots(1, 2, figsize=(6.2 + len(phrasings) * 1.1, 1.6 + len(encodings) * 0.95),
                             facecolor=SURFACE)

    _draw_panel(axes[0], hit, encodings, phrasings, "Hit@K", lambda v: f"{v:.1%}", hit[0, 0])
    _draw_panel(axes[1], mrr, encodings, phrasings, "MRR", lambda v: f"{v:.3f}", mrr[0, 0])

    fig.suptitle(chart_title, color=INK_PRIMARY, fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.text(
        0.01, 0.01,
        "Colour is the difference from the (source, source) control, not the absolute value. "
        "Boxed diagonal = corpus encoded the same way the query is phrased.",
        fontsize=7.5, color=INK_MUTED, ha="left", va="bottom",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    return fig
