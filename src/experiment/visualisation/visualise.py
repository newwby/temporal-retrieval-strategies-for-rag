
from __future__ import annotations
from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
import re
import sys

from experiment.visualisation.csv_export import write_csv
from experiment.visualisation.grid_plot import build_grid_figure, grid_csv_rows, has_grid_data
from experiment.visualisation.latency_plot import build_latency_figure, has_latency_data, latency_csv_rows
from experiment.visualisation.metrics_at_k import DEFAULT_KS, build_k_sweep_figure, curve, load_rows
from experiment.eval.metadata import MANIFEST_PATH
from experiment.visualisation.style import INK_MUTED, INK_PRIMARY, INK_SECONDARY, SURFACE, style_axis

VISUALISATION_DIR = Path(__file__).resolve().parent
METADATA_INPUT_PATH = VISUALISATION_DIR / "_metadata.jsonl"
DEFAULT_TITLE = "Experiment run metrics"


METRIC_STYLE = {
    "hit_at_k": ("Hit@K", "#2a78d6", lambda v: f"{v:.1%}"),
    "precision_at_k": ("Precision@K", "#eb6834", lambda v: f"{v:.1%}"),
    "recall_at_k": ("Recall@K", "#1baf7a", lambda v: f"{v:.1%}"),
    "mrr": ("MRR", "#eda100", lambda v: f"{v:.3f}"),
    "answer_f1": ("Answer F1", "#8455c9", lambda v: f"{v:.1%}"),
    "answer_exact_match": ("Answer EM", "#d64a8b", lambda v: f"{v:.1%}"),
    "chrf": ("chrF1", "#c2683c", lambda v: f"{v:.1%}"),
    "answer_bertscore_p": ("BERTScore P", "#4c8c4a", lambda v: f"{v:.1%}"),
    "answer_bertscore_r": ("BERTScore R", "#6fae6d", lambda v: f"{v:.1%}"),
    "answer_bertscore_f1": ("BERTScore F1", "#357a38", lambda v: f"{v:.1%}"),
}


def _manifest_suite_names() -> list[str]:
    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            return sorted(json.load(f), key=len, reverse=True)
    except (OSError, ValueError):
        return []


"""
the manifest suite an output file belongs to
matched longest first rather than splitting on the first _k<n>, otherwise temporal_filtering_k30
would be cut to temporal_filtering and clash with its k=1/3/5 sibling
"""
def _suite_name_from_id(id_value: str) -> str:
    for name in _manifest_suite_names():
        if id_value == name or id_value.startswith(f"{name}_"):
            return name
    return re.split(r"_k\d+", id_value, maxsplit=1)[0]


def _get_title(id_value: str | None) -> str:
    if id_value is None or id_value == "":
        return DEFAULT_TITLE

    cleaned_text = _suite_name_from_id(id_value).replace("_", " ").strip().title()

    return cleaned_text or DEFAULT_TITLE


def _get_chart_title(rows: list[dict]) -> str:
    title = _get_title(rows[0].get("id") or rows[0].get("file_stem"))
    model = _shared_model(rows)
    return f"{title} ({model})" if model else title


def _suite_prefix(rows: list[dict]) -> str:
    stem = rows[0].get("id") or rows[0].get("file_stem") or ""
    return _suite_name_from_id(stem) if stem else ""


def _shared_model(rows: list[dict]) -> str | None:
    models = {row.get("model_name") for row in rows if row.get("model_name")}
    return next(iter(models)) if len(models) == 1 else None


"""
the token each output file uses for its model
read from the file stems not model_name, as suites with model_aliases use the alias (minilm)
in file names while the metadata has the full name (all-MiniLM-L6-v2)
"""
def _model_slugs(rows: list[dict], suite_prefix: str) -> list[str]:
    slugs = set()
    for row in rows:
        stem = row.get("id") or row.get("file_stem") or ""
        if suite_prefix and stem.startswith(f"{suite_prefix}_"):
            stem = stem[len(suite_prefix) + 1:]
        match = re.match(r"k\d+_([^_]+)", stem)
        if match:
            slugs.add(match.group(1))
    return sorted(slugs, key=len, reverse=True)


"""
drops the suite name and, if every run shares one model, the model slug
both are already in the chart title, leaves the part that changes, e.g. k3_temporalmode-none
"""
def _shorten_label(label: str, suite_prefix: str, model_slugs: list[str]) -> str:
    if suite_prefix and label.startswith(f"{suite_prefix}_"):
        label = label[len(suite_prefix) + 1:]
    for slug in model_slugs:
        label = label.replace(f"_{slug}_", "_").removesuffix(f"_{slug}")
    return label.strip("_") or "results"


def _slugify_title(title: str) -> str:
    return title.strip().lower().replace(" ", "_")


def _resolve_metric_keys(rows: list[dict]) -> list[str]:
    declared = set()
    for row in rows:
        declared.update(row.get("declared_metrics") or [])
    if declared:
        return [key for key in METRIC_STYLE if key in declared]
    return [key for key in METRIC_STYLE if any(row.get("metrics", {}).get(key) is not None for row in rows)]


def _build_figure(rows: list[dict], labels: list[str], chart_title: str) -> plt.Figure:
    metric_keys = _resolve_metric_keys(rows)

    if not metric_keys:
        raise ValueError(
            "None of the loaded rows have any recognised metrics "
            "(hit_at_k/precision_at_k/recall_at_k/mrr) to plot."
        )

    n_groups = len(labels)
    n_series = len(metric_keys)

    fig, ax = plt.subplots(figsize=(max(6.0, n_groups * 1.6), 5.5), facecolor=SURFACE)

    group_width = 0.8
    bar_width = group_width / n_series
    x = np.arange(n_groups)

    for series_index, metric_key in enumerate(metric_keys):
        display_label, colour, format_value = METRIC_STYLE[metric_key]
        values = [row.get("metrics", {}).get(metric_key) for row in rows]
        offsets = x - group_width / 2 + bar_width * series_index + bar_width / 2
        bars = ax.bar(
            offsets,
            [value if value is not None else 0 for value in values],
            width=bar_width * 0.9,
            color=colour,
            label=display_label,
            zorder=3,
        )

        # bars get narrow with many arms, stagger value labels by series so they don't overlap
        label_offset = 0.012 + (series_index % 2) * 0.05
        label_size = 7.5 if n_groups <= 8 else 6.5
        for bar, value in zip(bars, values):
            if value is None:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + label_offset,
                format_value(value),
                ha="center",
                va="bottom",
                fontsize=label_size,
                color=INK_SECONDARY,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", color=INK_PRIMARY, fontsize=9)
    ax.set_ylabel("Score", color=INK_PRIMARY)
    ax.set_ylim(0, 1.08)

    if n_series == 1:
        ax.set_title(f"{chart_title} — {METRIC_STYLE[metric_keys[0]][0]}", color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left")
    else:
        ax.set_title(chart_title, color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, labelcolor=INK_PRIMARY)

    style_axis(ax)

    if "mrr" in metric_keys and len(metric_keys) > 1:
        fig.text(0.01, 0.01, "Hit/Precision/Recall shown as % of queries. MRR is a mean reciprocal-rank score (0-1), not a percentage.",
                  fontsize=7.5, color=INK_MUTED, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
    else:
        fig.tight_layout()

    return fig


def _load_metadata_rows(path: Path = METADATA_INPUT_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"No _metadata.jsonl found at {path}.\n"
            "Pass the path to the run you want to visualise, e.g. "
            "data/results/<date>/<time>/_metadata.jsonl."
        )

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line {i} in {path}: {e}")

    if not rows:
        raise ValueError(f"{path} exists but contains no valid rows, nothing to visualise.")

    return rows


def _validate_label(row: dict, index: int) -> str:
    if row.get("id"):
        return row["id"]
    if row.get("file_stem"):
        return row["file_stem"]

    parts = [
        row.get("model_name"),
        f"k{row['k']}" if row.get("k") else None,
        row.get("retriever"),
        row.get("temporal_mode"),
        "entity_filter" if row.get("entity_filter") else None,
        row.get("corpus_encoding"),
        row.get("question_form"),
        row.get("phrasing"),
    ]
    label = "_".join(part for part in parts if part)
    return label or f"run_{index}"


def _metrics_csv_rows(rows: list[dict], labels: list[str], metric_keys: list[str]) -> list[dict]:
    return [
        {"run": label, **{key: row.get("metrics", {}).get(key) for key in metric_keys}}
        for label, row in zip(labels, rows)
    ]


def _apply_gold_filters(rows: list[dict], run_dir: Path) -> list[dict]:
    out_rows = []
    for row in rows:
        if row.get("gold_filter") != "multi_gold":
            out_rows.append(row)
            continue

        result_id = row.get("id") or row.get("file_stem")
        result_path = run_dir / f"{result_id}.jsonl" if result_id else None
        if not result_path or not result_path.exists():
            print(f"gold_filter=multi_gold set but no result file at {result_path} - showing unfiltered metrics.")
            out_rows.append(row)
            continue

        result_rows = load_rows(result_path)
        multi_gold_rows = [r for r in result_rows if len(r.get("gold_doc_ids") or []) > 1]
        if not multi_gold_rows:
            print(f"gold_filter=multi_gold matched 0 rows in {result_path} - showing unfiltered metrics.")
            out_rows.append(row)
            continue

        k = row.get("k") or max(len(r["k_results"]) for r in multi_gold_rows if r.get("k_results"))
        filtered = curve(multi_gold_rows, (k,))[k]

        new_row = dict(row)
        new_row["metrics"] = {**row.get("metrics", {}), **{key: val for key, val in filtered.items() if key != "n"}}
        new_row["_gold_filter_n"] = filtered["n"]
        out_rows.append(new_row)
    return out_rows


def _k_sweep_curves(rows: list[dict], labels: list[str], run_dir: Path) -> dict[str, dict[int, dict]]:
    curves = {}
    for label, metadata_row in zip(labels, rows):
        if metadata_row.get("gold_filter") == "multi_gold":
            continue
        result_id = metadata_row.get("id") or metadata_row.get("file_stem")
        if not result_id:
            continue
        result_path = run_dir / f"{result_id}.jsonl"
        if not result_path.exists():
            print(f"No result file at {result_path} - skipping k-sweep for {label!r}.")
            continue
        result_rows = load_rows(result_path)
        if not any(row.get("k_results") for row in result_rows):
            continue
        max_k = max(len(row["k_results"]) for row in result_rows)
        usable_ks = [k for k in DEFAULT_KS if k <= max_k]
        if usable_ks:
            curves[label] = curve(result_rows, usable_ks)
    return curves


def _k_sweep_csv_rows(curves: dict[str, dict[int, dict]]) -> list[dict]:
    return [
        {"run": label, "k": k, **{m: data[k][m] for m in ("hit_at_k", "precision_at_k", "recall_at_k", "mrr")}, "n": data[k]["n"]}
        for label, data in curves.items()
        for k in data
    ]


def _write_table(rows: list[dict], labels: list[str], metric_keys: list[str], chart_title: str, run_dir: Path, slug: str) -> tuple[Path, Path]:
    headers = ["run"] + [METRIC_STYLE[key][0] for key in metric_keys]
    table_rows = []
    for label, row in zip(labels, rows):
        metrics = row.get("metrics", {})
        cells = [label] + [
            METRIC_STYLE[key][2](metrics[key]) if metrics.get(key) is not None else "—"
            for key in metric_keys
        ]
        table_rows.append(cells)

    md_path = run_dir / f"{slug}_table.md"
    md_lines = [f"# {chart_title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    md_lines += ["| " + " | ".join(cells) + " |" for cells in table_rows]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    label_width = max(len(cells[0]) for cells in table_rows)
    fig_width = max(6.0, label_width * 0.11 + (len(headers) - 1) * 1.4)
    fig, ax = plt.subplots(figsize=(fig_width, 0.7 + 0.4 * len(table_rows)), facecolor=SURFACE)
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(headers))))
    table.scale(1, 1.4)
    ax.set_title(chart_title, color=INK_PRIMARY, fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()
    png_path = run_dir / f"{slug}_table.png"
    fig.savefig(png_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    return md_path, png_path


def main(input_path: Path = METADATA_INPUT_PATH, run_dir: Path | None = None) -> dict[str, Path]:
    rows = _load_metadata_rows(input_path)
    chart_title = _get_chart_title(rows)
    slug_title = _get_title(rows[0].get("id") or rows[0].get("file_stem"))
    labels = [_validate_label(row, i) for i, row in enumerate(rows)]
    suite_prefix = _suite_prefix(rows)
    model_slugs = _model_slugs(rows, suite_prefix) if _shared_model(rows) else []
    labels = [_shorten_label(label, suite_prefix, model_slugs) for label in labels]
    slug = _slugify_title(slug_title)
    run_dir = run_dir or input_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = _apply_gold_filters(rows, run_dir)
    labels = [
        f"{label} (multi-gold, n={row['_gold_filter_n']})" if row.get("_gold_filter_n") is not None else label
        for label, row in zip(labels, rows)
    ]

    written: dict[str, Path] = {}

    fig = _build_figure(rows, labels, chart_title)
    written["metrics_png"] = run_dir / f"{slug}.png"
    fig.savefig(written["metrics_png"], dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    metric_keys = _resolve_metric_keys(rows)
    written["metrics_csv"] = run_dir / f"{slug}.csv"
    write_csv(written["metrics_csv"], _metrics_csv_rows(rows, labels, metric_keys))

    grid_rows = [row for row in rows if row.get("gold_filter") != "multi_gold"]
    if has_grid_data(grid_rows):
        grid_fig = build_grid_figure(grid_rows, chart_title)
        written["grid_png"] = run_dir / f"{slug}_grid.png"
        grid_fig.savefig(written["grid_png"], dpi=150, facecolor=grid_fig.get_facecolor())
        plt.close(grid_fig)
        written["grid_csv"] = run_dir / f"{slug}_grid.csv"
        write_csv(written["grid_csv"], grid_csv_rows(grid_rows))

    if has_latency_data(rows):
        latency_fig = build_latency_figure(rows, labels, chart_title)
        written["latency_png"] = run_dir / f"{slug}_latency.png"
        latency_fig.savefig(written["latency_png"], dpi=150, facecolor=latency_fig.get_facecolor())
        plt.close(latency_fig)

        written["latency_csv"] = run_dir / f"{slug}_latency.csv"
        write_csv(written["latency_csv"], latency_csv_rows(rows, labels))
    else:
        print(f"No latency fields in {input_path} - skipping latency figure (older run?).")

    curves = _k_sweep_curves(rows, labels, run_dir)
    if curves:
        scored_ks = sorted({k for data in curves.values() for k in data})
        k_sweep_fig = build_k_sweep_figure(curves, scored_ks, chart_title)
        written["k_sweep_png"] = run_dir / f"{slug}_k_sweep.png"
        k_sweep_fig.savefig(written["k_sweep_png"], dpi=150, facecolor=k_sweep_fig.get_facecolor())
        plt.close(k_sweep_fig)

        written["k_sweep_csv"] = run_dir / f"{slug}_k_sweep.csv"
        write_csv(written["k_sweep_csv"], _k_sweep_csv_rows(curves))
    else:
        print(f"No k_results found for any run in {input_path} - skipping k-sweep figure.")

    written["table_md"], written["table_png"] = _write_table(rows, labels, metric_keys, chart_title, run_dir, slug)

    print(f"Wrote visualisation for {len(rows)} run(s) to {run_dir}")
    return written


if __name__ == "__main__":
    try:
        input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else METADATA_INPUT_PATH
        main(input_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
