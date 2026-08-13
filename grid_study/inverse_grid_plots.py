"""Publication-style plots for completed inverse mesh studies."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator

from grid_study.plot_style import style_axes


DEFAULT_PLOT_NODES = (10, 20, 40, 80, 100, 200)
NOISE_STYLES = {
    0.0: {"color": "#1f77b4", "marker": "o"},
    2.0: {"color": "#ff7f0e", "marker": "s"},
    4.0: {"color": "#2ca02c", "marker": "^"},
    6.0: {"color": "#d62728", "marker": "D"},
    8.0: {"color": "#9467bd", "marker": "v"},
    10.0: {"color": "#8c564b", "marker": "P"},
}
FALLBACK_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h", "<", ">")


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _noise_style(noise: float, index: int, count: int) -> dict[str, object]:
    """Return a stable style for standard or arbitrary noise levels."""
    if noise in NOISE_STYLES:
        return NOISE_STYLES[noise]
    return {
        "color": plt.get_cmap("tab10")((index % 10) / 9.0),
        "marker": FALLBACK_MARKERS[index % len(FALLBACK_MARKERS)],
    }


def _legend_layout(handles: list[Line2D], max_columns: int = 3) -> tuple[list[Line2D], int]:
    """Arrange a variable number of handles in visually row-major order."""
    if not handles:
        return [], 1
    ncol = min(max_columns, len(handles))
    reordered = []
    for column in range(ncol):
        for index in range(column, len(handles), ncol):
            reordered.append(handles[index])
    return reordered, ncol


def load_inverse_metrics(metrics_path: Path | str, dataset: str = "bil") -> list[dict]:
    """Load a legacy or current inverse metrics CSV into typed rows."""
    metrics_path = Path(metrics_path)
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = []
    for raw in raw_rows:
        if str(raw.get("dataset", dataset)).strip().lower() != dataset.lower():
            continue
        nodes_value = raw.get("nodes", raw.get("nodesx"))
        if nodes_value in {None, ""}:
            raise KeyError("Inverse metrics must contain either 'nodes' or 'nodesx'")
        rows.append(
            {
                **raw,
                "dataset": dataset.lower(),
                "nodes": int(float(nodes_value)),
                "noise_percentage": float(raw["noise_percentage"]),
                "relative_l2_E": float(raw["relative_l2_E"]),
                "inverse_elapsed_time_seconds": float(raw["inverse_elapsed_time_seconds"]),
                "peak_python_memory_mb": float(raw["peak_python_memory_mb"]),
                "iterations": int(float(raw["iterations"])),
                "converged": _as_bool(raw["converged"]),
            }
        )
    if not rows:
        raise ValueError(f"No {dataset.upper()} inverse metrics were found in {metrics_path}")
    return rows


def _prepare_axis(ax, nodes: tuple[int, ...], ylabel: str) -> None:
    ax.set_xlabel("Nodes per direction")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xticks(nodes)
    ax.set_xticklabels([str(node) for node in nodes])
    ax.set_box_aspect(1.0)
    style_axes(ax)
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100)
    )
    ax.tick_params(axis="y", which="minor", length=2.8, width=0.75)


def _plot_noise_curves(
    ax,
    rows: list[dict],
    nodes: tuple[int, ...],
    metric: str,
) -> tuple[list[Line2D], bool]:
    handles = []
    has_nonconverged = False
    available_noise = sorted({float(row["noise_percentage"]) for row in rows})
    for noise_index, noise in enumerate(available_noise):
        subset = sorted(
            (
                row for row in rows
                if float(row["noise_percentage"]) == noise and int(row["nodes"]) in nodes
            ),
            key=lambda row: int(row["nodes"]),
        )
        if not subset:
            continue
        style = _noise_style(noise, noise_index, len(available_noise))
        x_values = [int(row["nodes"]) for row in subset]
        y_values = [float(row[metric]) for row in subset]
        ax.plot(
            x_values,
            y_values,
            color=style["color"],
            marker=style["marker"],
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            linestyle="-",
            linewidth=1.8,
            markersize=5.0,
        )
        failed = [row for row in subset if not bool(row["converged"])]
        if failed:
            has_nonconverged = True
            ax.plot(
                [int(row["nodes"]) for row in failed],
                [float(row[metric]) for row in failed],
                linestyle="none",
                marker=style["marker"],
                markersize=7.0,
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=1.4,
                zorder=5,
            )
        handles.append(
            Line2D(
                [0], [0], color=style["color"], marker=style["marker"],
                linestyle="-", linewidth=1.8, markersize=5.0,
                label=f"{noise:g}% noise",
            )
        )
    return handles, has_nonconverged


def _save_combined_figure(
    rows: list[dict],
    output_dir: Path,
    nodes: tuple[int, ...],
    stem: str,
) -> tuple[Path, Path]:
    """Save error, solve-time, and memory results as one three-panel figure."""
    panels = (
        ("relative_l2_E", "Relative $L_2$ error of reconstructed modulus", "(a)"),
        ("inverse_elapsed_time_seconds", "Final inverse solve time (s)", "(b)"),
        ("peak_python_memory_mb", "Peak Python memory (MB)", "(c)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.4))
    legend_handles = []
    for ax, (metric, ylabel, panel_label) in zip(axes, panels):
        handles, _ = _plot_noise_curves(ax, rows, nodes, metric)
        if not legend_handles:
            legend_handles = handles
        _prepare_axis(ax, nodes, ylabel)
        ax.text(
            -0.12,
            1.02,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=14,
            clip_on=False,
        )

    legend_handles, legend_columns = _legend_layout(legend_handles)
    axes[0].legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=legend_columns,
        fontsize=7.5,
        frameon=False,
        handlelength=1.8,
        columnspacing=0.85,
        handletextpad=0.35,
        labelspacing=0.35,
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.16, top=0.93, wspace=0.22)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=1200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _save_metric_figure(
    rows: list[dict],
    output_dir: Path,
    nodes: tuple[int, ...],
    metric: str,
    ylabel: str,
    stem: str,
) -> tuple[Path, Path]:
    """Save one metric figure for callers that need a standalone panel."""
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    handles, has_nonconverged = _plot_noise_curves(ax, rows, nodes, metric)
    _prepare_axis(ax, nodes, ylabel)
    if has_nonconverged:
        handles.append(
            Line2D(
                [0], [0], color="#555555", marker="o", linestyle="none",
                markerfacecolor="white", markeredgewidth=1.4,
                label="Iteration limit reached",
            )
        )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        fontsize=8.8,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.4,
        handletextpad=0.5,
        labelspacing=0.55,
    )
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.14, top=0.96)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=1200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_inverse_grid_study(
    metrics_path: Path | str,
    output_dir: Path | str | None = None,
    dataset: str = "bil",
    nodes: tuple[int, ...] = DEFAULT_PLOT_NODES,
) -> list[Path]:
    """Create three standalone figures plus one combined three-panel figure."""
    metrics_path = Path(metrics_path)
    output_dir = metrics_path.parent if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_nodes = tuple(int(node) for node in nodes)
    rows = load_inverse_metrics(metrics_path, dataset=dataset)
    present_nodes = {int(row["nodes"]) for row in rows}
    missing_nodes = [node for node in selected_nodes if node not in present_nodes]
    if missing_nodes:
        raise ValueError(f"Metrics are missing requested node grids: {missing_nodes}")

    prefix = dataset.lower()
    paths = []
    for metric, ylabel, stem in (
        (
            "relative_l2_E",
            "Relative $L_2$ error of reconstructed modulus",
            f"inverse_grid_convergence_{prefix}",
        ),
        (
            "inverse_elapsed_time_seconds",
            "Final inverse solve time (s)",
            f"inverse_solve_time_{prefix}",
        ),
        (
            "peak_python_memory_mb",
            "Peak Python memory (MB)",
            f"inverse_peak_memory_{prefix}",
        ),
    ):
        paths.extend(
            _save_metric_figure(rows, output_dir, selected_nodes, metric, ylabel, stem)
        )

    combined_stem = f"inverse_grid_convergence_combined_{prefix}"
    paths.extend(
        _save_combined_figure(rows, output_dir, selected_nodes, combined_stem)
    )
    return paths


__all__ = ["DEFAULT_PLOT_NODES", "load_inverse_metrics", "plot_inverse_grid_study"]
