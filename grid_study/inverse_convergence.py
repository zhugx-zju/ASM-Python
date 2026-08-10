"""Inverse-problem mesh and noise sensitivity study.

The main experiment keeps the regularization parameter fixed so that mesh
effects are measured independently from regularization-parameter selection.
Each case is saved separately and can be resumed after an interrupted run.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator

from fgm_asm.config_types import ForwardConfig, InverseConfig
from fgm_asm.results_io import save_inverse_results
from grid_study.case_runner import make_grf_reference, run_forward_case, run_inverse_case
from grid_study.demo_data import DEFAULT_DEMO_ROOT, load_demo_dataset
from grid_study.plot_style import style_axes as _style_axes
from grid_study.study_io import load_forward_case, save_forward_case, write_csv_rows
from grid_study.study_metrics import inverse_metrics
import config as cfg


DEFAULT_DATASETS = ("bil", "exp", "grf")
DEFAULT_NODES = (4, 10, 20, 40, 80, 100)
DEFAULT_NOISE_PERCENTAGE = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
GRF_REFERENCE_NODES = 200
NOISE_SEED = 42

NOISE_STYLES = {
    0.0: {"color": "#111111", "linestyle": "-", "marker": "o"},
    2.0: {"color": "#1f77b4", "linestyle": "--", "marker": "s"},
    4.0: {"color": "#ff7f0e", "linestyle": "-.", "marker": "^"},
    6.0: {"color": "#2ca02c", "linestyle": ":", "marker": "D"},
    8.0: {"color": "#d62728", "linestyle": (0, (5, 1, 1, 1)), "marker": "P"},
    10.0: {"color": "#9467bd", "linestyle": (0, (1, 1)), "marker": "X"},
}


def _noise_style(noise_percentage: float) -> dict:
    """Return a stable plotting style for a noise level."""
    key = float(noise_percentage)
    if key in NOISE_STYLES:
        return NOISE_STYLES[key]
    palette = ("#111111", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd")
    index = int(round(key)) % len(palette)
    return {"color": palette[index], "linestyle": "-", "marker": "o"}


def _write_metrics(rows: list[dict], output_dir: Path) -> Path:
    path = output_dir / "inverse_grid_metrics.csv"
    if not rows:
        raise ValueError("No inverse metrics were produced")
    return write_csv_rows(rows, path)


def _plot_metric(rows: list[dict], output_dir: Path, metric: str, ylabel: str, stem: str) -> tuple[Path, Path]:
    datasets = [dataset for dataset in DEFAULT_DATASETS if any(row["dataset"] == dataset for row in rows)]
    noise_levels = sorted({float(row["noise_percentage"]) for row in rows})
    nodes = sorted({int(row["nodes"]) for row in rows})
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 5.6), sharey=True)
    axes = np.atleast_1d(axes)
    handles = []
    labels = []
    for panel_index, dataset in enumerate(datasets):
        ax = axes[panel_index]
        for noise_percentage in noise_levels:
            subset = sorted(
                (
                    row for row in rows
                    if row["dataset"] == dataset and float(row["noise_percentage"]) == noise_percentage
                ),
                key=lambda row: int(row["nodes"]),
            )
            if not subset:
                continue
            style = _noise_style(noise_percentage)
            (line,) = ax.plot(
                [int(row["nodes"]) for row in subset],
                [float(row[metric]) for row in subset],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=1.5,
                markersize=4.0,
                markerfacecolor="white",
                markeredgewidth=0.8,
            )
            if panel_index == 0:
                handles.append(line)
                labels.append(f"{noise_percentage:g}% noise")
        ax.set_xlabel("Nodes per direction")
        ax.set_xscale("log", base=2)
        ax.set_xticks(nodes)
        ax.set_xticklabels([str(node) for node in nodes])
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
        ax.set_box_aspect(1.0)
        _style_axes(ax)
        ax.tick_params(axis="y", which="minor", length=2.8, width=0.75)
        ax.text(-0.12, 1.02, f"({chr(97 + panel_index)}) {dataset.upper()}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=14, clip_on=False)
    axes[0].set_ylabel(ylabel)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3,
               fontsize=9.0, frameon=False, handlelength=2.2, columnspacing=1.3,
               handletextpad=0.5)
    fig.subplots_adjust(top=0.82, bottom=0.15, left=0.08, right=0.98, wspace=0.25)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _plot_all(rows: list[dict], output_dir: Path) -> list[Path]:
    paths = []
    for metric, ylabel, stem in (
        ("relative_l2_E", r"Relative $L_2$ error of reconstructed modulus", "inverse_grid_accuracy"),
        ("inverse_elapsed_time_seconds", "Inverse solve time (s)", "inverse_grid_time"),
        ("peak_python_memory_mb", "Peak Python memory (MB)", "inverse_grid_memory"),
    ):
        paths.extend(_plot_metric(rows, output_dir, metric, ylabel, stem))
    return paths


def run_inverse_grid_study(
    forward_config: ForwardConfig,
    inverse_config: InverseConfig,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    nodes_list: tuple[int, ...] = DEFAULT_NODES,
    noise_percentage: tuple[float, ...] = DEFAULT_NOISE_PERCENTAGE,
    output_dir: Path | str = "results/grid_study/inverse",
    gamma: float | None = None,
    reference_nodes: int | None = None,
    max_iter: int | None = None,
    resume: bool = False,
    demo_root: Path | str | None = DEFAULT_DEMO_ROOT,
) -> Path:
    """Run fixed-gamma inverse cases over material fields, meshes, and noise."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_list = tuple(sorted({int(node) for node in nodes_list}))
    noise_percentage = tuple(sorted({float(level) for level in noise_percentage}))
    if not nodes_list or min(nodes_list) < 2:
        raise ValueError("nodes_list must contain values >= 2")
    if any(level < 0.0 for level in noise_percentage):
        raise ValueError("noise_percentage values must be non-negative")
    if gamma is None:
        raise ValueError(
            "gamma must be supplied from a separately selected reference-grid inverse case"
        )
    if reference_nodes is None:
        raise ValueError("reference_nodes must identify the grid used to select gamma")
    reference_nodes = int(reference_nodes)
    if reference_nodes not in nodes_list:
        raise ValueError("reference_nodes must be included in nodes_list")
    gamma = float(gamma)
    max_iter = int(inverse_config.max_iter if max_iter is None else max_iter)

    grf_reference = (
        make_grf_reference(forward_config, GRF_REFERENCE_NODES)
        if demo_root is None
        else None
    )
    rows = []
    manifest = {
        "study": "asm_inverse_grid_convergence_fixed_gamma",
        "datasets": list(datasets),
        "nodes": list(nodes_list),
        "reference_nodes": reference_nodes,
        "noise_percentage": list(noise_percentage),
        "gamma": gamma,
        "gamma_selection_method": "user_selected_reference_grid",
        "max_iter": max_iter,
        "noise_seed": NOISE_SEED,
        "forward_config": forward_config.to_dict(),
        "inverse_config": inverse_config.to_dict(),
        "data_source": "parameterized_demo_distributions" if demo_root is not None else "analytical_repository_fields",
        "cases": [],
    }

    for dataset in datasets:
        normalized = str(dataset).strip().lower()
        if normalized not in DEFAULT_DATASETS:
            raise ValueError(f"Unsupported inverse grid-study dataset: {dataset!r}")
        demo_data = load_demo_dataset(normalized, demo_root) if demo_root is not None else None
        for nodes in nodes_list:
            case_dir = output_dir / normalized / f"nodes_{nodes}"
            case_dir.mkdir(parents=True, exist_ok=True)
            forward_path = case_dir / "forward_case.pkl"
            if resume and forward_path.exists():
                case = load_forward_case(forward_path)
            else:
                case = run_forward_case(
                    normalized,
                    nodes,
                    forward_config=forward_config,
                    grf_reference=grf_reference if normalized == "grf" else None,
                    demo_data=demo_data,
                )
                save_forward_case(case, forward_path)

            (case_dir / "config.json").write_text(
                json.dumps(
                    {
                        "dataset": normalized,
                        "nodes": nodes,
                        "elements": nodes - 1,
                        "gamma": gamma,
                        "max_iter": max_iter,
                        "noise_percentage": list(noise_percentage),
                        "noise_seed": NOISE_SEED,
                        "data_source": manifest["data_source"],
                        "forward_config": case["config"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            for noise in noise_percentage:
                noise_dir = case_dir / f"noise_{noise:g}"
                noise_dir.mkdir(parents=True, exist_ok=True)
                (noise_dir / "config.json").write_text(
                    json.dumps(
                        {
                            "dataset": normalized,
                            "nodes": nodes,
                            "elements": nodes - 1,
                            "noise_percentage": noise,
                            "noise_level_fraction": noise / 100.0,
                            "noise_seed": NOISE_SEED,
                            "data_source": manifest["data_source"],
                            "gamma": gamma,
                            "reference_nodes": reference_nodes,
                            "forward_config": case["config"],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                inverse_path = noise_dir / "inverse_results.pkl"
                if resume and inverse_path.exists():
                    with inverse_path.open("rb") as handle:
                        saved = pickle.load(handle)
                    result = saved["results"]
                    U_measured = np.asarray(saved["U_measured"], dtype=float)
                    inverse_elapsed = float(saved["inverse_elapsed_time_seconds"])
                    peak_memory_mb = float(saved["peak_python_memory_mb"])
                else:
                    result, U_measured, inverse_elapsed, peak_memory_mb = run_inverse_case(
                        case,
                        noise,
                        gamma,
                        inverse_config,
                        max_iter=max_iter,
                    )
                    errors = inverse_metrics(
                        case, result, U_measured, noise, gamma, inverse_elapsed, peak_memory_mb
                    )
                    save_inverse_results(
                        result,
                        errors,
                        case["E_field"],
                        noise / 100.0,
                        noise_dir,
                        extra_data={
                            "dataset": normalized,
                            "nodes": nodes,
                            "noise_percentage": noise,
                            "U_clean": case["U"],
                            "U_measured": U_measured,
                            "inverse_elapsed_time_seconds": inverse_elapsed,
                            "peak_python_memory_mb": peak_memory_mb,
                            "forward_elapsed_time_seconds": case["elapsed_time_seconds"],
                            "forward_peak_python_memory_mb": case["peak_python_memory_mb"],
                            "gamma_selection_method": "fixed_gamma",
                        },
                    )
                row = inverse_metrics(
                    case, result, U_measured, noise, gamma, inverse_elapsed, peak_memory_mb
                )
                rows.append(row)
                manifest["cases"].append({
                    "dataset": normalized,
                    "nodes": nodes,
                    "noise_percentage": noise,
                    "result_dir": str(noise_dir.relative_to(output_dir)),
                })
                print(
                    f"Completed {normalized} {nodes}x{nodes}, noise={noise:g}%, "
                    f"iterations={row['iterations']}, converged={row['converged']}"
                )

    metrics_path = _write_metrics(rows, output_dir)
    figure_paths = _plot_all(rows, output_dir)
    manifest["metrics_file"] = metrics_path.name
    manifest["figures"] = [path.name for path in figure_paths]
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved inverse grid metrics to {metrics_path}")
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=DEFAULT_DATASETS)
    parser.add_argument("--nodes", nargs="+", type=int, default=list(DEFAULT_NODES))
    parser.add_argument(
        "--noise-percentage",
        "--noise-percent",
        dest="noise_percentage",
        nargs="+",
        type=float,
        default=list(DEFAULT_NOISE_PERCENTAGE),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/grid_study/inverse"))
    parser.add_argument(
        "--gamma",
        type=float,
        required=True,
        help="Fixed gamma selected beforehand on the reference grid",
    )
    parser.add_argument(
        "--reference-nodes",
        type=int,
        required=True,
        help="Reference grid used to select --gamma",
    )
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--demo-root",
        type=Path,
        default=DEFAULT_DEMO_ROOT,
        help="Read parameter metadata and 40x40 reference data from the repository demo package",
    )
    args = parser.parse_args()
    forward_config = cfg.get_forward_config()
    inverse_config = cfg.get_inverse_config()
    run_inverse_grid_study(
        forward_config=forward_config,
        inverse_config=inverse_config,
        datasets=tuple(args.datasets),
        nodes_list=tuple(args.nodes),
        noise_percentage=tuple(args.noise_percentage),
        output_dir=args.output_dir,
        gamma=args.gamma,
        reference_nodes=args.reference_nodes,
        max_iter=args.max_iter,
        resume=args.resume,
        demo_root=args.demo_root,
    )


if __name__ == "__main__":
    main()
