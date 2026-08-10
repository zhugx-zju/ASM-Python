"""Batch forward-problem mesh convergence study.

This module intentionally covers only the forward problem. It evaluates each
selected demo sample from its source distribution parameters on every mesh
and compares every mesh with the finest mesh. Inverse gamma selection is
deliberately outside this workflow.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import (
    FormatStrFormatter,
    LogFormatterMathtext,
    LogLocator,
    MaxNLocator,
    MultipleLocator,
)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from fgm_asm import MeshInfo
from fgm_asm.config_types import ForwardConfig
from fgm_asm.visualization import (
    plot_displacement_fields,
    plot_modulus_distribution,
    plot_single_displacement_field,
)
from grid_study.demo_data import DEFAULT_DEMO_ROOT, load_demo_dataset
from grid_study.distributions import validate_demo_modulus
from grid_study.case_runner import (
    interpolate_field as _interpolate_field,
    make_grf_reference,
    run_forward_case,
)
from grid_study.plot_style import (
    DATASET_STYLES,
    compact_tick as _compact_tick,
    style_axes as _style_axes,
)
from grid_study.study_io import write_csv_rows
from grid_study.study_metrics import relative_linf as _relative_linf


DEFAULT_DATASETS = ("bil", "exp", "grf")
DEFAULT_NODES = (4, 10, 20, 40, 80, 100, 200)

MESH_COLORS = {
    4: "#1f77b4",
    10: "#ff7f0e",
    20: "#2ca02c",
    40: "#d62728",
    80: "#9467bd",
    100: "#000000",
    200: "#7f7f7f",
}
MESH_MARKERS = {
    4: "o",
    10: "s",
    20: "^",
    40: "D",
    80: "P",
    100: "X",
    200: "v",
}
MESH_LINESTYLES = {
    4: "-",
    10: "--",
    20: "-.",
    40: ":",
    80: (0, (5, 1, 1, 1)),
    100: (0, (1, 1)),
    200: (0, (3, 1, 1, 1)),
}
MESH_LINEWIDTHS = {
    4: 1.8,
    10: 1.5,
    20: 1.5,
    40: 1.4,
    80: 1.35,
    100: 1.5,
    200: 1.4,
}
def _interpolate_right_boundary_profile(
    field: np.ndarray,
    mesh_info: MeshInfo,
    y_values: np.ndarray,
) -> np.ndarray:
    """Interpolate a nodal field on the loaded right boundary ``x = L``."""
    y_nodes = np.asarray(mesh_info.plot_y[:, 0], dtype=float)
    edge_values = np.asarray(field, dtype=float)[:, -1]
    return np.interp(y_values, y_nodes, edge_values)


def _plot_case_results(case: dict, output_dir: Path) -> None:
    """Save modulus and displacement contour figures for one mesh case."""
    figures = [
        plot_modulus_distribution(case["mesh"], case["E_field"], save_path=output_dir),
        plot_displacement_fields(case["mesh"], case["U"], save_path=output_dir),
        plot_single_displacement_field(case["mesh"], case["U"], component="ux", save_path=output_dir),
        plot_single_displacement_field(case["mesh"], case["U"], component="uy", save_path=output_dir),
    ]
    for figure in figures:
        plt.close(figure)


def _write_metrics(rows: list[dict], output_dir: Path) -> Path:
    path = output_dir / "forward_grid_metrics.csv"
    columns = [
        "dataset", "nodes", "elements", "reference_nodes",
        "relative_linf_ux_to_reference", "relative_linf_uy_to_reference",
        "maximum_relative_displacement_error", "max_abs_u_difference",
        "ux_norm", "uy_norm", "elapsed_time_seconds",
        "peak_python_memory_mb",
    ]
    return write_csv_rows(rows, path, columns)


def _plot_metrics(rows: list[dict], output_dir: Path) -> tuple[Path, Path]:
    fig, error_ax = plt.subplots(figsize=(7.2, 6.4))
    time_ax = error_ax.twinx()
    plot_nodes = sorted({int(row["nodes"]) for row in rows})
    reference_nodes = max(int(row["reference_nodes"]) for row in rows)
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = sorted((row for row in rows if row["dataset"] == dataset), key=lambda r: r["nodes"])
        style = DATASET_STYLES[dataset]
        accuracy_subset = [row for row in subset if int(row["nodes"]) < reference_nodes]
        error_ax.plot(
            [row["nodes"] for row in accuracy_subset],
            [float(row["maximum_relative_displacement_error"]) for row in accuracy_subset],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=1.8,
            markersize=4.5,
        )
        time_ax.plot(
            [row["nodes"] for row in subset],
            [float(row["elapsed_time_seconds"]) for row in subset],
            color=style["color"],
            marker=style["marker"],
            linestyle="--",
            linewidth=1.8,
            markersize=4.5,
        )

    error_ax.set_xlabel("Nodes per direction")
    error_ax.set_ylabel(
        f"Maximum relative displacement error to {reference_nodes} x {reference_nodes} mesh"
    )
    error_ax.set_xscale("log", base=2)
    error_ax.set_yscale("log")
    tick_nodes = [nodes for nodes in plot_nodes if nodes != reference_nodes]
    error_ax.set_xticks(tick_nodes)
    error_ax.set_xticklabels([str(nodes) for nodes in tick_nodes])
    time_ax.set_xticks(tick_nodes)
    time_ax.set_xticklabels([])
    error_ax.set_box_aspect(1.0)

    time_ax.set_ylabel("Forward solve time (s)")
    time_ax.set_yscale("log", base=10)
    time_ax.set_box_aspect(1.0)

    _style_axes(error_ax)
    _style_axes(time_ax)
    error_ax.yaxis.set_major_formatter(LogFormatterMathtext())
    time_ax.yaxis.set_major_formatter(LogFormatterMathtext())
    time_ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    time_ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100)
    )
    time_ax.tick_params(axis="x", bottom=False, labelbottom=False)
    time_ax.tick_params(
        axis="y",
        which="major",
        left=False,
        right=True,
        labelleft=False,
        labelright=True,
    )
    time_ax.tick_params(
        axis="y",
        which="minor",
        left=False,
        right=True,
        length=2.8,
        width=0.75,
    )
    time_ax.spines["bottom"].set_visible(False)
    time_ax.spines["left"].set_visible(False)

    error_handles = []
    time_handles = []
    for dataset in ("bil", "exp", "grf"):
        style = DATASET_STYLES[dataset]
        name = dataset.upper()
        error_handles.append(
            Line2D(
                [0], [0], color=style["color"], marker=style["marker"],
                linestyle="-", linewidth=1.8, markersize=4.5,
                label=rf"{name}, max. error (solid)",
            )
        )
        time_handles.append(
            Line2D(
                [0], [0], color=style["color"], marker=style["marker"],
                linestyle="--", linewidth=1.8, markersize=4.5,
                label=rf"{name}, time (dashed)",
            )
        )
    legend_handles = error_handles + time_handles
    error_ax.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=2,
        fontsize=8.0,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.4,
        handletextpad=0.5,
        labelspacing=0.55,
    )
    fig.subplots_adjust(left=0.16, right=0.84, bottom=0.14, top=0.96)
    png_path = output_dir / "forward_grid_convergence.png"
    pdf_path = output_dir / "forward_grid_convergence.pdf"
    fig.savefig(png_path, dpi=1200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _plot_forward_time(rows: list[dict], output_dir: Path) -> tuple[Path, Path]:
    """Save a standalone log-scale forward-solve time figure."""
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    plot_nodes = sorted({int(row["nodes"]) for row in rows})

    for dataset in sorted({row["dataset"] for row in rows}):
        subset = sorted(
            (row for row in rows if row["dataset"] == dataset),
            key=lambda row: int(row["nodes"]),
        )
        style = DATASET_STYLES[dataset]
        ax.plot(
            [int(row["nodes"]) for row in subset],
            [float(row["elapsed_time_seconds"]) for row in subset],
            color=style["color"],
            marker=style["marker"],
            linestyle="--",
            linewidth=1.8,
            markersize=4.5,
            label=dataset.upper(),
        )

    ax.set_xlabel("Nodes per direction")
    ax.set_ylabel("Forward solve time (s)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xticks(plot_nodes)
    ax.set_xticklabels([str(nodes) for nodes in plot_nodes])
    ax.set_box_aspect(1.0)
    _style_axes(ax)
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100)
    )
    ax.tick_params(axis="y", which="minor", length=2.8, width=0.75)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        fontsize=10.0,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.5,
        handletextpad=0.5,
    )

    png_path = output_dir / "forward_solve_time.png"
    pdf_path = output_dir / "forward_solve_time.pdf"
    fig.savefig(png_path, dpi=1200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _write_edge_profiles(profile_rows: list[dict], output_dir: Path) -> Path:
    path = output_dir / "forward_edge_profiles.csv"
    columns = ["dataset", "nodes", "y", "ux", "uy"]
    return write_csv_rows(profile_rows, path, columns)


def _plot_profile_panel(
    ax,
    cases: dict[int, dict],
    component: str,
    nodes_list: tuple[int, ...],
    y_values: np.ndarray,
    inset_anchor: tuple[float, float],
    panel_label: str,
    profile_pad_fraction: float = 0.06,
) -> tuple[list, list]:
    """Draw one displacement component and its local convergence inset."""
    displacement_scale = 1e3
    profiles = []
    handles = []
    labels = []

    for nodes in nodes_list:
        case = cases[nodes]
        profile = (
            _interpolate_right_boundary_profile(case[component], case["mesh"], y_values)
            * displacement_scale
        )
        profiles.append(profile)
        (line,) = ax.plot(
            y_values,
            profile,
            color=MESH_COLORS.get(nodes, "#444444"),
            marker=MESH_MARKERS.get(nodes, "o"),
            linestyle=MESH_LINESTYLES.get(nodes, "-"),
            markevery=max(1, len(y_values) // 8),
            linewidth=MESH_LINEWIDTHS.get(nodes, 1.5),
            markersize=4.0,
            markerfacecolor="white",
            markeredgewidth=0.8,
            solid_capstyle="round",
            label=f"{nodes} × {nodes} nodes",
        )
        handles.append(line)
        labels.append(f"{nodes} × {nodes} nodes")

    ax.set_xlabel(r"$y\; (\mathrm{mm})$", fontsize=14)
    ax.set_ylabel(rf"${component[0]}_{{{component[1:]}}}\; (\times 10^{{-3}}\,\mathrm{{mm}})$", fontsize=14)
    profile_matrix = np.asarray(profiles)
    x_margin = float(y_values[-1] - y_values[0]) * 0.02
    ax.set_xlim(float(y_values[0]) - x_margin, float(y_values[-1]) + x_margin)
    profile_range = float(np.ptp(profile_matrix))
    profile_pad = max(profile_range * profile_pad_fraction, np.finfo(float).eps)
    ax.set_ylim(float(np.min(profile_matrix)) - profile_pad, float(np.max(profile_matrix)) + profile_pad)
    ax.set_box_aspect(1.0)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    y_locator = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
    ax.yaxis.set_major_locator(y_locator)
    _style_axes(ax)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    y_ticks = y_locator.tick_values(
        float(np.min(profile_matrix)) - profile_pad,
        float(np.max(profile_matrix)) + profile_pad,
    )
    y_step = float(np.min(np.diff(y_ticks))) if len(y_ticks) > 1 else 1.0
    y_decimals = max(1, int(np.ceil(-np.log10(abs(y_step)))))
    ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{y_decimals}f"))
    ax.tick_params(axis="both", labelsize=10.5, width=0.9, length=4.5)
    ax.text(
        -0.12,
        1.02,
        f"({panel_label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        clip_on=False,
    )

    # Keep the highlighted data window clear of the panel edges.
    if component == "ux":
        x_low, x_high = 6.0, 7.0
    else:
        x_low, x_high = 1.5, 2.5
    focus_mask = (y_values >= x_low) & (y_values <= x_high)
    focus_values = profile_matrix[:, focus_mask]
    value_low = float(np.min(focus_values))
    value_high = float(np.max(focus_values))
    value_pad = max((value_high - value_low) * 0.12, np.finfo(float).eps)

    inset = inset_axes(
        ax,
        width="26%",
        height="26%",
        loc="lower left",
        bbox_to_anchor=(*inset_anchor, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    for nodes, profile in zip(nodes_list, profiles):
        inset.plot(
            y_values,
            profile,
            color=MESH_COLORS.get(nodes, "#444444"),
            linestyle=MESH_LINESTYLES.get(nodes, "-"),
            linewidth=MESH_LINEWIDTHS.get(nodes, 1.5) * 0.7,
        )
    inset.set_xlim(x_low, x_high)
    inset.set_ylim(value_low - value_pad, value_high + value_pad)
    inset.xaxis.set_major_locator(MultipleLocator(0.5))
    inset.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
    _style_axes(inset)
    inset.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    inset.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    inset.tick_params(
        axis="both",
        labelsize=8,
        width=0.7,
        length=3,
        pad=2,
    )
    for spine in inset.spines.values():
        spine.set_linewidth(1.1)
    zoom_y_low = value_low - value_pad
    zoom_y_high = value_high + value_pad
    zoom_box = Rectangle(
        (x_low, zoom_y_low),
        x_high - x_low,
        zoom_y_high - zoom_y_low,
        transform=ax.transData,
        fill=False,
        edgecolor="black",
        linewidth=0.9,
        zorder=1.5,
    )
    ax.add_patch(zoom_box)
    for inset_x, zoom_x in zip((0.0, 1.0), (x_low, x_high)):
        connector = ConnectionPatch(
            xyA=(inset_x, 1.0),
            coordsA=inset.transAxes,
            xyB=(zoom_x, zoom_y_low),
            coordsB=ax.transData,
            axesA=inset,
            axesB=ax,
            color="black",
            linewidth=0.7,
            zorder=1.4,
        )
        ax.figure.add_artist(connector)
    return handles, labels


def _plot_edge_dataset(
    cases: dict[int, dict],
    dataset: str,
    nodes_list: tuple[int, ...],
    output_dir: Path,
    y_values: np.ndarray,
) -> tuple[Path, Path]:
    """Write right-boundary ``u_x(y)``/``u_y(y)`` panels for one field."""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), sharex=True)
    panel_specs = (
        ("ux", (0.50, 0.12), "a"),
        ("uy", (0.22, 0.06 if dataset == "bil" else 0.12), "b"),
    )
    handles = labels = None
    for ax, (component, inset_anchor, panel_label) in zip(axes, panel_specs):
        panel_handles, panel_labels = _plot_profile_panel(
            ax,
            cases,
            component,
            nodes_list,
            y_values,
            inset_anchor,
            panel_label,
            profile_pad_fraction=(
                0.30
                if dataset == "bil" and component == "uy"
                else 0.14
                if dataset == "bil"
                else 0.06
            ),
        )
        if handles is None:
            handles, labels = panel_handles, panel_labels

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.76, 0.5),
        ncol=1,
        fontsize=11.5,
        frameon=False,
        handlelength=2.6,
        handletextpad=0.5,
        labelspacing=0.55,
    )
    fig.subplots_adjust(left=0.10, right=0.76, bottom=0.16, top=0.93, wspace=0.30)
    stem = f"forward_edge_{dataset}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=1200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _plot_edge_profiles(
    cases: dict[str, dict[int, dict]],
    datasets: tuple[str, ...],
    nodes_list: tuple[int, ...],
    output_dir: Path,
    y_values: np.ndarray,
) -> list[Path]:
    """Write one shared-legend two-panel figure per material field."""
    paths = []
    for dataset in datasets:
        normalized = str(dataset).lower()
        paths.extend(
            _plot_edge_dataset(
                cases[normalized],
                normalized,
                nodes_list,
                output_dir,
                y_values,
            )
        )
    return paths


def run_forward_grid_study(
    forward_config: ForwardConfig,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    nodes_list: tuple[int, ...] = DEFAULT_NODES,
    output_dir: Path | str = "results/grid_study/forward",
    demo_root: Path | str | None = DEFAULT_DEMO_ROOT,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_list = tuple(sorted({int(nodes) for nodes in nodes_list}))
    if not nodes_list or max(nodes_list) not in nodes_list:
        raise ValueError("nodes_list must contain at least one mesh")
    reference_nodes = max(nodes_list)

    raw_cases = {}
    manifest = {
        "study": "asm_forward_grid_convergence",
        "datasets": list(datasets),
        "nodes": list(nodes_list),
        "reference_nodes": max(nodes_list),
        "config": forward_config.to_dict(),
        "data_source": "parameterized_demo_distributions" if demo_root is not None else "analytical_repository_fields",
        "demo_validation": [],
        "cases": [],
    }

    for dataset in datasets:
        normalized = str(dataset).strip().lower()
        if normalized not in {"bil", "exp", "grf"}:
            raise ValueError(f"Forward grid study supports only bil/exp/grf, got {dataset!r}")
        raw_cases[normalized] = {}
        demo_data = load_demo_dataset(normalized, demo_root) if demo_root is not None else None
        if demo_data is not None:
            validation = validate_demo_modulus(demo_data, nu=forward_config.nu)
            if not validation["matches"]:
                raise ValueError(
                    f"Parameter-generated {normalized} demo does not reproduce its "
                    f"40x40 reference field: max difference={validation['max_absolute_difference']:.3e}"
                )
            manifest["demo_validation"].append(validation)
        grf_reference = None
        if normalized == "grf" and demo_data is None:
            grf_reference = make_grf_reference(forward_config, reference_nodes)
        for nodes in nodes_list:
            case = run_forward_case(
                normalized,
                nodes,
                forward_config=forward_config,
                grf_reference=grf_reference,
                demo_data=demo_data,
            )
            raw_cases[normalized][nodes] = case
            case_dir = output_dir / normalized / f"nodes_{nodes}"
            case_dir.mkdir(parents=True, exist_ok=True)
            with (case_dir / "forward_result.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        "dataset": normalized,
                        "nodes": nodes,
                        "elements": nodes - 1,
                        "E_field": case["E_field"],
                        "U": case["U"],
                        "ux": case["ux"],
                        "uy": case["uy"],
                        "elapsed_time_seconds": case["elapsed_time_seconds"],
                        "peak_python_memory_mb": case["peak_python_memory_mb"],
                        "config": case["config"],
                    },
                    handle,
                )
            (case_dir / "config.json").write_text(
                json.dumps(
                    {
                        "dataset": normalized,
                        "nodes": nodes,
                        "elements": nodes - 1,
                        "data_source": manifest["data_source"],
                        "config": case["config"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            _plot_case_results(case, case_dir)
            manifest["cases"].append({
                "dataset": normalized,
                "nodes": nodes,
                "elements": nodes - 1,
                "result_file": str((case_dir / "forward_result.pkl").relative_to(output_dir)),
            })

    metric_rows = []
    profile_rows = []
    y_values = np.linspace(0.0, float(forward_config.geo_h), 181)
    for dataset in datasets:
        normalized_dataset = str(dataset).lower()
        reference = raw_cases[normalized_dataset][reference_nodes]
        for nodes in nodes_list:
            case = raw_cases[normalized_dataset][nodes]
            ux_ref = _interpolate_field(reference["ux"], reference["mesh"], case["mesh"])
            uy_ref = _interpolate_field(reference["uy"], reference["mesh"], case["mesh"])
            u_ref = np.concatenate([ux_ref.ravel(order="C"), uy_ref.ravel(order="C")])
            u_case = np.concatenate([case["ux"].ravel(order="C"), case["uy"].ravel(order="C")])
            metric_rows.append({
                "dataset": str(dataset).lower(),
                "nodes": nodes,
                "elements": nodes - 1,
                "reference_nodes": reference_nodes,
                "relative_linf_ux_to_reference": _relative_linf(case["ux"], ux_ref),
                "relative_linf_uy_to_reference": _relative_linf(case["uy"], uy_ref),
                "maximum_relative_displacement_error": max(
                    _relative_linf(case["ux"], ux_ref),
                    _relative_linf(case["uy"], uy_ref),
                ),
                "max_abs_u_difference": float(np.max(np.abs(u_case - u_ref))),
                "ux_norm": float(np.linalg.norm(case["ux"])),
                "uy_norm": float(np.linalg.norm(case["uy"])),
                "elapsed_time_seconds": case["elapsed_time_seconds"],
                "peak_python_memory_mb": case["peak_python_memory_mb"],
            })

            ux_profile = _interpolate_right_boundary_profile(case["ux"], case["mesh"], y_values)
            uy_profile = _interpolate_right_boundary_profile(case["uy"], case["mesh"], y_values)
            profile_rows.extend(
                {
                    "dataset": normalized_dataset,
                    "nodes": nodes,
                    "y": float(y),
                    "ux": float(ux),
                    "uy": float(uy),
                }
                for y, ux, uy in zip(y_values, ux_profile, uy_profile)
            )

    metrics_path = _write_metrics(metric_rows, output_dir)
    figure_paths = (*_plot_metrics(metric_rows, output_dir), *_plot_forward_time(metric_rows, output_dir))
    edge_profile_csv = _write_edge_profiles(profile_rows, output_dir)
    edge_profile_figures = _plot_edge_profiles(
        raw_cases,
        tuple(str(dataset).lower() for dataset in datasets),
        nodes_list,
        output_dir,
        y_values,
    )
    manifest["metrics_file"] = str(metrics_path.relative_to(output_dir))
    manifest["figures"] = [str(path.relative_to(output_dir)) for path in figure_paths]
    manifest["edge_profile_file"] = str(edge_profile_csv.relative_to(output_dir))
    manifest["edge_profile_boundary"] = "right boundary x = geo_l"
    manifest["edge_profile_figures"] = [
        str(path.relative_to(output_dir)) for path in edge_profile_figures
    ]
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved forward grid metrics to {metrics_path}")
    return metrics_path
