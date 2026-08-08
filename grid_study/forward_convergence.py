"""Batch forward-problem mesh convergence study.

This module intentionally covers only the forward problem. It regenerates the
same analytical BIL/EXP modulus field on each mesh, solves the forward FEM
problem, and compares every mesh with the finest mesh after interpolation.
Inverse gamma selection is deliberately outside this workflow.
"""

from __future__ import annotations

import csv
import json
import pickle
import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, MaxNLocator, MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import RegularGridInterpolator

from fgm_asm import MeshInfo, fem_assemble, forward_solver, generate_fgm_modulus
from fgm_asm.mesh import setup_boundary_conditions
import config as cfg


DEFAULT_DATASETS = ("bil", "exp")
DEFAULT_NODES = (4, 10, 20, 40, 80, 100)

MESH_COLORS = {
    4: "#1f77b4",
    10: "#ff7f0e",
    20: "#2ca02c",
    40: "#d62728",
    80: "#9467bd",
    100: "#000000",
}
MESH_MARKERS = {
    4: "o",
    10: "s",
    20: "^",
    40: "D",
    80: "P",
    100: "X",
}
MESH_LINESTYLES = {
    4: "-",
    10: "--",
    20: "-.",
    40: ":",
    80: (0, (5, 1, 1, 1)),
    100: (0, (1, 1)),
}
MESH_LINEWIDTHS = {
    4: 1.8,
    10: 1.5,
    20: 1.5,
    40: 1.4,
    80: 1.35,
    100: 1.5,
}
DATASET_STYLES = {
    "bil": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "exp": {"color": "#d62728", "marker": "s", "linestyle": "--"},
}

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "Times New Roman"
rcParams["mathtext.it"] = "Times New Roman:italic"
rcParams["axes.linewidth"] = 1.0


def _compact_tick(value: float, _position: int) -> str:
    """Format ticks without trailing zeros or unnecessary decimal places."""
    if abs(value) < 1e-12:
        return "0"
    return f"{value:g}"


def _style_axes(ax) -> None:
    ax.grid(False)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=False,
        right=False,
        width=0.9,
        length=4.5,
    )
    ax.xaxis.set_major_formatter(FuncFormatter(_compact_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(_compact_tick))


def _relative_l1(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = float(np.sum(np.abs(b)))
    return float(np.sum(np.abs(a - b)) / (denominator + 1e-15))


def _field_from_displacement(mesh_info: MeshInfo, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert interleaved nodal displacement to [y, x] fields."""
    ux = np.asarray(U[0::2], dtype=float)
    uy = np.asarray(U[1::2], dtype=float)
    return (
        ux.reshape(mesh_info.nods_y, mesh_info.nods_x, order="C"),
        uy.reshape(mesh_info.nods_y, mesh_info.nods_x, order="C"),
    )


def _interpolate_field(field: np.ndarray, source_mesh: MeshInfo, target_mesh: MeshInfo) -> np.ndarray:
    x_source = np.asarray(source_mesh.plot_x[0, :], dtype=float)
    y_source = np.asarray(source_mesh.plot_y[:, 0], dtype=float)
    interpolator = RegularGridInterpolator(
        (y_source, x_source),
        np.asarray(field, dtype=float),
        bounds_error=True,
    )
    points = np.column_stack((target_mesh.Y, target_mesh.X))
    return interpolator(points).reshape(target_mesh.nods_y, target_mesh.nods_x)


def _interpolate_edge_profile(field: np.ndarray, mesh_info: MeshInfo, y_values: np.ndarray) -> np.ndarray:
    """Interpolate a nodal field on the loaded right edge x=L."""
    y_nodes = np.asarray(mesh_info.plot_y[:, 0], dtype=float)
    edge_values = np.asarray(field, dtype=float)[:, -1]
    return np.interp(y_values, y_nodes, edge_values)


def _measure_case(func):
    """Measure wall time and Python allocation peak for one forward case."""
    tracemalloc.start()
    start = time.perf_counter()
    try:
        value = func()
    finally:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return value, float(time.perf_counter() - start), float(peak_bytes / 1024**2)


def run_forward_case(dis_type: str, nodes: int) -> dict:
    if nodes < 2:
        raise ValueError(f"nodes must be at least 2, got {nodes}")

    forward_config = cfg.get_forward_config()
    mesh = MeshInfo(
        forward_config.geo_l,
        forward_config.geo_h,
        int(nodes) - 1,
        int(nodes) - 1,
    )

    def solve():
        E_field, material_info = generate_fgm_modulus(
            mesh,
            dis_type=dis_type,
            Ex=forward_config.Ex,
            Ey=forward_config.Ey,
        )
        material_info.nu = forward_config.nu
        material_info.update(E_field.ravel(order="C"), iteration=1)
        bc_info = setup_boundary_conditions(
            mesh,
            forward_config.geo_l,
            forward_config.geo_h,
            forward_config.f_tot,
        )
        fem_info = fem_assemble(mesh, material_info, bc_info)
        U = forward_solver(fem_info)
        return mesh, bc_info, E_field, U

    (mesh, bc_info, E_field, U), elapsed, peak_python_mb = _measure_case(solve)
    ux, uy = _field_from_displacement(mesh, U)
    case_config = forward_config.to_dict()
    case_config["nel_x"] = int(nodes - 1)
    case_config["nel_y"] = int(nodes - 1)
    return {
        "dataset": dis_type,
        "nodes": int(nodes),
        "elements": int(nodes - 1),
        "mesh": mesh,
        "bc_info": bc_info,
        "E_field": np.asarray(E_field, dtype=float),
        "U": np.asarray(U, dtype=float),
        "ux": ux,
        "uy": uy,
        "elapsed_time_seconds": elapsed,
        "peak_python_memory_mb": peak_python_mb,
        "config": case_config,
    }


def _write_metrics(rows: list[dict], output_dir: Path) -> Path:
    path = output_dir / "forward_grid_metrics.csv"
    columns = [
        "dataset", "nodes", "elements", "reference_nodes",
        "relative_l1_ux_to_reference", "relative_l1_uy_to_reference",
        "relative_l2_u_to_reference", "max_abs_u_difference",
        "ux_norm", "uy_norm", "elapsed_time_seconds",
        "peak_python_memory_mb",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in columns} for row in rows)
    return path


def _plot_metrics(rows: list[dict], output_dir: Path) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    plot_nodes = sorted({int(row["nodes"]) for row in rows})
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = sorted((row for row in rows if row["dataset"] == dataset), key=lambda r: r["nodes"])
        nodes = [row["nodes"] for row in subset]
        style = DATASET_STYLES[dataset]
        error_values = [
            row["relative_l2_u_to_reference"] if row["relative_l2_u_to_reference"] > 0 else np.nan
            for row in subset
        ]
        axes[0].plot(
            nodes,
            error_values,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=4.5,
            label=dataset.upper(),
        )
        axes[1].plot(
            nodes,
            [row["elapsed_time_seconds"] for row in subset],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=4.5,
            label=dataset.upper(),
        )

    axes[0].set_xlabel("Nodes per direction")
    axes[0].set_ylabel("Relative L2 difference to 100 x 100 mesh")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xticks(plot_nodes)
    axes[0].set_xticklabels([str(nodes) for nodes in plot_nodes])

    axes[1].set_xlabel("Nodes per direction")
    axes[1].set_ylabel("Forward solve time (s)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(plot_nodes)
    axes[1].set_xticklabels([str(nodes) for nodes in plot_nodes])
    axes[1].set_yscale("log")
    for ax in axes:
        _style_axes(ax)
    axes[0].yaxis.set_major_formatter(LogFormatterMathtext())
    axes[1].yaxis.set_major_formatter(LogFormatterMathtext())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        fontsize=10,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), pad=1.2)
    png_path = output_dir / "forward_grid_convergence.png"
    pdf_path = output_dir / "forward_grid_convergence.pdf"
    fig.savefig(png_path, dpi=1200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _write_edge_profiles(profile_rows: list[dict], output_dir: Path) -> Path:
    path = output_dir / "forward_edge_profiles.csv"
    columns = ["dataset", "nodes", "y", "ux", "uy"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in profile_rows)
    return path


def _plot_profile_panel(
    ax,
    cases: dict[int, dict],
    component: str,
    nodes_list: tuple[int, ...],
    y_values: np.ndarray,
    inset_anchor: tuple[float, float],
    panel_label: str,
) -> tuple[list, list]:
    """Draw one displacement component and its local convergence inset."""
    displacement_scale = 1e3
    profiles = []
    handles = []
    labels = []

    for nodes in nodes_list:
        case = cases[nodes]
        profile = (
            _interpolate_edge_profile(case[component], case["mesh"], y_values)
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
    profile_pad = max(profile_range * 0.06, np.finfo(float).eps)
    ax.set_ylim(float(np.min(profile_matrix)) - profile_pad, float(np.max(profile_matrix)) + profile_pad)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    _style_axes(ax)
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

    # Select a compact inset around the largest difference between meshes.
    spread = np.ptp(profile_matrix, axis=0)
    focus_index = int(np.argmax(spread))
    y_center = float(y_values[focus_index])
    y_width = max(float(y_values[-1] - y_values[0]) * 0.22, 1e-6)
    x_low = max(float(y_values[0]), y_center - y_width / 2.0)
    x_high = min(float(y_values[-1]), y_center + y_width / 2.0)
    x_low = max(float(y_values[0]), np.floor(x_low * 2.0) / 2.0)
    x_high = min(float(y_values[-1]), np.ceil(x_high * 2.0) / 2.0)
    focus_mask = (y_values >= x_low) & (y_values <= x_high)
    focus_values = profile_matrix[:, focus_mask]
    value_low = float(np.min(focus_values))
    value_high = float(np.max(focus_values))
    value_pad = max((value_high - value_low) * 0.12, np.finfo(float).eps)

    inset = inset_axes(
        ax,
        width="26%",
        height="28%",
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
    inset.yaxis.set_major_locator(MaxNLocator(nbins=3))
    _style_axes(inset)
    inset.tick_params(axis="both", labelsize=8, width=0.7, length=3)
    inset.tick_params(axis="y", labelleft=False)
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
    """Write one two-panel ux/uy profile figure for a material field."""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), sharex=True)
    panel_specs = (
        ("ux", (0.46, 0.12), "a"),
        ("uy", (0.22, 0.12), "b"),
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
        )
        if handles is None:
            handles, labels = panel_handles, panel_labels

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.79, 0.5),
        ncol=1,
        fontsize=10,
        frameon=False,
        handlelength=2.4,
        handletextpad=0.5,
        labelspacing=0.55,
    )
    fig.subplots_adjust(left=0.09, right=0.77, bottom=0.16, top=0.93, wspace=0.28)
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
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    nodes_list: tuple[int, ...] = DEFAULT_NODES,
    output_dir: Path | str = "results/grid_study/forward",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_list = tuple(sorted({int(nodes) for nodes in nodes_list}))
    if not nodes_list or max(nodes_list) not in nodes_list:
        raise ValueError("nodes_list must contain at least one mesh")

    raw_cases = {}
    manifest = {
        "study": "asm_forward_grid_convergence",
        "datasets": list(datasets),
        "nodes": list(nodes_list),
        "reference_nodes": max(nodes_list),
        "config": cfg.get_forward_config().to_dict(),
        "cases": [],
    }

    for dataset in datasets:
        normalized = str(dataset).strip().lower()
        if normalized not in {"bil", "exp"}:
            raise ValueError(f"Forward grid study supports only bil/exp, got {dataset!r}")
        raw_cases[normalized] = {}
        for nodes in nodes_list:
            case = run_forward_case(normalized, nodes)
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
                    {"dataset": normalized, "nodes": nodes, "elements": nodes - 1, "config": case["config"]},
                    indent=2,
                ),
                encoding="utf-8",
            )
            manifest["cases"].append({
                "dataset": normalized,
                "nodes": nodes,
                "elements": nodes - 1,
                "result_file": str((case_dir / "forward_result.pkl").relative_to(output_dir)),
            })

    reference_nodes = max(nodes_list)
    metric_rows = []
    profile_rows = []
    y_values = np.linspace(0.0, float(cfg.get_forward_config().geo_h), 181)
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
                "relative_l1_ux_to_reference": _relative_l1(case["ux"], ux_ref),
                "relative_l1_uy_to_reference": _relative_l1(case["uy"], uy_ref),
                "relative_l2_u_to_reference": float(
                    np.linalg.norm(u_case - u_ref) / (np.linalg.norm(u_ref) + 1e-15)
                ),
                "max_abs_u_difference": float(np.max(np.abs(u_case - u_ref))),
                "ux_norm": float(np.linalg.norm(case["ux"])),
                "uy_norm": float(np.linalg.norm(case["uy"])),
                "elapsed_time_seconds": case["elapsed_time_seconds"],
                "peak_python_memory_mb": case["peak_python_memory_mb"],
            })

            ux_profile = _interpolate_edge_profile(case["ux"], case["mesh"], y_values)
            uy_profile = _interpolate_edge_profile(case["uy"], case["mesh"], y_values)
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
    figure_paths = _plot_metrics(metric_rows, output_dir)
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
    manifest["edge_profile_figures"] = [
        str(path.relative_to(output_dir)) for path in edge_profile_figures
    ]
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved forward grid metrics to {metrics_path}")
    return metrics_path
