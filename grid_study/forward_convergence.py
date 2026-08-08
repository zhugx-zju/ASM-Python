"""Batch forward-problem mesh convergence study.

This module intentionally covers only the forward problem.  It regenerates the
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.interpolate import RegularGridInterpolator

from fgm_asm import MeshInfo, fem_assemble, forward_solver, generate_fgm_modulus
from fgm_asm.mesh import setup_boundary_conditions
import config as cfg


DEFAULT_DATASETS = ("bil", "exp")
DEFAULT_NODES = (10, 20, 30, 40, 60, 80, 100)

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "Times New Roman"
rcParams["mathtext.it"] = "Times New Roman:italic"
rcParams["axes.linewidth"] = 1.2


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
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = sorted((row for row in rows if row["dataset"] == dataset), key=lambda r: r["nodes"])
        nodes = [row["nodes"] for row in subset]
        axes[0].plot(nodes, [row["relative_l2_u_to_reference"] for row in subset], "o-", label=dataset.upper())
        axes[1].plot(nodes, [row["elapsed_time_seconds"] for row in subset], "o-", label=dataset.upper())

    axes[0].set_xlabel("Nodes per direction")
    axes[0].set_ylabel("Relative L2 displacement difference to finest mesh")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Nodes per direction")
    axes[1].set_ylabel("Forward solve time (s)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.suptitle("ASM Forward Mesh Convergence")
    fig.tight_layout()
    png_path = output_dir / "forward_grid_convergence.png"
    pdf_path = output_dir / "forward_grid_convergence.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
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


def _plot_single_edge_profile(
    cases: dict[int, dict],
    dataset: str,
    component: str,
    nodes_list: tuple[int, ...],
    output_dir: Path,
    y_values: np.ndarray,
) -> tuple[Path, Path]:
    """Plot one component for one material field with all mesh curves."""
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    colors = plt.cm.GnBu(np.linspace(0.35, 0.95, len(nodes_list)))
    profiles = []

    for color, nodes in zip(colors, nodes_list):
        case = cases[nodes]
        profile = _interpolate_edge_profile(case[component], case["mesh"], y_values)
        profiles.append(profile)
        ax.plot(
            y_values,
            profile,
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
            label=f"{nodes} x {nodes} Nodes",
        )

    ax.set_xlabel(r"$y$", fontsize=16)
    ax.set_ylabel(rf"${component[0]}_{{{component[1:]}}}$", fontsize=16)
    ax.set_title(
        rf"{dataset.upper()} modulus field: ${component[0]}_{{{component[1:]}}}$ along $x=L$",
        fontsize=15,
        pad=10,
    )
    ax.tick_params(axis="both", labelsize=12, width=1.1)
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.legend(
        loc="upper right",
        fontsize=10,
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=0.92,
    )

    # Add a small zoom around the location where mesh curves differ most.
    profile_matrix = np.asarray(profiles)
    spread = np.ptp(profile_matrix, axis=0)
    focus_index = int(np.argmax(spread))
    y_center = float(y_values[focus_index])
    y_width = max(float(y_values[-1] - y_values[0]) * 0.22, 1e-6)
    x_low = max(float(y_values[0]), y_center - y_width / 2.0)
    x_high = min(float(y_values[-1]), y_center + y_width / 2.0)
    focus_mask = (y_values >= x_low) & (y_values <= x_high)
    focus_values = profile_matrix[:, focus_mask]
    value_low = float(np.min(focus_values))
    value_high = float(np.max(focus_values))
    value_pad = max((value_high - value_low) * 0.12, np.finfo(float).eps)

    inset = inset_axes(ax, width="32%", height="31%", loc="lower left", borderpad=2.0)
    for color, profile in zip(colors, profiles):
        inset.plot(y_values, profile, color=color, linewidth=1.4)
    inset.set_xlim(x_low, x_high)
    inset.set_ylim(value_low - value_pad, value_high + value_pad)
    # Keep the zoom scale readable without colliding with the main y-axis.
    inset.tick_params(axis="x", labelsize=8, width=0.8)
    inset.tick_params(axis="y", labelleft=False, width=0.8)
    inset.grid(True, alpha=0.18, linewidth=0.5)
    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="black", lw=0.8)

    # The inset is positioned in axes coordinates, so explicit margins keep
    # the single-axis figure stable without tight_layout warnings.
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.12, top=0.88)
    stem = f"forward_edge_{dataset}_{component.lower()}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
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
    """Write one single-axis figure per dataset and displacement component."""
    paths = []
    for dataset in datasets:
        normalized = str(dataset).lower()
        for component in ("ux", "uy"):
            paths.extend(
                _plot_single_edge_profile(
                    cases[normalized],
                    normalized,
                    component,
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
