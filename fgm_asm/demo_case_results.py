"""Persistence and figures for one self-contained demo inverse case."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .l_curve import plot_lcurve_results
from .results_io import save_inverse_results, save_lcurve_analysis, write_python_config_snapshot
from .visualization import (
    plot_gradient_field,
    plot_iteration_history,
    plot_reconstruction_comparison,
    plot_reconstruction_results,
    reshape_nodal_values_for_plot,
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save_figure(fig, output_dir: Path, stem: str, dpi: int = 300) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_measured_displacement(mesh, measured: np.ndarray, output_dir: Path) -> None:
    ux = reshape_nodal_values_for_plot(mesh, measured[0::2])
    uy = reshape_nodal_values_for_plot(mesh, measured[1::2])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, field, title in zip(axes, (ux, uy), (r"Measured $u_x$", r"Measured $u_y$")):
        image = ax.pcolormesh(mesh.plot_x, mesh.plot_y, field, shading="gouraud", cmap="viridis")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.colorbar(image, ax=ax, fraction=0.0405, pad=0.04)
    fig.tight_layout()
    _save_figure(fig, output_dir, "measured_displacement")


def _plot_reconstruction_error(mesh, relative_error: np.ndarray, output_dir: Path) -> None:
    field = reshape_nodal_values_for_plot(mesh, relative_error)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.pcolormesh(mesh.plot_x, mesh.plot_y, field, shading="gouraud", cmap="hot")
    ax.set_title("Relative Modulus Error")
    ax.set_aspect("equal")
    ax.axis("off")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.0405, pad=0.04)
    colorbar.set_label("Error (%)")
    fig.tight_layout()
    _save_figure(fig, output_dir, "reconstruction_error")


def _write_lcurve_csv(lcurve_results: dict, output_dir: Path) -> Path:
    path = output_dir / "lcurve_curve.csv"
    optimal_idx = int(lcurve_results["optimal_idx"])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("gamma", "residual_norm", "regularization_norm", "curvature", "selected"))
        for index, values in enumerate(zip(
            lcurve_results["gamma_values"],
            lcurve_results["residual_norms"],
            lcurve_results["regularization_norms"],
            lcurve_results["curvature"],
        )):
            writer.writerow((*[float(value) for value in values], index == optimal_idx))
    return path


def save_demo_inverse_case(
    *,
    output_dir: Path,
    case: dict,
    case_config: dict,
    forward_config,
    inverse_config,
    lcurve_config,
    U_measured: np.ndarray,
    result: dict,
    errors: dict,
    metrics: dict,
    gamma: float,
    gamma_source: str,
    inverse_elapsed: float,
    inverse_peak_memory_mb: float,
    lcurve_results: dict | None = None,
    lcurve_elapsed: float = 0.0,
    lcurve_peak_memory_mb: float = 0.0,
    scan_result: dict | None = None,
    scan_errors: dict | None = None,
) -> dict[str, Path]:
    """Save all numerical data and PNG/PDF figures for one inverse case."""
    output_dir.mkdir(parents=True, exist_ok=True)
    noise_fraction = float(case_config["noise_level"]) / 100.0
    mesh = case["mesh"]

    np.savez_compressed(
        output_dir / "measured_displacement.npz",
        ux=U_measured[0::2].reshape(mesh.nods_y, mesh.nods_x, order="C"),
        uy=U_measured[1::2].reshape(mesh.nods_y, mesh.nods_x, order="C"),
        U_measured_C=U_measured,
        noise_percentage=float(case_config["noise_level"]),
    )
    np.savez_compressed(
        output_dir / "asm.npz",
        target=np.asarray(case["E_field"], dtype=float),
        prediction=np.asarray(result["E_final"], dtype=float).reshape(
            mesh.nods_y, mesh.nods_x, order="C"
        ),
        gamma=float(gamma),
        noise_percentage=float(case_config["noise_level"]),
        nodesx=int(case_config["nodesx"]),
        nodesy=int(case_config["nodesy"]),
    )

    inverse_path = save_inverse_results(
        result,
        errors,
        case["E_field"],
        noise_fraction,
        output_dir,
        filename="asm_results.pkl",
        extra_data={
            "case_config": case_config,
            "metrics": metrics,
            "gamma_used": float(gamma),
            "gamma_source": gamma_source,
            "U_clean": case["U"],
            "U_measured": U_measured,
            "inverse_elapsed_time_seconds": float(inverse_elapsed),
            "inverse_peak_python_memory_mb": float(inverse_peak_memory_mb),
            "lcurve_elapsed_time_seconds": float(lcurve_elapsed),
            "lcurve_peak_python_memory_mb": float(lcurve_peak_memory_mb),
            **({"scan_optimal": scan_result} if scan_result is not None else {}),
        },
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_value(metrics), indent=2), encoding="utf-8"
    )
    metadata = {
        "dataset": case_config["dataset"],
        "demo_sample": case["config"].get("sample_index"),
        "source_sample_index": case["config"].get("source_sample_index"),
        "nodesx": int(case_config["nodesx"]),
        "nodesy": int(case_config["nodesy"]),
        "noise_percentage": float(case_config["noise_level"]),
        "gamma": float(gamma),
        "gamma_source": gamma_source,
        "distribution_parameters": case["config"].get("distribution_parameters", {}),
    }
    (output_dir / "case_metadata.json").write_text(
        json.dumps(_json_value(metadata), indent=2), encoding="utf-8"
    )
    write_python_config_snapshot(
        output_dir,
        [
            ("Case Configuration", {key.upper(): value for key, value in case_config.items()}),
            ("Resolved Run", {"GAMMA_USED": gamma, "GAMMA_SOURCE": gamma_source}),
            ("Forward Configuration", forward_config.to_dict()),
            ("Inverse Configuration", inverse_config.to_dict()),
            ("L-curve Configuration", lcurve_config.to_dict()),
        ],
    )

    lcurve_path = None
    if lcurve_results is not None:
        lcurve_path = save_lcurve_analysis(
            lcurve_results,
            output_dir,
            extra_data={"gamma_optimal": gamma, "noise_level": noise_fraction},
        )
        _write_lcurve_csv(lcurve_results, output_dir)
        figures = plot_lcurve_results(lcurve_results, save_path=output_dir)
        for figure in figures:
            plt.close(figure)

    figures = [
        plot_reconstruction_results(
            mesh, case["E_field"], result["E_final"], errors, noise_fraction,
            save_path=output_dir, filename_stem="reconstruction_results",
        ),
        plot_iteration_history(
            result, save_path=output_dir, noise_level=noise_fraction,
            filename_stem="iteration_history",
        ),
        plot_gradient_field(
            mesh, result, noise_level=noise_fraction, save_path=output_dir,
            filename_stem="gradient_field",
        ),
    ]
    if scan_result is not None and scan_errors is not None:
        figures.append(plot_reconstruction_comparison(
            mesh, case["E_field"], scan_result["E_final"], scan_errors,
            result["E_final"], errors, noise_fraction, save_path=output_dir,
        ))
    for figure in figures:
        if figure is not None:
            plt.close(figure)
    _plot_measured_displacement(mesh, U_measured, output_dir)
    _plot_reconstruction_error(mesh, errors["rel_error_field"], output_dir)
    return {"output_dir": output_dir, "inverse_results": inverse_path, "lcurve": lcurve_path}


__all__ = ["save_demo_inverse_case"]
