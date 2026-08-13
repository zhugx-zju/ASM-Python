"""Independently runnable inverse workflow for repository-local demo cases."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from grid_study.demo_data import generate_demo_case, load_demo_dataset

from .case_paths import resolve_demo_inverse_case_dir
from .config_types import (
    ForwardConfig,
    InverseConfig,
    LCurveConfig,
    coerce_forward_config,
    coerce_inverse_config,
    coerce_lcurve_config,
)
from .demo_case_results import save_demo_inverse_case
from .fem_forward import fem_assemble, forward_solver
from .inverse_solver import lbfgs_inverse_solver_scipy
from .l_curve import find_optimal_gamma_lcurve
from .mesh import MeshInfo, setup_boundary_conditions
from .utils import add_noise_to_displacement


SUPPORTED_DATASETS = {"bil", "exp", "grf"}
DEFAULT_OUTPUT_ROOT = Path("results/grid_study/inverse_demo")


def _compute_errors(E_true: np.ndarray, E_reconstructed: np.ndarray) -> dict[str, Any]:
    """Compute plot-compatible errors safely for near-zero modulus values."""
    truth = np.asarray(E_true, dtype=float).ravel(order="C")
    prediction = np.asarray(E_reconstructed, dtype=float).ravel(order="C")
    relative_error = 100.0 * np.abs(truth - prediction) / (np.abs(truth) + 1e-15)
    return {
        "mae": float(np.mean(relative_error)),
        "max_error": float(np.max(relative_error)),
        "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
        "rel_error_field": relative_error,
    }


def _normalise_case(
    case: Mapping[str, Any],
    inverse_config: InverseConfig,
    lcurve_config: LCurveConfig,
) -> dict[str, Any]:
    dataset = str(case["dataset"]).strip().lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported demo dataset: {dataset!r}")
    nodesx = int(case.get("nodesx", case.get("nodes", 40)))
    nodesy = int(case.get("nodesy", nodesx))
    if nodesx != nodesy:
        raise ValueError(
            "The current demo grid study uses square meshes; nodesx and nodesy must match"
        )
    if nodesx < 2:
        raise ValueError("nodesx and nodesy must be at least 2")

    noise = float(case.get("noise_level", case.get("noise_percentage", 0.0)))
    if noise < 0:
        raise ValueError("noise_level is a percentage and cannot be negative")
    enable_lcurve = bool(case.get("enable_lcurve", case.get("gamma") is None))
    gamma = case.get("gamma", case.get("asm_gamma"))
    if not enable_lcurve and gamma is None:
        gamma = inverse_config.gamma

    normalized = {
        "dataset": dataset,
        "nodesx": nodesx,
        "nodesy": nodesy,
        "noise_level": noise,
        "noise_seed": int(case.get("noise_seed", 42)),
        "gamma": None if gamma is None else float(gamma),
        "enable_lcurve": enable_lcurve,
        "lcurve_points": int(case.get("lcurve_points", case.get("n_gamma", lcurve_config.n_gamma))),
        "lcurve_gamma_min": float(case.get("lcurve_gamma_min", case.get("gamma_min", lcurve_config.gamma_min))),
        "lcurve_gamma_max": float(case.get("lcurve_gamma_max", case.get("gamma_max", lcurve_config.gamma_max))),
        "max_iter": int(case.get("max_iter", case.get("asm_max_iter", inverse_config.max_iter))),
        "ftol": float(case.get("ftol", case.get("asm_ftol", inverse_config.ftol))),
        "gtol": float(case.get("gtol", case.get("asm_gtol", inverse_config.gtol))),
        "E_min": float(case.get("E_min", inverse_config.E_min)),
        "E_max": float(case.get("E_max", inverse_config.E_max)),
        "overwrite": bool(case.get("overwrite", False)),
        "output_root": str(case.get("output_root", DEFAULT_OUTPUT_ROOT)),
    }
    if normalized["enable_lcurve"] and normalized["lcurve_points"] < 5:
        raise ValueError("lcurve_points must be at least 5 for interior curvature selection")
    if normalized["lcurve_gamma_min"] <= 0 or normalized["lcurve_gamma_max"] <= 0:
        raise ValueError("L-curve gamma bounds must be positive")
    return normalized


def _mass_norm(mesh, vector: np.ndarray) -> float:
    if mesh.M is None:
        mesh.assemble_mass_matrix()
    value = float(vector @ (mesh.M @ vector))
    return float(np.sqrt(max(value, 0.0)))


def _build_metrics(
    case: dict,
    result: dict,
    measured: np.ndarray,
    case_config: dict,
    gamma: float,
    inverse_elapsed: float,
    peak_memory_mb: float,
    lcurve_elapsed: float,
) -> dict[str, Any]:
    truth = np.asarray(case["E_field"], dtype=float).ravel(order="C")
    prediction = np.asarray(result["E_final"], dtype=float)
    difference = prediction - truth
    clean = np.asarray(case["U"], dtype=float)
    final_displacement = np.asarray(result["U_final"], dtype=float)
    clean_norm = _mass_norm(case["mesh"], clean)
    measured_norm = _mass_norm(case["mesh"], measured)
    return {
        "dataset": case_config["dataset"],
        "nodesx": case_config["nodesx"],
        "nodesy": case_config["nodesy"],
        "noise_percentage": case_config["noise_level"],
        "gamma": float(gamma),
        "relative_l1_E": float(np.sum(np.abs(difference)) / (np.sum(np.abs(truth)) + 1e-15)),
        "relative_l2_E": float(np.linalg.norm(difference) / (np.linalg.norm(truth) + 1e-15)),
        "relative_linf_E": float(np.max(np.abs(difference)) / (np.max(np.abs(truth)) + 1e-15)),
        "mae_E_percent": float(np.mean(np.abs(difference) / (np.abs(truth) + 1e-15)) * 100.0),
        "max_E_error_percent": float(np.max(np.abs(difference) / (np.abs(truth) + 1e-15)) * 100.0),
        "rmse_E": float(np.sqrt(np.mean(difference * difference))),
        "relative_displacement_fit_to_clean": _mass_norm(case["mesh"], final_displacement - clean) / (clean_norm + 1e-15),
        "relative_displacement_fit_to_measured": _mass_norm(case["mesh"], final_displacement - measured) / (measured_norm + 1e-15),
        "relative_noise_to_clean_displacement": _mass_norm(case["mesh"], measured - clean) / (clean_norm + 1e-15),
        "initial_cost": float(result["cost_history"][0]),
        "final_cost": float(result["final_cost"]),
        "residual_norm": float(result["residual_norm"]),
        "regularization_norm": float(result["regularization_norm"]),
        "iterations": int(result["n_iterations"]),
        "converged": bool(result["converged"]),
        "inverse_elapsed_time_seconds": float(inverse_elapsed),
        "lcurve_elapsed_time_seconds": float(lcurve_elapsed),
        "forward_elapsed_time_seconds": float(case["elapsed_time_seconds"]),
        "peak_python_memory_mb": float(peak_memory_mb),
        "forward_peak_python_memory_mb": float(case["peak_python_memory_mb"]),
    }


def _measure(function):
    tracemalloc.start()
    start = time.perf_counter()
    try:
        value = function()
    finally:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return value, float(time.perf_counter() - start), float(peak_bytes / 1024**2)


def _build_demo_forward_context(
    dataset: str,
    nodes: int,
    forward_config: ForwardConfig,
    demo_data: dict,
) -> dict[str, Any]:
    """Build the clean numerical context required by one inverse case."""
    mesh = MeshInfo(
        forward_config.geo_l,
        forward_config.geo_h,
        nodes - 1,
        nodes - 1,
    )

    def solve():
        generated = generate_demo_case(demo_data, mesh, forward_config)
        modulus = np.asarray(generated["E_field"], dtype=float)
        material = generated["material_info"]
        material.update(modulus.ravel(order="C"), iteration=1)
        boundary = setup_boundary_conditions(
            mesh,
            forward_config.geo_l,
            forward_config.geo_h,
            forward_config.f_tot,
        )
        displacement = forward_solver(fem_assemble(mesh, material, boundary))
        return modulus, boundary, np.asarray(displacement, dtype=float)

    (modulus, boundary, displacement), elapsed, peak_memory = _measure(solve)
    spec = demo_data["spec"]
    case_config = forward_config.to_dict()
    case_config.update({
        "nel_x": nodes - 1,
        "nel_y": nodes - 1,
        "sample_index": int(spec["sample_index"]),
        "source_sample_index": int(spec["source_sample_index"]),
        "distribution_parameters": spec["parameters"],
    })
    return {
        "dataset": dataset,
        "nodes": nodes,
        "elements": nodes - 1,
        "mesh": mesh,
        "bc_info": boundary,
        "E_field": modulus,
        "U": displacement,
        "elapsed_time_seconds": elapsed,
        "peak_python_memory_mb": peak_memory,
        "config": case_config,
    }


def run_demo_inverse_case(
    *,
    project_root: Path | str,
    forward_config: ForwardConfig | Mapping[str, Any],
    inverse_config: InverseConfig | Mapping[str, Any],
    lcurve_config: LCurveConfig | Mapping[str, Any],
    case: Mapping[str, Any],
) -> dict[str, Any]:
    """Run and save one ``distribution + mesh + noise`` demo inverse case."""
    project_root = Path(project_root).resolve()
    forward_config = coerce_forward_config(forward_config)
    inverse_config = coerce_inverse_config(inverse_config)
    lcurve_config = coerce_lcurve_config(lcurve_config)
    case_config = _normalise_case(case, inverse_config, lcurve_config)
    output_dir = resolve_demo_inverse_case_dir(
        project_root,
        case_config["dataset"],
        case_config["nodesx"],
        case_config["nodesy"],
        case_config["noise_level"],
        case_config["output_root"],
    )
    completed_file = output_dir / "asm_results.pkl"
    if completed_file.exists() and not case_config["overwrite"]:
        print(f"[SKIP] Existing demo inverse case: {output_dir}")
        return {"status": "skipped", "output_dir": output_dir, "case": case_config}

    print(
        f"[RUN] {case_config['dataset'].upper()} | "
        f"{case_config['nodesx']}x{case_config['nodesy']} nodes | "
        f"noise={case_config['noise_level']:g}%"
    )
    demo_data = load_demo_dataset(case_config["dataset"], project_root / "demo_distributions")
    forward_case = _build_demo_forward_context(
        case_config["dataset"],
        case_config["nodesx"],
        forward_config,
        demo_data,
    )
    mesh = forward_case["mesh"]
    if mesh.M is None:
        mesh.assemble_mass_matrix()
    measured = add_noise_to_displacement(
        forward_case["U"],
        case_config["noise_level"] / 100.0,
        seed=case_config["noise_seed"],
    )

    resolved_inverse_config = replace(
        inverse_config,
        E_min=case_config["E_min"],
        E_max=case_config["E_max"],
        max_iter=case_config["max_iter"],
        ftol=case_config["ftol"],
        gtol=case_config["gtol"],
    )
    resolved_forward_config = replace(
        forward_config,
        nel_x=case_config["nodesx"] - 1,
        nel_y=case_config["nodesy"] - 1,
        dis_type=case_config["dataset"],
    )
    resolved_lcurve_config = replace(
        lcurve_config,
        gamma_min=case_config["lcurve_gamma_min"],
        gamma_max=case_config["lcurve_gamma_max"],
        n_gamma=case_config["lcurve_points"],
        E_min=case_config["E_min"],
        E_max=case_config["E_max"],
        max_iter=case_config["max_iter"],
        ftol=case_config["ftol"],
        gtol=case_config["gtol"],
    )

    lcurve_results = None
    scan_result = None
    scan_errors = None
    lcurve_elapsed = 0.0
    lcurve_peak = 0.0
    if case_config["enable_lcurve"]:
        (gamma, lcurve_results), lcurve_elapsed, lcurve_peak = _measure(
            lambda: find_optimal_gamma_lcurve(
                mesh_info=mesh,
                bc_info=forward_case["bc_info"],
                U_measured=measured,
                config=forward_config,
                gamma_min=resolved_lcurve_config.gamma_min,
                gamma_max=resolved_lcurve_config.gamma_max,
                n_gamma=resolved_lcurve_config.n_gamma,
                E_min=resolved_lcurve_config.E_min,
                E_max=resolved_lcurve_config.E_max,
                max_iter=resolved_lcurve_config.max_iter,
                ftol=resolved_lcurve_config.ftol,
                gtol=resolved_lcurve_config.gtol,
            )
        )
        optimal_idx = int(lcurve_results["optimal_idx"])
        scan_result = lcurve_results["all_results"][optimal_idx]
        scan_errors = _compute_errors(forward_case["E_field"], scan_result["E_final"])
        gamma_source = "lcurve_max_curvature"
    else:
        gamma = float(case_config["gamma"])
        gamma_source = "fixed"

    result, inverse_elapsed, inverse_peak = _measure(
        lambda: lbfgs_inverse_solver_scipy(
            mesh_info=mesh,
            bc_info=forward_case["bc_info"],
            U_measured=measured,
            E_init=None,
            gamma=float(gamma),
            E_min=resolved_inverse_config.E_min,
            E_max=resolved_inverse_config.E_max,
            max_iter=resolved_inverse_config.max_iter,
            ftol=resolved_inverse_config.ftol,
            gtol=resolved_inverse_config.gtol,
            nu=forward_config.nu,
        )
    )
    errors = _compute_errors(forward_case["E_field"], result["E_final"])
    metrics = _build_metrics(
        forward_case, result, measured, case_config, float(gamma),
        inverse_elapsed, inverse_peak, lcurve_elapsed,
    )
    saved = save_demo_inverse_case(
        output_dir=output_dir,
        case=forward_case,
        case_config=case_config,
        forward_config=resolved_forward_config,
        inverse_config=resolved_inverse_config,
        lcurve_config=resolved_lcurve_config,
        U_measured=measured,
        result=result,
        errors=errors,
        metrics=metrics,
        gamma=float(gamma),
        gamma_source=gamma_source,
        inverse_elapsed=inverse_elapsed,
        inverse_peak_memory_mb=inverse_peak,
        lcurve_results=lcurve_results,
        lcurve_elapsed=lcurve_elapsed,
        lcurve_peak_memory_mb=lcurve_peak,
        scan_result=scan_result,
        scan_errors=scan_errors,
    )
    print(
        f"[DONE] {output_dir} | gamma={float(gamma):.3e} | "
        f"converged={metrics['converged']} | relative_l2={metrics['relative_l2_E']:.3e}"
    )
    return {
        "status": "completed",
        "output_dir": output_dir,
        "case": case_config,
        "metrics": metrics,
        "gamma": float(gamma),
        **saved,
    }


def run_demo_inverse_cases(
    *,
    project_root: Path | str,
    forward_config: ForwardConfig | Mapping[str, Any],
    inverse_config: InverseConfig | Mapping[str, Any],
    lcurve_config: LCurveConfig | Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Run manual demo cases independently in the order supplied by the user."""
    if not cases:
        raise ValueError("cases cannot be empty")
    return [
        run_demo_inverse_case(
            project_root=project_root,
            forward_config=forward_config,
            inverse_config=inverse_config,
            lcurve_config=lcurve_config,
            case=case,
        )
        for case in cases
    ]


__all__ = ["run_demo_inverse_case", "run_demo_inverse_cases"]
