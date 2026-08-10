"""Numerical metrics used by grid-study workflows."""

from __future__ import annotations

import numpy as np

from fgm_asm import MeshInfo


def relative_linf(a: np.ndarray, b: np.ndarray) -> float:
    """Return the infinity-norm error relative to ``b``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = float(np.max(np.abs(b)))
    return float(np.max(np.abs(a - b)) / (denominator + 1e-15))


def mass_norm(mesh: MeshInfo, vector: np.ndarray) -> float:
    """Return the displacement norm induced by the mesh mass matrix."""
    if mesh.M is None:
        mesh.assemble_mass_matrix()
    vector = np.asarray(vector, dtype=float)
    value = float(vector @ (mesh.M @ vector))
    return float(np.sqrt(max(value, 0.0)))


def relative_norm(numerator: np.ndarray, denominator: np.ndarray) -> float:
    """Return an L2 norm ratio."""
    return float(
        np.linalg.norm(np.asarray(numerator, dtype=float))
        / (np.linalg.norm(np.asarray(denominator, dtype=float)) + 1e-15)
    )


def inverse_metrics(
    case: dict,
    result: dict,
    U_measured: np.ndarray,
    noise_percentage: float,
    gamma: float,
    inverse_elapsed: float,
    peak_memory_mb: float,
) -> dict:
    """Build scalar accuracy, data-fit, optimization, and cost metrics."""
    E_true = np.asarray(case["E_field"], dtype=float).ravel(order="C")
    E_reconstructed = np.asarray(result["E_final"], dtype=float)
    E_difference = E_reconstructed - E_true
    U_clean = np.asarray(case["U"], dtype=float)
    U_measured = np.asarray(U_measured, dtype=float)
    U_final = np.asarray(result["U_final"], dtype=float)

    clean_norm = mass_norm(case["mesh"], U_clean)
    measured_norm = mass_norm(case["mesh"], U_measured)
    return {
        "dataset": case["dataset"],
        "nodes": int(case["nodes"]),
        "elements": int(case["elements"]),
        "noise_percentage": float(noise_percentage),
        "gamma": float(gamma),
        "relative_l1_E": float(np.sum(np.abs(E_difference)) / (np.sum(np.abs(E_true)) + 1e-15)),
        "relative_l2_E": relative_norm(E_difference, E_true),
        "relative_linf_E": float(np.max(np.abs(E_difference)) / (np.max(np.abs(E_true)) + 1e-15)),
        "mae_E_percent": float(np.mean(np.abs(E_difference) / (np.abs(E_true) + 1e-15)) * 100.0),
        "max_E_error_percent": float(np.max(np.abs(E_difference) / (np.abs(E_true) + 1e-15)) * 100.0),
        "rmse_E": float(np.sqrt(np.mean(E_difference * E_difference))),
        "relative_displacement_fit_to_clean": mass_norm(case["mesh"], U_final - U_clean) / (clean_norm + 1e-15),
        "relative_displacement_fit_to_measured": mass_norm(case["mesh"], U_final - U_measured) / (measured_norm + 1e-15),
        "relative_noise_to_clean_displacement": mass_norm(case["mesh"], U_measured - U_clean) / (clean_norm + 1e-15),
        "initial_cost": float(result["cost_history"][0]),
        "final_cost": float(result["final_cost"]),
        "residual_norm": float(result["residual_norm"]),
        "regularization_norm": float(result["regularization_norm"]),
        "iterations": int(result["n_iterations"]),
        "converged": bool(result["converged"]),
        "inverse_elapsed_time_seconds": float(inverse_elapsed),
        "forward_elapsed_time_seconds": float(case["elapsed_time_seconds"]),
        "peak_python_memory_mb": float(peak_memory_mb),
        "forward_peak_python_memory_mb": float(case["peak_python_memory_mb"]),
    }


__all__ = ["inverse_metrics", "mass_norm", "relative_linf", "relative_norm"]
