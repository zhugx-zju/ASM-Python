"""
Reduced-Hessian diagnostics for the inverse modulus reconstruction problem.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from .inverse_solver import apply_reduced_gauss_newton_hessian


def build_gauss_newton_operator(mesh_info, state, gamma):
    """
    Build a symmetric LinearOperator for the reduced Gauss-Newton Hessian.
    """
    n_nod = mesh_info.n_nod

    def matvec(vec):
        h_vec, _ = apply_reduced_gauss_newton_hessian(mesh_info, state, gamma, vec)
        return h_vec

    return LinearOperator((n_nod, n_nod), matvec=matvec, dtype=float)


def _safe_smallest_eigs(operator, k):
    """Estimate the smallest algebraic eigenpairs of a symmetric operator."""
    n = operator.shape[0]
    k = max(1, min(int(k), max(n - 2, 1)))
    try:
        vals, vecs = eigsh(operator, k=k, which="SA")
    except Exception:
        dense_cols = []
        eye = np.eye(n)
        for i in range(n):
            dense_cols.append(operator @ eye[:, i])
        dense = np.column_stack(dense_cols)
        vals, vecs = np.linalg.eigh(dense)
        vals = vals[:k]
        vecs = vecs[:, :k]
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def _safe_largest_eig(operator):
    """Estimate the largest algebraic eigenvalue of a symmetric operator."""
    n = operator.shape[0]
    if n <= 2:
        dense_cols = []
        eye = np.eye(n)
        for i in range(n):
            dense_cols.append(operator @ eye[:, i])
        dense = np.column_stack(dense_cols)
        vals = np.linalg.eigvalsh(dense)
        return float(np.max(vals))

    try:
        vals = eigsh(operator, k=1, which="LA", return_eigenvectors=False)
        return float(vals[0])
    except Exception:
        dense_cols = []
        eye = np.eye(n)
        for i in range(n):
            dense_cols.append(operator @ eye[:, i])
        dense = np.column_stack(dense_cols)
        vals = np.linalg.eigvalsh(dense)
        return float(np.max(vals))


def analyze_reduced_hessian(mesh_info, state, gamma, n_eigs=6, kernel_probe_count=3, tol=1e-10):
    """
    Estimate local uniqueness and ill-conditioning diagnostics in Ehat space.

    Args:
        mesh_info: MeshInfo object
        state: Cached inverse state dictionary
        gamma: Regularization coefficient
        n_eigs: Number of smallest eigenpairs to estimate
        kernel_probe_count: Number of softest modes to inspect
        tol: Numerical threshold for "near-zero" diagnostics only

    Returns:
        diagnostics: Dictionary of spectral and kernel-probe information
    """
    operator = build_gauss_newton_operator(mesh_info, state, gamma)
    smallest_vals, smallest_vecs = _safe_smallest_eigs(operator, n_eigs)
    largest_val = _safe_largest_eig(operator)
    lambda_min = float(smallest_vals[0])
    lambda_max = float(largest_val)
    if lambda_min > 0.0:
        condition_number = float(lambda_max / lambda_min)
    else:
        condition_number = float("inf")

    g_matrix = mesh_info.get_regularization_matrix()
    kernel_probe_count = max(1, min(int(kernel_probe_count), smallest_vecs.shape[1]))
    kernel_probes = []

    for i in range(kernel_probe_count):
        vec = smallest_vecs[:, i]
        vec_norm = np.linalg.norm(vec)
        if vec_norm <= 0.0:
            continue
        vec = vec / vec_norm
        _, d_u = apply_reduced_gauss_newton_hessian(mesh_info, state, gamma, vec)
        data_visibility = float(d_u @ np.asarray(mesh_info.M @ d_u).ravel())
        reg_energy = float(vec @ np.asarray(g_matrix @ vec).ravel())
        total_energy = float(vec @ (operator @ vec))
        kernel_probes.append(
            {
                "mode_index": int(i),
                "hessian_rayleigh": total_energy,
                "data_visibility": data_visibility,
                "regularization_energy": reg_energy,
                "is_near_invisible": bool(data_visibility <= tol),
                "is_near_unpenalized": bool(reg_energy <= tol),
            }
        )

    return {
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "condition_number": condition_number,
        "has_negative_curvature": bool(lambda_min < -tol),
        "near_nullspace_detected": bool(lambda_min <= tol),
        "smallest_eigenvalues": np.asarray(smallest_vals, dtype=float),
        "n_eigs": int(n_eigs),
        "kernel_probes": kernel_probes,
        "analysis_tolerance": float(tol),
    }
