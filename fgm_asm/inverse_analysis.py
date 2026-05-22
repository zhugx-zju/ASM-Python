"""
Explicit reduced-Hessian diagnostics for the inverse modulus reconstruction problem.
"""

from __future__ import annotations

import numpy as np

def assemble_stiffness_parameter_matrix(fem_info, forw_U):
    """
    Assemble the matrix ``[(dK/dE_1)U, ..., (dK/dE_n)U]`` in global DOF space.

    Args:
        fem_info: FEMInfo object at the current state
        forw_U: Forward displacement vector [n_dof]

    Returns:
        global_mat: Global DOF matrix [n_dof, n_nod]
    """
    mesh_info = fem_info.mesh_info
    u_ele = np.asarray(forw_U, dtype=float).ravel()[mesh_info.ele_dof_id]
    nod_ids = mesh_info.ele_nods_id - 1

    rhs_ele = np.zeros((mesh_info.n_el, 8, mesh_info.n_nod), dtype=float)

    for ig in range(mesh_info.gauss_N.shape[0]):
        b_i = fem_info.B_gauss[ig]
        shape_vals = mesh_info.gauss_N[ig]
        strain = np.einsum('eia,ea->ei', b_i, u_ele)
        stress = np.einsum('ij,ej->ei', fem_info.D0, strain)
        internal = np.einsum('eia,ei->ea', b_i, stress)
        weight = mesh_info.gauss_w[ig] * fem_info.det_j_gauss[ig]
        weighted_internal = weight[:, None] * internal

        for local_node in range(shape_vals.size):
            rhs_ele[:, :, nod_ids[:, local_node]] += (
                shape_vals[local_node] * weighted_internal
            )[:, :, None]

    global_mat = np.zeros((mesh_info.n_dof, mesh_info.n_nod), dtype=float)
    np.add.at(global_mat, mesh_info.ele_dof_id, rhs_ele)
    return global_mat

def assemble_displacement_sensitivity_matrix(mesh_info, state):
    """
    Assemble the displacement sensitivity matrix ``J_U = dU/dE`` explicitly.

    Args:
        mesh_info: MeshInfo object
        state: Cached inverse state dictionary built at the analysis modulus field

    Returns:
        sensitivity_matrix: Dense matrix [n_dof, n_nod]
    """
    fem_info = state["fem_info"]
    forw_U = state["forw_U"]
    rhs_matrix = -assemble_stiffness_parameter_matrix(fem_info, forw_U)

    sensitivity_matrix = np.zeros((mesh_info.n_dof, mesh_info.n_nod), dtype=float)
    free_dof = fem_info.free_dof
    k_free = fem_info.K[free_dof][:, free_dof].toarray()
    rhs_free = rhs_matrix[free_dof, :]

    sensitivity_matrix[free_dof, :] = np.linalg.solve(k_free, rhs_free)
    return sensitivity_matrix


def assemble_explicit_data_hessian(mesh_info, state):
    """
    Assemble the data reduced Hessian ``J_U^T M J_U`` explicitly.

    Args:
        mesh_info: MeshInfo object
        state: Cached inverse state dictionary built at the analysis modulus field

    Returns:
        data_hessian: Dense symmetric matrix [n_nod, n_nod]
        sensitivity_matrix: Dense displacement sensitivity matrix [n_dof, n_nod]
    """
    sensitivity_matrix = assemble_displacement_sensitivity_matrix(mesh_info, state)
    data_hessian = sensitivity_matrix.T @ np.asarray(mesh_info.M @ sensitivity_matrix, dtype=float)
    return data_hessian, sensitivity_matrix


def _build_data_smallest_eigenpair_summaries(eigvals, eigvecs, eigenpair_count):
    """Build per-mode summaries for the smallest-eigenvalue eigenpairs."""
    eigenpair_summaries = []

    for i in range(eigenpair_count):
        vec = eigvecs[:, i]
        eigenvalue = float(eigvals[i])
        eigenpair_summaries.append(
            {
                "mode_index": int(i),
                "eigenvalue": eigenvalue,
                "vector": np.array(vec, copy=True),
                "data_energy": eigenvalue,
                "min_component": float(np.min(vec)),
                "max_component": float(np.max(vec)),
                "mean_component": float(np.mean(vec)),
            }
        )

    return eigenpair_summaries


def _build_total_smallest_eigenpair_summaries(eigvals, eigvecs, reg_matrix, gamma, eigenpair_count):
    """Build per-mode summaries for the smallest regularized eigenpairs."""
    eigenpair_summaries = []

    for i in range(eigenpair_count):
        vec = eigvecs[:, i]
        eigenvalue = float(eigvals[i])
        regularization_energy = float(vec @ (reg_matrix @ vec))
        eigenpair_summaries.append(
            {
                "mode_index": int(i),
                "eigenvalue": eigenvalue,
                "vector": np.array(vec, copy=True),
                "data_energy": eigenvalue - gamma * regularization_energy,
                "regularization_energy": regularization_energy,
                "total_energy": eigenvalue,
                "min_component": float(np.min(vec)),
                "max_component": float(np.max(vec)),
                "mean_component": float(np.mean(vec)),
            }
        )

    return eigenpair_summaries


def _summarize_data_spectrum(matrix, operator_name, n_eigs, tol=None):
    """Return spectral summaries for an explicitly assembled symmetric matrix."""
    eigvals, eigvecs = np.linalg.eigh(matrix)

    lambda_min = float(eigvals[0])
    lambda_max = float(eigvals[-1])
    if lambda_min != 0.0:
        condition_number = float(abs(lambda_max / lambda_min))
    else:
        condition_number = float("inf")

    eigenpair_count = max(1, min(int(n_eigs), eigvals.size))
    eigenpair_summaries = _build_data_smallest_eigenpair_summaries(
        eigvals,
        eigvecs,
        eigenpair_count,
    )

    return {
        "space": "physical_modulus_E",
        "operator_name": operator_name,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "condition_number": condition_number,
        "is_positive_definite": bool(lambda_min > 0.0),
        "has_negative_curvature": bool(lambda_min < 0.0),
        "near_nullspace_detected": bool(lambda_min == 0.0),
        "smallest_eigenvalues": np.asarray(eigvals[:eigenpair_count], dtype=float),
        "largest_eigenvalue": lambda_max,
        "smallest_eigenvectors": np.asarray(eigvecs[:, :eigenpair_count], dtype=float),
        "smallest_eigenpair_summaries": eigenpair_summaries,
        "n_eigs": int(eigenpair_count),
    }


def _summarize_total_spectrum(matrix, reg_matrix, gamma, operator_name, n_eigs):
    """Return spectral summaries for the regularized reduced Hessian."""
    eigvals, eigvecs = np.linalg.eigh(matrix)

    lambda_min = float(eigvals[0])
    lambda_max = float(eigvals[-1])
    if lambda_min != 0.0:
        condition_number = float(abs(lambda_max / lambda_min))
    else:
        condition_number = float("inf")

    eigenpair_count = max(1, min(int(n_eigs), eigvals.size))
    eigenpair_summaries = _build_total_smallest_eigenpair_summaries(
        eigvals,
        eigvecs,
        reg_matrix,
        gamma,
        eigenpair_count,
    )

    return {
        "space": "physical_modulus_E",
        "operator_name": operator_name,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "condition_number": condition_number,
        "is_positive_definite": bool(lambda_min > 0.0),
        "has_negative_curvature": bool(lambda_min < 0.0),
        "near_nullspace_detected": bool(lambda_min == 0.0),
        "smallest_eigenvalues": np.asarray(eigvals[:eigenpair_count], dtype=float),
        "largest_eigenvalue": lambda_max,
        "smallest_eigenvectors": np.asarray(eigvecs[:, :eigenpair_count], dtype=float),
        "smallest_eigenpair_summaries": eigenpair_summaries,
        "n_eigs": int(eigenpair_count),
    }


def analyze_reduced_hessian(mesh_info, state, gamma, n_eigs=6, tol=1e-10):
    """
    Explicitly analyze the physical-E-space reduced Hessians at the converged state.

    Args:
        mesh_info: MeshInfo object
        state: Cached inverse state dictionary built at the analysis modulus field
        gamma: Regularization coefficient
        n_eigs: Number of smallest-eigenvalue eigenpairs to keep
        tol: Numerical threshold for positivity diagnostics

    Returns:
        diagnostics: Dictionary with separate data-only and regularized Hessian summaries
    """
    data_hessian, sensitivity_matrix = assemble_explicit_data_hessian(mesh_info, state)
    reg_matrix = np.asarray(mesh_info.get_regularization_matrix().toarray(), dtype=float)
    total_hessian = data_hessian + gamma * reg_matrix

    data_diagnostics = _summarize_data_spectrum(
        data_hessian,
        operator_name="J_U^T M J_U",
        n_eigs=n_eigs,
    )
    total_diagnostics = _summarize_total_spectrum(
        total_hessian,
        reg_matrix,
        gamma,
        operator_name="J_U^T M J_U + gamma G",
        n_eigs=n_eigs,
    )

    return {
        "space": "physical_modulus_E",
        "sensitivity_matrix": sensitivity_matrix,
        "data_hessian": data_diagnostics,
        "total_hessian": total_diagnostics,
    }
