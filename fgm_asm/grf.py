"""Gaussian random field generation for structured ASM meshes.

The MATLAB reference uses a squared-exponential RBF covariance matrix. On the
rectangular tensor-product mesh used by ASM, that covariance is separable, so
the same Gaussian field can be sampled from two one-dimensional Cholesky
factors without constructing a dense ``(nodes_x * nodes_y)^2`` matrix.
"""

from __future__ import annotations

import numpy as np


def _rbf_factor(coordinates: np.ndarray, ell: float, jitter: float) -> np.ndarray:
    """Return a stable factor for a one-dimensional RBF covariance matrix."""
    differences = coordinates[:, None] - coordinates[None, :]
    covariance = np.exp(-(differences * differences) / (2.0 * ell * ell))
    covariance = covariance + jitter * np.eye(coordinates.size)
    return np.linalg.cholesky(covariance)


def generate_grf_field(
    mesh_info,
    E_max: float = 8.0,
    sigma_g: float = 1.0,
    ell: float = 1.0,
    seed: int = 42,
    jitter: float = 1e-6,
) -> np.ndarray:
    """Generate one GRF modulus field on an ASM structured mesh.

    The MATLAB routine also supports batch dataset generation and shuffles a
    vector of ``E_max`` values. ASM forward workflows solve one case at a
    time, so this API deliberately keeps only the single-case part: one
    configured ``E_max`` and one reproducible Gaussian realization.
    """
    if E_max <= 0.0:
        raise ValueError(f"E_max must be positive, got {E_max}")
    if sigma_g < 0.0:
        raise ValueError(f"sigma_g must be non-negative, got {sigma_g}")
    if ell <= 0.0:
        raise ValueError(f"ell must be positive, got {ell}")
    if jitter <= 0.0:
        raise ValueError(f"jitter must be positive, got {jitter}")

    rng = np.random.RandomState(int(seed))

    x_coordinates = np.asarray(mesh_info.plot_x[0, :], dtype=float)
    y_coordinates = np.asarray(mesh_info.plot_y[:, 0], dtype=float)
    L_x = _rbf_factor(x_coordinates, float(ell), float(jitter))
    L_y = _rbf_factor(y_coordinates, float(ell), float(jitter))

    standard_normal = rng.standard_normal((mesh_info.nods_y, mesh_info.nods_x))
    gaussian_field = L_y @ standard_normal @ L_x.T
    normalized = (np.tanh(float(sigma_g) * gaussian_field) + 1.0) / 2.0
    return float(E_max) * normalized
