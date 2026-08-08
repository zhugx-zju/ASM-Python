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
    num: int = 1,
    E_max: float = 8.0,
    sigma_g: float = 1.0,
    ell: float = 1.0,
    seed_max: int = 42,
    jitter: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate GRF modulus samples using the MATLAB RBF construction.

    Returns ``(E_field, E_max_vec)`` with field shape
    ``[num, nodes_y, nodes_x]``. The same seed controls the shuffled
    ``E_max`` sequence and the Gaussian samples, as in the MATLAB routine.
    """
    if int(num) != num or num < 1:
        raise ValueError(f"num must be a positive integer, got {num!r}")
    if E_max <= 0.0:
        raise ValueError(f"E_max must be positive, got {E_max}")
    if sigma_g < 0.0:
        raise ValueError(f"sigma_g must be non-negative, got {sigma_g}")
    if ell <= 0.0:
        raise ValueError(f"ell must be positive, got {ell}")
    if jitter <= 0.0:
        raise ValueError(f"jitter must be positive, got {jitter}")

    num = int(num)
    rng = np.random.RandomState(int(seed_max))
    # A single forward sample should use the configured maximum directly.
    # For multiple samples, retain the MATLAB linspace-and-shuffle behavior.
    E_max_vec = np.array([float(E_max)]) if num == 1 else np.linspace(1.0, float(E_max), num)
    rng.shuffle(E_max_vec)

    x_coordinates = np.asarray(mesh_info.plot_x[0, :], dtype=float)
    y_coordinates = np.asarray(mesh_info.plot_y[:, 0], dtype=float)
    L_x = _rbf_factor(x_coordinates, float(ell), float(jitter))
    L_y = _rbf_factor(y_coordinates, float(ell), float(jitter))

    fields = np.empty((num, mesh_info.nods_y, mesh_info.nods_x), dtype=float)
    for index, sample_max in enumerate(E_max_vec):
        standard_normal = rng.standard_normal((mesh_info.nods_y, mesh_info.nods_x))
        gaussian_field = L_y @ standard_normal @ L_x.T
        normalized = (np.tanh(float(sigma_g) * gaussian_field) + 1.0) / 2.0
        fields[index] = float(sample_max) * normalized

    return fields, E_max_vec
