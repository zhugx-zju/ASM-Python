"""
Utility functions for FGM solvers.
"""

import numpy as np
from .results_io import format_noise_tag, get_noise_output_dir, save_inverse_results as save_results


def add_noise_to_displacement(U, noise_level=0.00, seed=42):
    """
    Add Gaussian noise to displacement data.

    Args:
        U: Clean displacement vector [n_dof]
        noise_level: Noise level as fraction of signal magnitude
        seed: Random seed for reproducibility

    Returns:
        U_noisy: Noisy displacement vector
    """
    if noise_level > 0:
        np.random.seed(seed)
        noise_amp = noise_level * np.abs(U)
        U_noisy = U + noise_amp * np.random.randn(len(U))
    else:
        U_noisy = U.copy()

    return U_noisy


def compute_errors(E_true, E_reconstructed):
    """
    Compute error metrics between true and reconstructed modulus.

    Args:
        E_true: True modulus field
        E_reconstructed: Reconstructed modulus field

    Returns:
        errors: Dictionary with error metrics
    """
    rel_error = 100 * np.abs(E_true - E_reconstructed) / E_true
    mae = np.mean(rel_error)
    max_err = np.max(rel_error)
    rmse = np.sqrt(np.mean((E_true - E_reconstructed)**2))

    errors = {
        'mae': mae,
        'max_error': max_err,
        'rmse': rmse,
        'rel_error_field': rel_error
    }

    return errors
