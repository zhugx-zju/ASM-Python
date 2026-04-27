"""
Configuration file for FGM forward and inverse problem solvers.

This file contains all parameters for geometry, mesh, material properties,
and regularization settings.
"""

from fgm_asm.config_types import ForwardConfig, InverseConfig, LCurveConfig, normalize_noise_levels


# ============================================================
# Forward Problem Configuration
# ============================================================

# Geometry parameters
GEO_L = 9.0  # Domain length
GEO_H = 9.0  # Domain height

# Mesh parameters
NEL_X = 39  # Number of elements in x direction
NEL_Y = 39  # Number of elements in y direction

# Material properties
EX = 2.0        # Modulus ratio at x = geo_l
EY = 0.5        # Modulus ratio at y = geo_h
NU = 0.3        # Poisson's ratio

# Load condition
disp_amp = 0.001 * GEO_L     # Prescribed x-displacement on the right edge

# Distribution type: 'bil' (bilinear) or 'exp' (exponential)
DIS_TYPE = 'bil'


# ============================================================
# Inverse Problem Configuration
# ============================================================

# Regularization parameters
ALPHA = 0.0     # Will be computed from material distribution
BETA = 0.0      # Will be computed from material distribution
GAMMA = 1e-6    # Regularization coefficient (default, can be found by L-curve)

# Optimization parameters
E_MIN = 0.001       # Minimum modulus bound
E_MAX = 1000.0      # Maximum modulus bound
MAX_ITER = 2000     # Maximum iterations
FTOL = 1e-30         
GTOL = 1e-12

# Noise levels for testing
NOISE_LEVELS = 0.0  # 0%, 1%, 3%

# L-curve parameters
GAMMA_MIN = 1e-10    # Minimum gamma to test
GAMMA_MAX = 1e-8    # Maximum gamma to test
N_GAMMA = 50       # Number of gamma values to test


# ============================================================
# Helper Functions
# ============================================================

def get_forward_config():
    """Get the typed configuration object for the forward problem."""
    return ForwardConfig(
        geo_l=GEO_L,
        geo_h=GEO_H,
        nel_x=NEL_X,
        nel_y=NEL_Y,
        disp_amp=disp_amp,
        Ex=EX,
        Ey=EY,
        nu=NU,
        dis_type=DIS_TYPE,
    )


def get_inverse_config():
    """Get the typed configuration object for the inverse problem."""
    return InverseConfig(
        gamma=GAMMA,
        E_min=E_MIN,
        E_max=E_MAX,
        max_iter=MAX_ITER,
        ftol=FTOL,
        gtol=GTOL,
        noise_levels=normalize_noise_levels(NOISE_LEVELS),
        nu=NU,
    )


def get_lcurve_config():
    """Get the typed configuration object for the L-curve analysis."""
    return LCurveConfig(
        gamma_min=GAMMA_MIN,
        gamma_max=GAMMA_MAX,
        n_gamma=N_GAMMA,
        E_min=E_MIN,
        E_max=E_MAX,
        max_iter=MAX_ITER,
        ftol=FTOL,
        gtol=GTOL,
        nu=NU,
    )


def get_folder_name(alpha, beta, gamma):
    """
    Generate folder name similar to MATLAB sprintf format.

    Format: Geo_{GEO_L}x{GEO_H}_Mesh_{NEL_X}x{NEL_Y}_Alpha_{alpha:.4f}_Beta_{beta:.4f}_Gamma_{gamma:.0e}

    Args:
        alpha: Alpha parameter from material distribution
        beta: Beta parameter from material distribution
        gamma: Regularization parameter

    Returns:
        folder_name: Formatted folder name string
    """
    return get_forward_config().output_folder_name(alpha, beta, gamma)
