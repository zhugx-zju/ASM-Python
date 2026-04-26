"""
Python FGM (Functionally Graded Material) Solver Package

A finite element analysis package for:
1. Forward problem: Computing displacement for FGM plane stress problems
2. Inverse problem: Reconstructing modulus distribution from displacement data
   using SciPy L-BFGS-B with Tikhonov regularization
"""

from .mesh import MeshInfo, shape_fun_at_gauss, get_body_force_load
from .material import MaterialInfo, generate_fgm_modulus
from .fem_forward import fem_assemble, forward_solver, compute_reaction_forces
from .regularization import get_tikhonov_regularization, get_tikhonov_gradient
from .inverse_solver import (
    lbfgs_inverse_solver_scipy,
    lambda_distance,
    get_stiffness_gradient
)
from .l_curve import (
    find_optimal_gamma_lcurve,
    plot_lcurve_results
)
from .utils import (
    add_noise_to_displacement,
    compute_errors,
    save_results
)
from .visualization import (
    plot_displacement_fields,
    plot_gradient_field,
    plot_modulus_distribution,
    plot_reconstruction_comparison,
    plot_reconstruction_results,
    plot_single_displacement_field,
    plot_iteration_history,
    visualize_forward_results,
    visualize_inverse_results
)

__version__ = '1.0.0'
__all__ = [
    # Mesh
    'MeshInfo',
    'shape_fun_at_gauss',
    'get_body_force_load',
    # Material
    'MaterialInfo',
    'generate_fgm_modulus',
    # Forward solver
    'fem_assemble',
    'forward_solver',
    'compute_reaction_forces',
    # Regularization
    'get_tikhonov_regularization',
    'get_tikhonov_gradient',
    # Inverse solver
    'lbfgs_inverse_solver_scipy',
    'lambda_distance',
    'get_stiffness_gradient',
    # L-curve
    'find_optimal_gamma_lcurve',
    'plot_lcurve_results',
    # Utils
    'add_noise_to_displacement',
    'compute_errors',
    'save_results',
    # Visualization
    'plot_displacement_fields',
    'plot_gradient_field',
    'plot_modulus_distribution',
    'plot_reconstruction_comparison',
    'plot_reconstruction_results',
    'plot_single_displacement_field',
    'plot_iteration_history',
    'visualize_forward_results',
    'visualize_inverse_results',
]

