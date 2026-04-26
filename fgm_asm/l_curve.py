"""
L-curve method for optimal regularization parameter selection.

Implements the L-curve criterion for finding the optimal regularization
parameter in inverse problems using the maximum curvature method.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from .config_types import coerce_forward_config
from .inverse_solver import lbfgs_inverse_solver_scipy


def find_optimal_gamma_lcurve(mesh_info, bc_info, U_measured, config,
                               gamma_min=1e-10, gamma_max=1e-4, n_gamma=15,
                               E_min=0.001, E_max=1000.0, max_iter=2000, ftol=1e-12, gtol=1e-8):
    """
    Use L-curve method to find optimal regularization parameter.

    This function tests a range of regularization parameters (gamma values)
    and uses the L-curve criterion to automatically select the optimal one.
    The L-curve is a log-log plot of the residual norm versus the regularization
    norm. The optimal parameter is at the corner (maximum curvature) of this curve.

    Implementation Details:
        - Gamma values are tested from large to small (strong to weak regularization)
        - Uses warm-start continuation: each gamma uses the previous solution as initial guess
        - Convergence is controlled primarily by gtol for consistency across gamma values
        - ftol is set very small to avoid premature stopping

    Parameters
    ----------
    mesh_info : MeshInfo
        Mesh information object
    bc_info : dict
        Boundary condition information
    U_measured : ndarray
        Measured displacement data
    config : ForwardConfig or dict
        Forward configuration containing the Poisson ratio
    gamma_min : float
        Minimum gamma value to test
    gamma_max : float
        Maximum gamma value to test
    n_gamma : int
        Number of gamma values to test
    E_min : float
        Minimum modulus bound
    E_max : float
        Maximum modulus bound
    max_iter : int
        Maximum iterations for optimization
    ftol : float
        Function tolerance (default: 1e-12, set small to rely on gtol)
    gtol : float
        Gradient tolerance (default: 1e-8, primary convergence criterion)

    Returns
    -------
    gamma_optimal : float
        Optimal regularization parameter
    results_dict : dict
        Dictionary containing all results for each gamma:
        - gamma_values: Array of tested gamma values
        - residual_norms: Residual norms for each gamma
        - regularization_norms: Regularization norms for each gamma
        - curvature: Curvature values at each point
        - optimal_idx: Index of optimal gamma
        - gamma_optimal: Optimal gamma value
        - E_solutions: List of reconstructed modulus distributions
        - all_results: List of optimization results for each gamma
    """
    print("\n" + "="*70)
    print("L-curve Analysis for Regularization Parameter Selection")
    print("="*70)
    print(f"Testing gamma range: [{gamma_min:.2e}, {gamma_max:.2e}]")
    print(f"Number of test points: {n_gamma}")
    print("="*70)
    forward_config = coerce_forward_config(config)

    # Generate gamma values on log scale (from large to small for stability)
    gamma_values = np.logspace(np.log10(gamma_max), np.log10(gamma_min), n_gamma)

    # Storage for L-curve
    residual_norms = []
    regularization_norms = []
    E_solutions = []
    all_results = []

    # Initialize E_current for warm-start continuation
    E_current = None  # Will be set to E_init or previous solution

    # Test each gamma value with warm-start continuation
    for i, gamma in enumerate(gamma_values):
        print(f"\n[{i+1}/{n_gamma}] Testing gamma = {gamma:.6e}")

        if i == 0:
            print("  Starting from initial guess (first gamma)")
        else:
            print("  Using warm-start from previous gamma solution")

        # Solve inverse problem with current gamma
        results = lbfgs_inverse_solver_scipy(
            mesh_info=mesh_info,
            bc_info=bc_info,
            U_measured=U_measured,
            E_init=E_current,  # Use previous solution as initial guess
            gamma=gamma,
            E_min=E_min,
            E_max=E_max,
            max_iter=max_iter,
            ftol=ftol,
            gtol=gtol,
            nu=forward_config.nu
        )

        # Update E_current for next iteration (warm-start)
        E_current = results['E_final'].copy()

        E_opt = results['E_final']
        E_solutions.append(E_opt)
        all_results.append(results)

        residual_norm = results['residual_norm']
        reg_norm = results['regularization_norm']
        residual_norms.append(residual_norm)
        regularization_norms.append(reg_norm)

        print(f"  Residual norm: {residual_norm:.6e}")
        print(f"  Regularization norm: {reg_norm:.6e}")
        print(f"  Converged: {results['converged']}, Iterations: {results['n_iterations']}")

    # Convert to arrays
    residual_norms = np.array(residual_norms)
    regularization_norms = np.array(regularization_norms)

    # Find L-curve corner using maximum curvature (Menger curvature method)
    print("\n" + "="*70)
    print("Computing L-curve curvature...")
    print("="*70)

    # Curvature in log-log space
    eps = 1e-30  # Prevent log(0)
    x = np.log10(np.maximum(residual_norms, eps))
    y = np.log10(np.maximum(regularization_norms, eps))

    # Initialize curvature array
    curvature = np.zeros(n_gamma)

    # Compute Menger curvature for interior points
    for i in range(1, n_gamma - 1):
        p1 = np.array([x[i-1], y[i-1]])
        p2 = np.array([x[i],   y[i]])
        p3 = np.array([x[i+1], y[i+1]])

        # Compute distances
        d12 = np.linalg.norm(p2 - p1)
        d23 = np.linalg.norm(p3 - p2)
        d31 = np.linalg.norm(p1 - p3)
        denom = d12 * d23 * d31

        if denom <= 0:
            curvature[i] = 0.0
            continue

        # Menger curvature: κ = 4*Area / (d12 * d23 * d31)
        area2 = abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
        curvature[i] = 2.0 * area2 / denom

    # Print curvature values
    print("\nCurvature values:")
    for i in range(n_gamma):
        print(f" Gamma={gamma_values[i]:.2e}: {curvature[i]:.6e}")

    # Find optimal gamma using maximum curvature
    k = 2  # Exclude k points at each end to avoid boundary effects
    optimal_idx = np.argmax(curvature[k:-k]) + k
    gamma_optimal = gamma_values[optimal_idx]

    print("\n" + "="*70)
    print(f"Optimal gamma selected by L-curve: {gamma_optimal:.6e}")
    print(f"  Index: {optimal_idx} (excluded {k} points at each end)")
    print(f"  Residual norm: {residual_norms[optimal_idx]:.6e}")
    print(f"  Regularization norm: {regularization_norms[optimal_idx]:.6e}")
    print(f"  Curvature: {curvature[optimal_idx]:.6e}")
    print("="*70)

    # Package results
    results_dict = {
        'gamma_values': gamma_values,
        'residual_norms': residual_norms,
        'regularization_norms': regularization_norms,
        'curvature': curvature,
        'optimal_idx': optimal_idx,
        'gamma_optimal': gamma_optimal,
        'E_solutions': E_solutions,
        'all_results': all_results
    }

    return gamma_optimal, results_dict


def plot_lcurve_results(results_dict, save_path=None):
    """
    Plot L-curve and curvature analysis results in two separate figures.

    Creates two separate figures:
    1. Figure 1: The L-curve (residual norm vs regularization norm) in log-log scale
    2. Figure 2: Curvature as a function of the regularization parameter

    The plots are displayed with gamma values sorted from small to large for clarity,
    even though the optimization may have been performed in reverse order.

    Parameters
    ----------
    results_dict : dict
        Results from find_optimal_gamma_lcurve
    save_path : Path or None
        Directory to save plots. If None, plots are not saved.

    Returns
    -------
    fig_lcurve : matplotlib.figure.Figure
        The L-curve figure
    fig_curvature : matplotlib.figure.Figure
        The curvature figure
    """

    # Set font to Times New Roman
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['mathtext.fontset'] = 'custom'
    rcParams['mathtext.rm'] = 'Times New Roman'

    # Extract data
    gamma_values = results_dict['gamma_values']
    residual_norms = results_dict['residual_norms']
    regularization_norms = results_dict['regularization_norms']
    curvature = results_dict['curvature']
    optimal_idx = results_dict['optimal_idx']
    gamma_optimal = results_dict['gamma_optimal']

    # Sort by gamma (small to large) for more intuitive visualization
    order = np.argsort(gamma_values)
    gv_sorted = gamma_values[order]
    rn_sorted = residual_norms[order]
    regn_sorted = regularization_norms[order]
    curv_sorted = curvature[order]

    # Find where optimal_idx maps to in sorted array
    opt_sorted_idx = np.where(order == optimal_idx)[0][0]

    # ============================================================
    # Figure 1: L-curve (using norms, not squared norms)
    # ============================================================
    fig_lcurve = plt.figure(figsize=(8, 6))
    ax1 = plt.subplot(1, 1, 1)

    # Plot in log-log scale using norms (classical L-curve definition)
    ax1.loglog(rn_sorted, regn_sorted, 'b-o',
             linewidth=2.5, markersize=8, label='L-curve', alpha=0.7)
    ax1.loglog(rn_sorted[opt_sorted_idx], regn_sorted[opt_sorted_idx],
             'r*', markersize=20, label=f'OptimalGamma = {gamma_optimal:.2e}', zorder=10)

    # Axis labels (without squared notation)
    ax1.set_xlabel(r'Data mismatch $||u - u_{obs}||_2$', fontsize=13)
    ax1.set_ylabel(r'Regularization $||\nabla E||_2$', fontsize=13)
    ax1.set_title('L-curve', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.tick_params(labelsize=11)
    ax1.grid(True, which='both', alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        fig_lcurve.savefig(save_path / 'lcurve.png', dpi=300, bbox_inches='tight')
        fig_lcurve.savefig(save_path / 'lcurve.pdf', dpi=300, bbox_inches='tight')
        print(f"  L-curve plot saved to {save_path / 'lcurve.png'}")

    # ============================================================
    # Figure 2: Curvature vs gamma (sorted by gamma)
    # ============================================================
    fig_curvature = plt.figure(figsize=(8, 6))
    ax2 = plt.subplot(1, 1, 1)

    # Plot curvature vs gamma (ignoring NaN for visual clarity)
    valid_mask = np.isfinite(curv_sorted)
    ax2.plot(gv_sorted[valid_mask], curv_sorted[valid_mask], 'g-o',
             linewidth=2.5, markersize=8, label='Curvature')

    # Mark optimal point
    if np.isfinite(curv_sorted[opt_sorted_idx]):
        ax2.plot(gamma_optimal, curv_sorted[opt_sorted_idx],
                 'r*', markersize=20, label=f'OptimalGamma = {gamma_optimal:.2e}', zorder=10)

    ax2.set_xlabel(r'Regularization Parameter $\gamma$', fontsize=13)
    ax2.set_ylabel('Curvature', fontsize=13)
    ax2.set_title(r'Curvature vs $\gamma$', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(fontsize=11, loc='best')
    ax2.tick_params(labelsize=11)
    ax2.grid(True, which='both', alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig_curvature.savefig(save_path / 'curvature_vs_gamma.png', dpi=300, bbox_inches='tight')
        fig_curvature.savefig(save_path / 'curvature_vs_gamma.pdf', dpi=300, bbox_inches='tight')
        print(f"  Curvature plot saved to {save_path / 'curvature_vs_gamma.png'}")

    return fig_lcurve, fig_curvature
