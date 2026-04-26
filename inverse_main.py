"""
Inverse problem solver for FGM modulus reconstruction.

This script loads displacement data from the forward problem and reconstructs
the modulus distribution using SciPy L-BFGS-B with Tikhonov
regularization.
"""

import time
import numpy as np

from fgm_asm import lbfgs_inverse_solver_scipy
from fgm_asm.results_io import get_noise_output_dir, save_inverse_results
from fgm_asm.utils import add_noise_to_displacement, compute_errors
from fgm_asm.visualization import visualize_inverse_results
from fgm_asm.workflows import load_latest_forward_problem, resolve_results_dir
import config as cfg


print("=" * 70)
print("Inverse Problem Solver for FGM Modulus Reconstruction")
print("SciPy L-BFGS-B with Tikhonov Regularization")
print("=" * 70)

print("\nSearching for forward problem data...")
forward_data_path, forward_data = load_latest_forward_problem()

mesh_info = forward_data["mesh_info"]
bc_info = forward_data["bc_info"]
U_clean = forward_data["U"]
E_true = forward_data["E_field"]
forward_config = forward_data["config"]

print(f"  Loaded data from {forward_data_path}")
print(f"  Mesh: {mesh_info.nel_x} x {mesh_info.nel_y} elements")
print(f"  Number of nodes: {mesh_info.n_nod}")

inverse_config = cfg.get_inverse_config()

print(f"\nInverse Problem Parameters:")
print(f"  Regularization coefficient: {inverse_config.gamma:.2e}")
print(f"  Modulus bounds: [{inverse_config.E_min}, {inverse_config.E_max}]")
print(f"  Max iterations: {inverse_config.max_iter}")
print(f"  ftol: {inverse_config.ftol:.2e}")
print(f"  gtol: {inverse_config.gtol:.2e}")

output_dir = resolve_results_dir(forward_data_path, forward_data)
print(f"\nResults will be saved to: {output_dir}")

print("\nAssembling mass matrix...")
mesh_info.assemble_mass_matrix()

for noise_level in np.asarray(inverse_config.noise_levels, dtype=float):
    print("\n" + "=" * 70)
    print(f"Processing Noise Level: {noise_level*100:.2f}%")
    print("=" * 70)

    U_measured = add_noise_to_displacement(U_clean, noise_level)
    print(f"  Displacement noise added: {noise_level*100:.2f}%")

    print(f"\nStarting L-BFGS-B optimization...")
    start_time = time.time()

    results = lbfgs_inverse_solver_scipy(
        mesh_info=mesh_info,
        bc_info=bc_info,
        U_measured=U_measured,
        E_init=None,
        gamma=inverse_config.gamma,
        E_min=inverse_config.E_min,
        E_max=inverse_config.E_max,
        max_iter=inverse_config.max_iter,
        ftol=inverse_config.ftol,
        gtol=inverse_config.gtol,
        nu=forward_config.nu,
    )

    elapsed_time = time.time() - start_time
    E_reconstructed = results["E_final"]
    errors = compute_errors(E_true.ravel(), E_reconstructed)

    print(f"\n{'='*70}")
    print(f"Results for Noise Level {noise_level*100:.2f}%:")
    print(f"{'='*70}")
    print(f"  Converged: {results['converged']}")
    print(f"  Iterations: {results['n_iterations']}")
    print(f"  Elapsed time: {elapsed_time:.2f} seconds")
    print(f"  Final cost: {results['cost_history'][-1]:.6e}")
    print(f"  MAE: {errors['mae']:.4f}%")
    print(f"  Max error: {errors['max_error']:.4f}%")
    print(f"  RMSE: {errors['rmse']:.4f}")

    noise_output_dir = get_noise_output_dir(output_dir, noise_level)
    print(f"\nSaving results to {noise_output_dir}...")
    save_inverse_results(results, errors, E_true, noise_level, noise_output_dir)

    print("Generating visualizations...")
    visualize_inverse_results(
        mesh_info,
        E_true,
        E_reconstructed,
        errors,
        results,
        noise_level,
        save_path=noise_output_dir,
    )

print("\n" + "=" * 70)
print("All inverse problems solved successfully!")
print(f"Results saved to {output_dir}")
print("=" * 70)
