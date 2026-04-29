"""
Inverse problem solver for FGM modulus reconstruction.

This script loads displacement data from the forward problem and reconstructs
the modulus distribution using SciPy L-BFGS-B with Tikhonov
regularization.
"""

import time
import numpy as np

from fgm_asm import lbfgs_inverse_solver_scipy
from fgm_asm.results_io import get_noise_output_dir, save_inverse_results, write_python_config_snapshot
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
tensile_end_force = float(forward_data["tensile_end_force"])
forward_config = forward_data["config"]

print(f"  Loaded data from {forward_data_path}")
print(f"  Mesh: {mesh_info.nel_x} x {mesh_info.nel_y} elements")
print(f"  Number of nodes: {mesh_info.n_nod}")
print(f"  Saved tensile-end force: {tensile_end_force:.6e}")

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
        tensile_end_force=tensile_end_force,
        raw_init=None,
        gamma=inverse_config.gamma,
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
    save_inverse_results(
        results,
        errors,
        E_true,
        noise_level,
        noise_output_dir,
        extra_data={
            "n_iterations": results["n_iterations"],
            "elapsed_time_total_seconds": elapsed_time,
        },
    )
    config_snapshot_path = write_python_config_snapshot(
        noise_output_dir,
        [
            (
                "Run Metadata",
                {
                    "WORKFLOW": "inverse_main",
                    "FORWARD_DATA_PATH": str(forward_data_path),
                    "RESULTS_DIR": str(output_dir),
                    "NOISE_OUTPUT_DIR": str(noise_output_dir),
                    "NOISE_LEVEL": float(noise_level),
                    "TRUE_TENSILE_END_FORCE": float(tensile_end_force),
                    "RECONSTRUCTED_TENSILE_END_FORCE": float(results["alpha_final"] * results["force_unit_final"]),
                    "GAMMA_USED": float(inverse_config.gamma),
                    "RESULT_SOURCE": "fixed_gamma_inverse_main",
                    "CONVERGED": bool(results["converged"]),
                    "MESSAGE": str(results["message"]),
                    "N_ITERATIONS": int(results["n_iterations"]),
                    "ELAPSED_TIME_TOTAL_SECONDS": float(elapsed_time),
                },
            ),
            (
                "Forward Configuration",
                {
                    "GEO_L": forward_config.geo_l,
                    "GEO_H": forward_config.geo_h,
                    "NEL_X": forward_config.nel_x,
                    "NEL_Y": forward_config.nel_y,
                    "disp_amp": forward_config.disp_amp,
                    "EX": forward_config.Ex,
                    "EY": forward_config.Ey,
                    "NU": forward_config.nu,
                    "DIS_TYPE": forward_config.dis_type,
                },
            ),
            (
                "Inverse Configuration",
                {
                    "GAMMA": inverse_config.gamma,
                    "E_MIN": inverse_config.E_min,
                    "E_MAX": inverse_config.E_max,
                    "MAX_ITER": inverse_config.max_iter,
                    "FTOL": inverse_config.ftol,
                    "GTOL": inverse_config.gtol,
                    "NOISE_LEVELS": inverse_config.noise_levels,
                },
            ),
        ],
    )
    print(f"  Config snapshot saved to {config_snapshot_path}")

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
