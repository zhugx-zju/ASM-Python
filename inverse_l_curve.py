"""
Inverse problem solver for FGM modulus reconstruction using scipy L-BFGS-B.

This script uses the L-curve method to select the regularization parameter and
then reruns the inverse solve from the default initialization using the
selected gamma.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from fgm_asm import find_optimal_gamma_lcurve, plot_lcurve_results, lbfgs_inverse_solver_scipy
from fgm_asm.visualization import (
    plot_gradient_field,
    plot_hessian_spectrum,
    plot_iteration_history,
    plot_reconstruction_comparison,
    plot_reconstruction_results,
)
from fgm_asm.results_io import (
    get_noise_output_dir,
    save_inverse_results,
    save_lcurve_analysis,
    write_python_config_snapshot,
)
from fgm_asm.utils import add_noise_to_displacement, compute_errors
from fgm_asm.workflows import load_latest_forward_problem, resolve_results_dir
import config as cfg


def _print_hessian_summary(label, diagnostics):
    """Print a compact Hessian spectral summary."""
    if diagnostics is None:
        return
    print(f"  {label}:")
    print(f"    lambda_min: {diagnostics['lambda_min']:.6e}")
    print(f"    lambda_max: {diagnostics['lambda_max']:.6e}")
    print(f"    condition number: {diagnostics['condition_number']:.6e}")
    print(f"    positive definite: {diagnostics['is_positive_definite']}")
    print(f"    near-nullspace detected: {diagnostics['near_nullspace_detected']}")
    print(f"    negative curvature detected: {diagnostics['has_negative_curvature']}")


def _diagnostics_snapshot_entries(prefix, diagnostics):
    """Serialize Hessian diagnostics into config-snapshot-friendly literals."""
    def _snapshot_number(value):
        value = float(value)
        if np.isfinite(value):
            return value
        if np.isnan(value):
            return "nan"
        return "inf" if value > 0.0 else "-inf"

    if diagnostics is None:
        return {
            f"{prefix}_OPERATOR": None,
            f"{prefix}_LAMBDA_MIN": None,
            f"{prefix}_LAMBDA_MAX": None,
            f"{prefix}_CONDITION_NUMBER": None,
            f"{prefix}_IS_POSITIVE_DEFINITE": None,
            f"{prefix}_NEAR_NULLSPACE_DETECTED": None,
            f"{prefix}_NEGATIVE_CURVATURE_DETECTED": None,
            f"{prefix}_SMALLEST_EIGENVALUES": None,
            f"{prefix}_SMALLEST_EIGENPAIR_SUMMARIES": None,
            f"{prefix}_SMALLEST_EIGENVECTORS": None,
        }

    eigenpair_summaries = diagnostics.get("smallest_eigenpair_summaries", [])
    serialized_summaries = [
        {
            "mode_index": int(mode["mode_index"]),
            "eigenvalue": float(mode["eigenvalue"]),
            "data_energy": float(mode["data_energy"]),
            "regularization_energy": (
                None if mode.get("regularization_energy") is None else float(mode["regularization_energy"])
            ),
            "total_energy": None if mode.get("total_energy") is None else float(mode["total_energy"]),
            "min_component": float(mode["min_component"]),
            "max_component": float(mode["max_component"]),
            "mean_component": float(mode["mean_component"]),
        }
        for mode in eigenpair_summaries
    ]
    vectors = np.asarray(diagnostics.get("smallest_eigenvectors"), dtype=float)

    return {
        f"{prefix}_OPERATOR": str(diagnostics["operator_name"]),
        f"{prefix}_LAMBDA_MIN": _snapshot_number(diagnostics["lambda_min"]),
        f"{prefix}_LAMBDA_MAX": _snapshot_number(diagnostics["lambda_max"]),
        f"{prefix}_CONDITION_NUMBER": _snapshot_number(diagnostics["condition_number"]),
        f"{prefix}_IS_POSITIVE_DEFINITE": bool(diagnostics["is_positive_definite"]),
        f"{prefix}_NEAR_NULLSPACE_DETECTED": bool(diagnostics["near_nullspace_detected"]),
        f"{prefix}_NEGATIVE_CURVATURE_DETECTED": bool(diagnostics["has_negative_curvature"]),
        f"{prefix}_SMALLEST_EIGENVALUES": np.asarray(diagnostics["smallest_eigenvalues"], dtype=float).tolist(),
        f"{prefix}_SMALLEST_EIGENPAIR_SUMMARIES": serialized_summaries,
        f"{prefix}_SMALLEST_EIGENVECTORS": vectors.T.tolist(),
    }


lcurve_config = cfg.get_lcurve_config()
inverse_config = cfg.get_inverse_config()
noise_level = inverse_config.primary_noise_level

print("=" * 70)
print("Inverse Problem Solver with L-curve Analysis")
print("Using scipy L-BFGS-B Optimizer with Tikhonov Regularization")
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

print(f"\nParameters:")
print(f"  Noise level: {noise_level*100:.2f}%")
print(f"  Modulus bounds: [{lcurve_config.E_min}, {lcurve_config.E_max}]")
print(f"  Max iterations: {lcurve_config.max_iter}")
print(f"  ftol: {lcurve_config.ftol:.2e}, gtol: {lcurve_config.gtol:.2e}")
print(f"  Hessian analysis: {lcurve_config.enable_hessian_analysis}")

output_dir = resolve_results_dir(forward_data_path, forward_data)
noise_output_dir = get_noise_output_dir(output_dir, noise_level)
print(f"\nResults will be saved to: {output_dir}")
print(f"Noise-specific output directory: {noise_output_dir}")

print("\nAssembling mass matrix...")
mesh_info.assemble_mass_matrix()

print(f"\nAdding {noise_level*100:.2f}% noise to displacement data...")
U_measured = add_noise_to_displacement(U_clean, noise_level)

start_time_lcurve = time.time()
gamma_optimal, lcurve_results = find_optimal_gamma_lcurve(
    mesh_info=mesh_info,
    bc_info=bc_info,
    U_measured=U_measured,
    tensile_end_force=tensile_end_force,
    config=forward_config,
    gamma_min=lcurve_config.gamma_min,
    gamma_max=lcurve_config.gamma_max,
    n_gamma=lcurve_config.n_gamma,
    max_iter=lcurve_config.max_iter,
    ftol=lcurve_config.ftol,
    gtol=lcurve_config.gtol,
    enable_hessian_analysis=False,
    hessian_n_eigs=lcurve_config.hessian_n_eigs,
    analysis_tol=lcurve_config.analysis_tol,
)
elapsed_time_lcurve = time.time() - start_time_lcurve
print(f"\nL-curve analysis completed in {elapsed_time_lcurve:.2f} seconds")
print("  Reduced-Hessian diagnostics were skipped during the gamma scan.")

optimal_idx = lcurve_results["optimal_idx"]
scan_results = lcurve_results["all_results"][optimal_idx]
scan_E_reconstructed = lcurve_results["E_solutions"][optimal_idx]
scan_errors = compute_errors(E_true.ravel(), scan_E_reconstructed)

print("\n" + "=" * 70)
print("Re-running inverse solve with selected gamma from default initialization")
print("=" * 70)

start_time_rerun = time.time()
results = lbfgs_inverse_solver_scipy(
    mesh_info=mesh_info,
    bc_info=bc_info,
    U_measured=U_measured,
    tensile_end_force=tensile_end_force,
    raw_init=None,
    gamma=gamma_optimal,
    max_iter=lcurve_config.max_iter,
    ftol=lcurve_config.ftol,
    gtol=lcurve_config.gtol,
    nu=forward_config.nu,
    enable_hessian_analysis=lcurve_config.enable_hessian_analysis,
    hessian_n_eigs=lcurve_config.hessian_n_eigs,
    analysis_tol=lcurve_config.analysis_tol,
)
elapsed_time_rerun = time.time() - start_time_rerun
E_reconstructed = results["E_final"]
errors = compute_errors(E_true.ravel(), E_reconstructed)

comparison_summary = {
    "scan_mae": scan_errors["mae"],
    "rerun_mae": errors["mae"],
    "scan_rmse": scan_errors["rmse"],
    "rerun_rmse": errors["rmse"],
    "modulus_diff_l2": float(np.linalg.norm(E_reconstructed - scan_E_reconstructed)),
    "modulus_diff_rel_l2": float(
        np.linalg.norm(E_reconstructed - scan_E_reconstructed) /
        (np.linalg.norm(scan_E_reconstructed) + 1e-15)
    ),
}

print(f"\n{'='*70}")
print(f"Final Results with Optimal Gamma = {gamma_optimal:.6e}")
print(f"{'='*70}")
print(f"  Converged: {results['converged']}")
print(f"  Iterations: {results['n_iterations']}")
print(f"  Final cost: {results['final_cost']:.6e}")
print(f"  Message: {results['message']}")
data_diagnostics = results.get("final_data_hessian_diagnostics")
diagnostics = results.get("final_hessian_diagnostics")
_print_hessian_summary("Data Hessian J_E^T M J_E", data_diagnostics)
_print_hessian_summary("Regularized Hessian J_E^T M J_E + gamma G", diagnostics)
print(f"\nReconstruction Errors:")
print(f"  MAE: {errors['mae']:.4f}%")
print(f"  Max error: {errors['max_error']:.4f}%")
print(f"  RMSE: {errors['rmse']:.4f}")
print(f"\nScan-optimum vs final rerun:")
print(f"  Scan MAE: {scan_errors['mae']:.4f}%")
print(f"  Rerun MAE: {errors['mae']:.4f}%")
print(f"  Relative L2 difference: {comparison_summary['modulus_diff_rel_l2']:.6e}")
print(f"\nTiming:")
print(f"  L-curve scan: {elapsed_time_lcurve:.2f} seconds")
print(f"  Final rerun: {elapsed_time_rerun:.2f} seconds")
print(f"  Total: {elapsed_time_lcurve + elapsed_time_rerun:.2f} seconds")

print(f"\nSaving results to {noise_output_dir}...")
lcurve_save_path = save_lcurve_analysis(
    lcurve_results,
    noise_output_dir,
    extra_data={
        "noise_level": noise_level,
        "noise_output_dir": str(noise_output_dir),
        "selection_method": "lcurve_max_curvature",
        "gamma_optimal": gamma_optimal,
        "optimal_idx": optimal_idx,
    },
)
print(f"  L-curve analysis saved to {lcurve_save_path}")

save_inverse_results(
    results,
    errors,
    E_true,
    noise_level,
    noise_output_dir,
    extra_data={
        "gamma_used": gamma_optimal,
        "n_iterations": results["n_iterations"],
        "elapsed_time_total_seconds": elapsed_time_rerun,
        "result_source": "final_rerun_after_lcurve",
        "gamma_selection_method": "lcurve_max_curvature",
        "lcurve_analysis_file": lcurve_save_path.name,
        "scan_optimal": {
            "gamma_used": gamma_optimal,
            "optimal_idx": optimal_idx,
            "result_source": "lcurve_scan_optimal",
            "E_reconstructed": scan_E_reconstructed,
            "errors": scan_errors,
            "results": scan_results,
        },
        "comparison_summary": comparison_summary,
    },
)
config_snapshot_path = write_python_config_snapshot(
    noise_output_dir,
    [
        (
            "Run Metadata",
            {
                "WORKFLOW": "inverse_l_curve",
                "FORWARD_DATA_PATH": str(forward_data_path),
                "RESULTS_DIR": str(output_dir),
                "NOISE_OUTPUT_DIR": str(noise_output_dir),
                "NOISE_LEVEL": float(noise_level),
                "TRUE_TENSILE_END_FORCE": float(tensile_end_force),
                "RECONSTRUCTED_TENSILE_END_FORCE": float(results["alpha_final"] * results["force_unit_final"]),
                "GAMMA_USED": float(gamma_optimal),
                "RESULT_SOURCE": "final_rerun_after_lcurve",
                "SELECTION_METHOD": "lcurve_max_curvature",
                "OPTIMAL_IDX": int(optimal_idx),
                "CONVERGED": bool(results["converged"]),
                "MESSAGE": str(results["message"]),
                "N_ITERATIONS": int(results["n_iterations"]),
                "ELAPSED_TIME_TOTAL_SECONDS": float(elapsed_time_rerun),
                "LCURVE_ANALYSIS_FILE": lcurve_save_path.name,
                "LCURVE_SCAN_TIME_SECONDS": float(elapsed_time_lcurve),
                "SCAN_OPTIMAL_MAE_PCT": float(scan_errors["mae"]),
                "FINAL_RERUN_MAE_PCT": float(errors["mae"]),
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
                "ENABLE_HESSIAN_ANALYSIS": inverse_config.enable_hessian_analysis,
                "HESSIAN_N_EIGS": inverse_config.hessian_n_eigs,
                "ANALYSIS_TOL": inverse_config.analysis_tol,
            },
        ),
        (
            "L-curve Configuration",
            {
                "GAMMA_MIN": lcurve_config.gamma_min,
                "GAMMA_MAX": lcurve_config.gamma_max,
                "N_GAMMA": lcurve_config.n_gamma,
                "ENABLE_HESSIAN_ANALYSIS": lcurve_config.enable_hessian_analysis,
                "HESSIAN_N_EIGS": lcurve_config.hessian_n_eigs,
                "ANALYSIS_TOL": lcurve_config.analysis_tol,
            },
        ),
        (
            "Ill-Posedness Diagnostics",
            {
                **_diagnostics_snapshot_entries("DATA_HESSIAN", data_diagnostics),
                **_diagnostics_snapshot_entries("TOTAL_HESSIAN", diagnostics),
            },
        ),
    ],
)
print(f"  Config snapshot saved to {config_snapshot_path}")

print(f"\nGenerating visualizations...")
print("  Plotting L-curve analysis...")
plot_lcurve_results(lcurve_results, save_path=noise_output_dir)

print("  Plotting final rerun reconstruction results...")
plot_reconstruction_results(
    mesh_info,
    E_true,
    E_reconstructed,
    errors,
    noise_level,
    save_path=noise_output_dir,
    filename_stem="reconstruction_results",
)
plot_iteration_history(
    results,
    save_path=noise_output_dir,
    noise_level=noise_level,
    filename_stem="iteration_history",
)
plot_gradient_field(
    mesh_info,
    results,
    noise_level=noise_level,
    save_path=noise_output_dir,
    filename_stem="gradient_field",
)
plot_hessian_spectrum(
    results,
    save_path=noise_output_dir,
    filename_stem="hessian_spectrum",
)

print("  Plotting scan-vs-rerun comparison...")
plot_reconstruction_comparison(
    mesh_info,
    E_true,
    scan_E_reconstructed,
    scan_errors,
    E_reconstructed,
    errors,
    noise_level,
    save_path=noise_output_dir,
)

print("\n" + "=" * 70)
print("Inverse problem solved successfully with L-curve analysis!")
print(f"Results saved to {noise_output_dir}")
print(f"Optimal gamma: {gamma_optimal:.6e}")
print("=" * 70)

if 'agg' in plt.get_backend().lower():
    plt.close('all')
else:
    plt.show()
