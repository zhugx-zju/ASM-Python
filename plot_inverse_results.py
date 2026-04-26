"""
Plot inverse problem results from saved data.

This script reads the inverse problem results from the folder specified in config.py
and creates up to four separate figures:
1. Figure 1: Reconstruction results (True, Reconstructed, Error fields in 1 row, 3 columns)
2. Figure 2: Iteration history (Convergence, Gradient norm, Relative gradient norm in 1 row, 3 columns)
3. Figure 3: Gradient field (Total gradient, Data misfit gradient, Regularization gradient in 1 row, 3 columns)
4. Figure 4: Comparison between the L-curve scan optimum and the final rerun
"""

import matplotlib.pyplot as plt
from fgm_asm.visualization import (
    plot_reconstruction_comparison,
    plot_reconstruction_results,
    plot_iteration_history,
    plot_gradient_field
)
from fgm_asm.results_io import find_inverse_results_path, load_inverse_data


# ============================================================
# Main execution
# ============================================================
print("=" * 70)
print("Inverse Results Plotter")
print("=" * 70)

# Search for results data
print("\nSearching for results data...")
inverse_results_path, results_folder = find_inverse_results_path()

# Load data
forward_data, inverse_results = load_inverse_data(inverse_results_path)
print(f"  Loaded inverse results from {inverse_results_path}")

mesh_info = forward_data['mesh_info']
E_true = forward_data['E_field']

# Extract inverse results
# For backward compatibility, check both possible locations
if 'E_reconstructed' in inverse_results:
    E_reconstructed = inverse_results['E_reconstructed']
else:
    # Fallback for older saved files
    E_reconstructed = inverse_results['results']['E_final']

errors = inverse_results['errors']
results = inverse_results['results']
noise_level = inverse_results['noise_level']
scan_optimal = inverse_results.get('scan_optimal')
comparison_summary = inverse_results.get('comparison_summary', None)

print(f"  Noise level: {noise_level*100:.1f}%")
print(f"  MAE: {errors['mae']:.2f}%")
print(f"  RMSE: {errors['rmse']:.4f}")
print(f"  Iterations: {results['n_iterations']}")
if comparison_summary is not None:
    print(f"  Scan MAE: {comparison_summary['scan_mae']:.2f}%")
    print(f"  Rerun-vs-scan rel. L2: {comparison_summary['modulus_diff_rel_l2']:.6e}")

# Generate plots
print("\nGenerating plots...")

print("  Plotting reconstruction results...")
fig1 = plot_reconstruction_results(mesh_info, E_true, E_reconstructed,
                                   errors, noise_level, save_path=results_folder,
                                   filename_stem='reconstruction_results')

print("  Plotting iteration history...")
fig2 = plot_iteration_history(results, save_path=results_folder,
                              noise_level=noise_level,
                              filename_stem='iteration_history')

print("  Plotting gradient field...")
fig3 = plot_gradient_field(mesh_info, results, noise_level=noise_level,
                           save_path=results_folder,
                           filename_stem='gradient_field')

if scan_optimal is not None:
    print("  Plotting scan-vs-rerun comparison...")
    fig4 = plot_reconstruction_comparison(
        mesh_info, E_true,
        scan_optimal['E_reconstructed'], scan_optimal['errors'],
        E_reconstructed, errors,
        noise_level, save_path=results_folder
    )
else:
    print("  Comparison data not found, skipping comparison plot.")
    fig4 = None

print("\n" + "=" * 70)
print("Inverse results plotted successfully!")
print(f"Figures saved to {results_folder}")
print("=" * 70)

plt.show()
