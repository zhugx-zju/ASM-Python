"""
Plot L-curve analysis results from saved data.

This script reads the saved L-curve data file and creates two figures:
1. Figure 1: L-curve (data mismatch vs regularization)
2. Figure 2: Curvature vs regularization parameter
"""

import matplotlib.pyplot as plt

from fgm_asm import plot_lcurve_results
from fgm_asm.results_io import find_lcurve_data_path, load_lcurve_data


print("=" * 70)
print("L-curve Results Plotter")
print("=" * 70)

print("\nSearching for L-curve analysis data...")
lcurve_path, results_folder = find_lcurve_data_path()

lcurve_results = load_lcurve_data(lcurve_path)
print(f"  Loaded L-curve analysis from {lcurve_path}")

print("\nGenerating plots...")
print("  Plotting L-curve and curvature analysis...")
plot_lcurve_results(lcurve_results, save_path=results_folder)

print("\n" + "=" * 70)
print("L-curve analysis plotted successfully!")
print(f"Figures saved to {results_folder}")
print("=" * 70)

plt.show()
