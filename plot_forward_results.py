"""
Plot forward problem results from saved data.

This script reads the forward problem data from the folder specified in config.py
and creates four separate figures:
1. Figure 1: Modulus distribution
2. Figure 2: Displacement fields (x and y displacements in 1 row, 2 columns)
3. Figure 3: X-displacement only
4. Figure 4: Y-displacement only
"""

import matplotlib.pyplot as plt
from fgm_asm.visualization import (
    plot_displacement_fields,
    plot_modulus_distribution,
    plot_single_displacement_field,
)
from fgm_asm.results_io import (
    find_forward_data_path,
    load_forward_data,
)


# ============================================================
# Main execution
# ============================================================
print("=" * 70)
print("Forward Results Plotter")
print("=" * 70)

# Search for and load forward problem data
print("\nSearching for forward problem data...")
forward_data_path, _ = find_forward_data_path()

# Load data
forward_data = load_forward_data(forward_data_path)
print(f"  Loaded data from {forward_data_path}")

mesh_info = forward_data['mesh_info']
E_field = forward_data['E_field']
U = forward_data['U']
save_path = forward_data_path.parent

# Generate plots
print("\nGenerating plots...")

print("  Plotting modulus distribution...")
fig1 = plot_modulus_distribution(mesh_info, E_field, save_path=save_path)

print("  Plotting displacement fields...")
fig2 = plot_displacement_fields(mesh_info, U, save_path=save_path)

print("  Plotting X-displacement only...")
fig3 = plot_single_displacement_field(mesh_info, U, component='ux', save_path=save_path)

print("  Plotting Y-displacement only...")
fig4 = plot_single_displacement_field(mesh_info, U, component='uy', save_path=save_path)

print("\n" + "=" * 70)
print("Forward results plotted successfully!")
print(f"Figures saved to {save_path}")
print("=" * 70)

plt.show()
