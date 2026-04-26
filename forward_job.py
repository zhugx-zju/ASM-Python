"""
Forward problem solver for FGM plane stress analysis.

This script solves the forward problem to obtain displacement field
for a functionally graded material under plane stress conditions.
Results are saved for later use in inverse problem.
"""

from pathlib import Path
from fgm_asm import MeshInfo, fem_assemble, forward_solver, generate_fgm_modulus
from fgm_asm.mesh import setup_boundary_conditions
from fgm_asm.results_io import save_forward_data
from fgm_asm.visualization import visualize_forward_results
import config as cfg


# ============================================================
# Configuration
# ============================================================
print("=" * 60)
print("Forward Problem Solver for FGM Plane Stress")
print("=" * 60)

# Load configuration from config.py
config = cfg.get_forward_config()
inverse_config = cfg.get_inverse_config()

print(f"\nConfiguration:")
print(f"  Geometry: {config.geo_l:.1f} x {config.geo_h:.1f}")
print(f"  Mesh: {config.nel_x} x {config.nel_y} elements")
print(f"  Distribution type: {config.dis_type}")
print(f"  Ex: {config.Ex:.2f}, Ey: {config.Ey:.2f}")


# ============================================================
# Create mesh
# ============================================================
print("\nGenerating mesh...")
mesh_info = MeshInfo(config.geo_l, config.geo_h, config.nel_x, config.nel_y)
print(f"  Number of nodes: {mesh_info.n_nod}")
print(f"  Number of elements: {mesh_info.n_el}")
print(f"  Number of DOFs: {mesh_info.n_dof}")


# ============================================================
# Generate FGM modulus field
# ============================================================
print("\nGenerating FGM modulus field...")
E_field, material_info = generate_fgm_modulus(
    mesh_info,
    dis_type=config.dis_type,
    Ex=config.Ex,
    Ey=config.Ey
)
material_info.nu = config.nu
print(f"  Alpha: {material_info.alpha:.6f}")
print(f"  Beta: {material_info.beta:.6f}")
print(f"  E range: [{E_field.min():.4f}, {E_field.max():.4f}]")

# Update material info with initial modulus
E_vec = E_field.ravel()
material_info.update(E_vec, iteration=1)


# ============================================================
# Setup boundary conditions
# ============================================================
print("\nSetting up boundary conditions...")
bc_info = setup_boundary_conditions(mesh_info, config.geo_l, config.geo_h, config.f_tot)
print(f"  Fixed DOFs: {len(bc_info['fixdof'])}")
print(f"  Total load: {bc_info['force'].sum():.6f}")


# ============================================================
# Assemble and solve
# ============================================================
print("\nAssembling FEM system...")
fem_info = fem_assemble(mesh_info, material_info, bc_info)

print("Solving forward problem...")
U = forward_solver(fem_info)
print(f"  Max displacement: {U.max():.6e}")


# ============================================================
# Save results
# ============================================================
print("\nSaving results...")

# Generate folder name using alpha, beta, and gamma from config
alpha = material_info.alpha
beta = material_info.beta
gamma = inverse_config.gamma

folder_name = config.output_folder_name(alpha, beta, gamma)
output_dir = Path(folder_name)
output_dir.mkdir(exist_ok=True)
print(f"  Output directory: {output_dir}")

results = {
    'config': config,
    'mesh_info': mesh_info,
    'bc_info': bc_info,
    'E_field': E_field,
    'U': U,
    'material_info': material_info,
    'folder_name': folder_name,  # Save folder name for reference
}

output_dir = save_forward_data(results, folder_name).parent
print(f"  Data saved to {output_dir / 'forward_problem_data.pkl'}")


# ============================================================
# Visualize
# ============================================================
print("\nGenerating visualizations...")
visualize_forward_results(mesh_info, E_field, U, config, save_path=output_dir, show=False)

print("\n" + "=" * 60)
print("Forward problem solved successfully!")
print("=" * 60)
