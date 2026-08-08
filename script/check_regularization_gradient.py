"""
Check gradient of the Tikhonov regularization term.

This script verifies two levels of gradients for the current inverse framework:
1. The analytical gradient dR/dEhat from get_tikhonov_gradient()
2. The chained gradient dR/draw after Ehat = exp(raw) / mean(exp(raw))
"""

import numpy as np

from fgm_asm.mesh import MeshInfo
from fgm_asm.material import MaterialInfo
from fgm_asm.regularization import get_tikhonov_regularization, get_tikhonov_gradient
from fgm_asm.inverse_solver import _raw_to_ehat, _transform_gradient_to_raw


def build_material_info(mesh_info, nu=0.3, ex=2.0, ey=4.0):
    """
    Build a smooth positive test field for gradient checks.

    Args:
        mesh_info: MeshInfo object
        nu: Poisson's ratio
        ex: Target modulus ratio in x direction
        ey: Target modulus ratio in y direction

    Returns:
        material_info: MaterialInfo object updated with Ehat field
        raw_vec: Unconstrained field parameters
        ehat_vec: Positive normalized field
    """
    raw_vec = (
        np.log(ex) / mesh_info.geo_l * mesh_info.X +
        np.log(ey) / mesh_info.geo_h * mesh_info.Y
    )
    ehat_vec = _raw_to_ehat(raw_vec)

    material_info = MaterialInfo(nu=nu)
    material_info.update(ehat_vec, iteration=1)
    return material_info, raw_vec, ehat_vec


def evaluate_regularization(mesh_info, nu, raw_vec):
    """
    Evaluate the regularization value under the current raw-to-Ehat parameterization.

    Args:
        mesh_info: MeshInfo object
        nu: Poisson's ratio
        raw_vec: Unconstrained field parameters

    Returns:
        reg_value: Tikhonov regularization value
    """
    material_info = MaterialInfo(nu=nu)
    material_info.update(_raw_to_ehat(raw_vec), iteration=1)
    return get_tikhonov_regularization(mesh_info, material_info)


def check_single_node_gradient_ehat(mesh_info, material_info, node_idx, delta=1e-6):
    """
    Check dR/dEhat at a single node using finite differences.

    Args:
        mesh_info: MeshInfo object
        material_info: MaterialInfo object
        node_idx: Node index to check
        delta: Perturbation size

    Returns:
        relative_error: Relative error between analytical and finite-difference gradients
        grad_analytical: Analytical gradient at node_idx
        grad_fd: Finite-difference gradient at node_idx
    """
    ehat_vec = material_info.get_current_modulus()
    grad_analytical_full = get_tikhonov_gradient(mesh_info, material_info)
    grad_analytical = float(grad_analytical_full[node_idx])

    ehat_plus = ehat_vec.copy()
    ehat_plus[node_idx] += delta
    material_plus = MaterialInfo(nu=material_info.nu)
    material_plus.update(ehat_plus, iteration=1)
    reg_plus = get_tikhonov_regularization(mesh_info, material_plus)

    ehat_minus = ehat_vec.copy()
    ehat_minus[node_idx] -= delta
    material_minus = MaterialInfo(nu=material_info.nu)
    material_minus.update(ehat_minus, iteration=1)
    reg_minus = get_tikhonov_regularization(mesh_info, material_minus)

    grad_fd = float((reg_plus - reg_minus) / (2.0 * delta))
    relative_error = abs(grad_fd - grad_analytical) / (abs(grad_analytical) + 1e-12)
    return relative_error, grad_analytical, grad_fd


def check_single_node_gradient_raw(mesh_info, nu, raw_vec, node_idx, delta=1e-6):
    """
    Check dR/draw at a single node using finite differences.

    Args:
        mesh_info: MeshInfo object
        nu: Poisson's ratio
        raw_vec: Unconstrained field parameters
        node_idx: Node index to check
        delta: Perturbation size

    Returns:
        relative_error: Relative error between analytical and finite-difference gradients
        grad_analytical: Analytical chained gradient at node_idx
        grad_fd: Finite-difference gradient at node_idx
    """
    material_info = MaterialInfo(nu=nu)
    material_info.update(_raw_to_ehat(raw_vec), iteration=1)

    grad_ehat = get_tikhonov_gradient(mesh_info, material_info)
    grad_analytical_full = _transform_gradient_to_raw(raw_vec, grad_ehat)
    grad_analytical = float(grad_analytical_full[node_idx])

    raw_plus = raw_vec.copy()
    raw_plus[node_idx] += delta
    reg_plus = evaluate_regularization(mesh_info, nu, raw_plus)

    raw_minus = raw_vec.copy()
    raw_minus[node_idx] -= delta
    reg_minus = evaluate_regularization(mesh_info, nu, raw_minus)

    grad_fd = float((reg_plus - reg_minus) / (2.0 * delta))
    relative_error = abs(grad_fd - grad_analytical) / (abs(grad_analytical) + 1e-12)
    return relative_error, grad_analytical, grad_fd


def check_directional_gradient_raw(mesh_info, nu, raw_vec, delta=1e-6):
    """
    Check the full chained gradient dR/draw using a random directional derivative.

    Args:
        mesh_info: MeshInfo object
        nu: Poisson's ratio
        raw_vec: Unconstrained field parameters
        delta: Perturbation size

    Returns:
        relative_error: Relative error between analytical and finite-difference derivatives
        directional_analytical: Analytical directional derivative
        directional_fd: Finite-difference directional derivative
    """
    material_info = MaterialInfo(nu=nu)
    material_info.update(_raw_to_ehat(raw_vec), iteration=1)

    grad_ehat = get_tikhonov_gradient(mesh_info, material_info)
    grad_raw = _transform_gradient_to_raw(raw_vec, grad_ehat)

    direction = np.random.randn(raw_vec.size)
    direction /= np.linalg.norm(direction)

    directional_analytical = float(grad_raw @ direction)
    reg_plus = evaluate_regularization(mesh_info, nu, raw_vec + delta * direction)
    reg_minus = evaluate_regularization(mesh_info, nu, raw_vec - delta * direction)
    directional_fd = float((reg_plus - reg_minus) / (2.0 * delta))

    relative_error = abs(directional_fd - directional_analytical) / (abs(directional_analytical) + 1e-12)
    return relative_error, directional_analytical, directional_fd


def main():
    """Main function to check regularization gradients."""
    print("=" * 70)
    print("CHECK TIKHONOV REGULARIZATION GRADIENT")
    print("=" * 70)

    print("\n[1] Setting up mesh...")
    geo_l = 9.0
    geo_h = 9.0
    nel_x = 40
    nel_y = 40
    mesh_info = MeshInfo(geo_l, geo_h, nel_x, nel_y)
    mesh_info.get_regularization_matrix()

    print(f"    Domain: {geo_l} x {geo_h}")
    print(f"    Mesh: {nel_x} x {nel_y} elements")
    print(f"    Number of nodes: {mesh_info.n_nod}")

    print("\n[2] Building test field under raw -> Ehat parameterization...")
    nu = 0.3
    material_info, raw_vec, ehat_vec = build_material_info(mesh_info, nu=nu)

    print(f"    Poisson's ratio: {nu}")
    print(f"    Raw field range: [{raw_vec.min():.4f}, {raw_vec.max():.4f}]")
    print(f"    Ehat range: [{ehat_vec.min():.4f}, {ehat_vec.max():.4f}]")
    print(f"    Ehat mean: {ehat_vec.mean():.6f}")

    np.random.seed(42)
    node_idx = np.random.randint(0, mesh_info.n_nod)
    delta = 1e-6

    print("\n" + "=" * 70)
    print("[3] Single Node Gradient Check in Ehat Space")
    print("=" * 70)
    print(f"    Node index: {node_idx}")
    print(f"    Node position: ({mesh_info.X[node_idx]:.4f}, {mesh_info.Y[node_idx]:.4f})")
    print(f"    Delta: {delta:.1e}")

    rel_ehat, grad_ehat_ana, grad_ehat_fd = check_single_node_gradient_ehat(
        mesh_info, material_info, node_idx, delta=delta
    )

    print(f"    Analytical dR/dEhat: {grad_ehat_ana:.8e}")
    print(f"    Finite diff dR/dEhat: {grad_ehat_fd:.8e}")
    print(f"    Relative error: {rel_ehat:.8e}")

    print("\n" + "=" * 70)
    print("[4] Single Node Gradient Check in Raw Space")
    print("=" * 70)

    rel_raw, grad_raw_ana, grad_raw_fd = check_single_node_gradient_raw(
        mesh_info, nu, raw_vec, node_idx, delta=delta
    )

    print(f"    Analytical dR/draw: {grad_raw_ana:.8e}")
    print(f"    Finite diff dR/draw: {grad_raw_fd:.8e}")
    print(f"    Relative error: {rel_raw:.8e}")

    print("\n" + "=" * 70)
    print("[5] Directional Gradient Check in Raw Space")
    print("=" * 70)

    rel_dir, dir_ana, dir_fd = check_directional_gradient_raw(
        mesh_info, nu, raw_vec, delta=delta
    )

    print(f"    Analytical directional derivative: {dir_ana:.8e}")
    print(f"    Finite diff directional derivative: {dir_fd:.8e}")
    print(f"    Relative error: {rel_dir:.8e}")

    threshold = 1e-4

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Ehat-space node check: {'[PASS]' if rel_ehat < threshold else '[FAIL]'}")
    print(f"Raw-space node check: {'[PASS]' if rel_raw < threshold else '[FAIL]'}")
    print(f"Raw-space directional check: {'[PASS]' if rel_dir < threshold else '[FAIL]'}")
    print(f"Threshold: {threshold:.2e}")

    if rel_ehat < threshold and rel_raw < threshold and rel_dir < threshold:
        print("\n*** All gradient checks PASSED! ***")
        print("The regularization gradient and chain-rule transformation are consistent.")
    else:
        print("\n*** WARNING: Some gradient checks FAILED! ***")
        print("Please review regularization.py or the raw-to-Ehat gradient transformation.")

    print("=" * 70)


if __name__ == "__main__":
    main()
