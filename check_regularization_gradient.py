"""
Check gradient of Tikhonov regularization term.

This script verifies that the analytical gradient computed by
get_tikhonov_gradient() matches the numerical finite difference gradient.

Based on the approach in check_Tik_grad.m
"""

import numpy as np
from pathlib import Path
from fgm_asm.mesh import MeshInfo
from fgm_asm.material import MaterialInfo
from fgm_asm.regularization import get_tikhonov_regularization, get_tikhonov_gradient


def check_single_node_gradient(mesh_info, material_info, node_idx, delta=1e-3):
    """
    Check gradient at a single node using finite differences.

    Args:
        mesh_info: MeshInfo object
        material_info: MaterialInfo object
        node_idx: Node index to check
        delta: Perturbation size for finite differences

    Returns:
        relative_error: Relative error between analytical and numerical gradient
        grad_analytical: Analytical gradient value at node_idx
        grad_fd: Finite difference gradient value at node_idx
    """
    # Get current modulus distribution
    E_vec = material_info.get_current_modulus()

    # Compute analytical gradient at all nodes
    grad_analytical_full = get_tikhonov_gradient(mesh_info, material_info)
    grad_analytical = grad_analytical_full[node_idx]

    # Positive perturbation: E_vec[node_idx] + delta
    E_test_plus = E_vec.copy()
    E_test_plus[node_idx] = E_vec[node_idx] + delta

    material_info_test = MaterialInfo(nu=material_info.nu,
                                     dis_type=material_info.dis_type,
                                     alpha=material_info.alpha,
                                     beta=material_info.beta)
    material_info_test.update(E_test_plus, iteration=1)

    Tik_plus = get_tikhonov_regularization(mesh_info, material_info_test)

    # Negative perturbation: E_vec[node_idx] - delta
    E_test_minus = E_vec.copy()
    E_test_minus[node_idx] = E_vec[node_idx] - delta

    material_info_test = MaterialInfo(nu=material_info.nu,
                                     dis_type=material_info.dis_type,
                                     alpha=material_info.alpha,
                                     beta=material_info.beta)
    material_info_test.update(E_test_minus, iteration=1)

    Tik_minus = get_tikhonov_regularization(mesh_info, material_info_test)

    # Finite difference gradient (central difference)
    grad_fd = (Tik_plus - Tik_minus) / (2.0 * delta)

    # Compute relative error
    relative_error = abs(grad_fd - grad_analytical) / (abs(grad_analytical) + 1e-10)

    return relative_error, grad_analytical, grad_fd


def check_all_nodes_gradient(mesh_info, material_info, delta=1e-3,
                             sample_nodes=None, verbose=True):
    """
    Check gradient at multiple nodes.

    Args:
        mesh_info: MeshInfo object
        material_info: MaterialInfo object
        delta: Perturbation size for finite differences
        sample_nodes: List of node indices to check (if None, check random nodes)
        verbose: Whether to print progress

    Returns:
        results: Dictionary with error statistics
    """
    n_nod = mesh_info.n_nod

    # If sample_nodes not specified, randomly select nodes
    if sample_nodes is None:
        n_samples = min(10, n_nod)
        sample_nodes = np.random.choice(n_nod, size=n_samples, replace=False)

    errors = []
    grad_analytical_list = []
    grad_fd_list = []

    if verbose:
        print(f"Checking {len(sample_nodes)} nodes...")
        print("-" * 70)

    for i, node_idx in enumerate(sample_nodes):
        rel_error, grad_ana, grad_fd = check_single_node_gradient(
            mesh_info, material_info, node_idx, delta
        )

        errors.append(rel_error)
        grad_analytical_list.append(grad_ana)
        grad_fd_list.append(grad_fd)

        if verbose:
            status = "PASS" if rel_error < 1e-4 else "FAIL"
            print(f"Node {node_idx:4d}: Rel Error = {rel_error:.6e}  [{status}]")

    errors = np.array(errors)

    results = {
        'sample_nodes': sample_nodes,
        'relative_errors': errors,
        'grad_analytical': np.array(grad_analytical_list),
        'grad_fd': np.array(grad_fd_list),
        'max_error': np.max(errors),
        'mean_error': np.mean(errors),
        'passed': np.all(errors < 1e-4),
        'delta': delta
    }

    return results


def main():
    """Main function to check regularization gradient."""

    print("=" * 70)
    print("CHECK TIKHONOV REGULARIZATION GRADIENT")
    print("=" * 70)

    # ============================================================
    # Setup: Geometry & Mesh (matching check_Tik_grad.m)
    # ============================================================
    print("\n[1] Setting up mesh...")
    geo_l = 9.0
    geo_h = 9.0
    nel_x = 40
    nel_y = 40

    mesh_info = MeshInfo(geo_l, geo_h, nel_x, nel_y)

    print(f"    Domain: {geo_l} × {geo_h}")
    print(f"    Mesh: {nel_x} × {nel_y} elements")
    print(f"    Number of nodes: {mesh_info.n_nod}")

    # ============================================================
    # Initial Material Properties (matching check_Tik_grad.m)
    # ============================================================
    print("\n[2] Setting up material properties...")
    nu = 0.3
    E_min = 0.1
    Ex = 2.0
    Ey = 4.0
    dis_type = 'bil'  # 'bil' or 'exp'

    # Calculate alpha and beta
    if dis_type == 'exp':
        alpha = (Ex - 1.0) / geo_l
        beta = (Ey - 1.0) / geo_h
        iter_E_vec = 1.0 + alpha * mesh_info.X + beta * mesh_info.Y
    else:  # 'bil'
        alpha = np.log(Ex) / geo_l
        beta = np.log(Ey) / geo_h
        iter_E_vec = np.exp(alpha * mesh_info.X + beta * mesh_info.Y)

    # Create MaterialInfo and update
    material_info = MaterialInfo(nu=nu, dis_type=dis_type, alpha=alpha, beta=beta)
    material_info.update(iter_E_vec, iteration=1)

    print(f"    Distribution type: {dis_type}")
    print(f"    Poisson's ratio: {nu}")
    print(f"    Modulus range: [{iter_E_vec.min():.4f}, {iter_E_vec.max():.4f}]")

    # ============================================================
    # Check 1: Single Random Node (matching check_Tik_grad.m)
    # ============================================================
    print("\n" + "=" * 70)
    print("[3] Single Node Gradient Check (Random)")
    print("=" * 70)

    np.random.seed(42)  # For reproducibility
    node_idx = np.random.randint(0, mesh_info.n_nod)
    delta = 1e-3

    print(f"    Random node index: {node_idx}")
    print(f"    Perturbation size (delta): {delta}")
    print(f"    Node position: ({mesh_info.X[node_idx]:.4f}, {mesh_info.Y[node_idx]:.4f})")
    print(f"    Modulus at node: {iter_E_vec[node_idx]:.4f}")

    rel_error, grad_ana, grad_fd = check_single_node_gradient(
        mesh_info, material_info, node_idx, delta
    )

    print("\n    Results:")
    print(f"    Analytical gradient: {grad_ana:.8e}")
    print(f"    Finite diff gradient: {grad_fd:.8e}")
    print(f"    Absolute difference: {abs(grad_ana - grad_fd):.8e}")
    print(f"    Relative error: {rel_error:.8e}")

    threshold = 1e-4
    if rel_error < threshold:
        print(f"\n    [PASS] Relative error < {threshold}")
    else:
        print(f"\n    [FAIL] Relative error >= {threshold}")

    # ============================================================
    # Check 2: Multiple Random Nodes
    # ============================================================
    print("\n" + "=" * 70)
    print("[4] Multiple Nodes Gradient Check")
    print("=" * 70)

    n_samples = 20
    sample_nodes = np.random.choice(mesh_info.n_nod, size=n_samples, replace=False)

    results = check_all_nodes_gradient(
        mesh_info, material_info, delta=1e-3,
        sample_nodes=sample_nodes, verbose=True
    )

    # ============================================================
    # Check 3: Different Delta Values
    # ============================================================
    print("\n" + "=" * 70)
    print("[5] Sensitivity to Delta (Perturbation Size)")
    print("=" * 70)

    deltas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    print(f"Testing node {node_idx} with different delta values:")
    print("-" * 70)
    print(f"{'Delta':>12s}  {'Rel Error':>12s}  {'Status':>8s}")
    print("-" * 70)

    for d in deltas:
        rel_err, _, _ = check_single_node_gradient(mesh_info, material_info, node_idx, d)
        status = "PASS" if rel_err < 1e-4 else "FAIL"
        print(f"{d:12.2e}  {rel_err:12.6e}  {status:>8s}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Single node check: {'[PASS]' if rel_error < threshold else '[FAIL]'}")
    print(f"Multiple nodes check: {'[PASS]' if results['passed'] else '[FAIL]'}")
    print(f"  - Nodes tested: {len(results['sample_nodes'])}")
    print(f"  - Max relative error: {results['max_error']:.6e}")
    print(f"  - Mean relative error: {results['mean_error']:.6e}")
    print(f"  - Threshold: {threshold:.2e}")

    if results['passed']:
        print("\n*** All gradient checks PASSED! ***")
        print("The analytical gradient implementation is correct.")
    else:
        print("\n*** WARNING: Some gradient checks FAILED! ***")
        print("Please review the gradient implementation in regularization.py")

    print("=" * 70)


if __name__ == "__main__":
    main()
