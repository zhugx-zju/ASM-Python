"""
Inverse problem solver using SciPy L-BFGS-B with Tikhonov regularization.
"""

import numpy as np
from scipy.optimize import minimize

from .fem_forward import fem_assemble, forward_solver, solve_system
from .regularization import get_tikhonov_regularization, get_tikhonov_gradient


def lambda_distance(fem_info, forw_U=None, U_measured=None, rhs=None):
    """
    Compute the adjoint variable for the inverse problem.

    Args:
        fem_info: FEMInfo object
        forw_U: Forward solution displacement [n_dof]
        U_measured: Measured displacement [n_dof]
        rhs: Optional precomputed adjoint right-hand side [n_dof]

    Returns:
        lambda_vec: Lagrange multiplier [n_dof]
    """
    if rhs is None:
        if forw_U is None or U_measured is None:
            raise ValueError("Either rhs or both forw_U and U_measured must be provided.")
        mesh_info = fem_info.mesh_info
        if mesh_info.M is None:
            mesh_info.assemble_mass_matrix()
        rhs = np.asarray(mesh_info.M @ (forw_U - U_measured)).ravel()
    return solve_system(fem_info, rhs)


def get_stiffness_gradient(fem_info, lambda_vec, forw_U):
    """
    Calculate the gradient of the objective function with respect to nodal modulus.

    Args:
        fem_info: FEMInfo object
        lambda_vec: Lagrange multiplier [n_dof]
        forw_U: Forward solution displacement [n_dof]

    Returns:
        grad: Gradient vector [n_nod]
    """
    mesh_info = fem_info.mesh_info
    ele_dof_id = mesh_info.ele_dof_id
    ele_nods_id = mesh_info.ele_nods_id - 1
    id_list = ele_nods_id.T.ravel()

    lambda_ele = lambda_vec[ele_dof_id]
    forw_U_ele = forw_U[ele_dof_id]

    grad_E = np.zeros((mesh_info.n_el, 4))

    for ig in range(mesh_info.gauss_N.shape[0]):
        B_i = fem_info.B_gauss[ig]
        BD0B = np.einsum('eia,ij,ejb,e->eab', B_i, fem_info.D0, B_i, fem_info.det_j_gauss[ig])
        directional_term = np.einsum('ea,eab,eb->e', lambda_ele, BD0B, forw_U_ele)
        grad_E += (mesh_info.gauss_w[ig] * directional_term)[:, None] * mesh_info.gauss_N[ig][None, :]

    return -np.bincount(id_list, weights=grad_E.T.ravel(), minlength=mesh_info.n_nod)


def _evaluate_inverse_state(mesh_info, bc_info, U_measured, material_info, gamma, E_vec, iteration):
    """
    Evaluate the inverse objective, gradient, and reusable solver state.

    Args:
        mesh_info: MeshInfo object
        bc_info: Boundary condition dictionary
        U_measured: Measured displacement data [n_dof]
        material_info: MaterialInfo object
        gamma: Regularization coefficient
        E_vec: Current modulus field [n_nod]
        iteration: Iteration or evaluation identifier

    Returns:
        state: Dictionary with objective, gradient, and final solver state
    """
    material_info.update(E_vec, iteration=iteration)
    fem_info = fem_assemble(mesh_info, material_info, bc_info)
    forw_U = forward_solver(fem_info)

    residual = U_measured - forw_U
    mass_residual = np.asarray(mesh_info.M @ residual).ravel()
    cost_tar = 0.5 * float(residual @ mass_residual)

    reg_value = get_tikhonov_regularization(mesh_info, material_info)
    cost_reg = 0.5 * gamma * reg_value

    lambda_vec = lambda_distance(fem_info, rhs=-mass_residual)
    grad_tar_E = get_stiffness_gradient(fem_info, lambda_vec, forw_U)
    grad_reg_E = 0.5 * gamma * get_tikhonov_gradient(mesh_info, material_info)

    grad_tar_m = grad_tar_E * E_vec
    grad_reg_m = grad_reg_E * E_vec
    grad_m = grad_tar_m + grad_reg_m

    return {
        'E_vec': np.array(E_vec, copy=True),
        'fem_info': fem_info,
        'forw_U': forw_U,
        'lambda_vec': lambda_vec,
        'residual': residual,
        'cost_tar': cost_tar,
        'cost_reg': cost_reg,
        'objective': cost_tar + cost_reg,
        'reg_value': reg_value,
        'residual_norm': float(np.sqrt(max(2.0 * cost_tar, 0.0))),
        'regularization_norm': float(np.sqrt(max(reg_value, 0.0))),
        'grad_tar_E': grad_tar_E,
        'grad_reg_E': grad_reg_E,
        'grad_tar_m': grad_tar_m,
        'grad_reg_m': grad_reg_m,
        'grad_m': grad_m,
    }


def lbfgs_inverse_solver_scipy(mesh_info, bc_info, U_measured, E_init=None,
                               gamma=1e-6, E_min=0.001, E_max=1000.0,
                               max_iter=2000, ftol=1e-12, gtol=1e-8, nu=0.3):
    """
    Solve the inverse problem using SciPy's L-BFGS-B optimizer in m = ln(E) space.

    Args:
        mesh_info: MeshInfo object
        bc_info: Boundary condition dictionary
        U_measured: Measured displacement data [n_dof]
        E_init: Initial modulus guess [n_nod]
        gamma: Regularization coefficient
        E_min: Minimum modulus for initialization (only used if E_init is None)
        E_max: Maximum modulus (kept for compatibility)
        max_iter: Maximum iterations
        ftol: Function tolerance
        gtol: Gradient tolerance in m space
        nu: Poisson's ratio

    Returns:
        results: Dictionary containing optimization results and cached final state
    """
    del E_max
    from .material import MaterialInfo

    if mesh_info.M is None:
        mesh_info.assemble_mass_matrix()
    mesh_info.get_regularization_matrix()

    if E_init is None:
        E_vec0 = E_min + 0.5 * np.ones(mesh_info.n_nod)
    else:
        E_vec0 = np.asarray(E_init, dtype=float).copy()

    m_vec0 = np.log(E_vec0)
    material_info = MaterialInfo(nu=nu)

    print("Starting L-BFGS-B optimization in m = ln(E) space...")
    print(f"  Initial modulus: min={E_vec0.min():.4f}, max={E_vec0.max():.4f}, mean={E_vec0.mean():.4f}")
    print(f"  Max iterations: {max_iter}, ftol: {ftol:.2e}, gtol: {gtol:.2e}")

    evaluation_counter = [0]
    state_cache = {'m': np.array(m_vec0, copy=True), 'state': None}

    initial_state = _evaluate_inverse_state(
        mesh_info, bc_info, U_measured, material_info, gamma, E_vec0, iteration=1
    )
    state_cache['state'] = initial_state

    cost_history = [initial_state['objective']]
    cost_tar_history = [initial_state['cost_tar']]
    cost_reg_history = [initial_state['cost_reg']]
    gradient_norms = [np.linalg.norm(initial_state['grad_m'])]

    def objective_and_gradient(m_vec_flat):
        m_vec_use = np.array(m_vec_flat, copy=False)
        E_vec = np.exp(m_vec_use)
        evaluation_counter[0] += 1
        state = _evaluate_inverse_state(
            mesh_info,
            bc_info,
            U_measured,
            material_info,
            gamma,
            E_vec,
            iteration=evaluation_counter[0] + 1,
        )
        state_cache['m'] = np.array(m_vec_use, copy=True)
        state_cache['state'] = state
        return state['objective'], state['grad_m']

    def callback(xk):
        m_current = np.array(xk, copy=False)
        state = state_cache['state']

        if state is None or state_cache['m'] is None or not np.array_equal(m_current, state_cache['m']):
            state = _evaluate_inverse_state(
                mesh_info,
                bc_info,
                U_measured,
                material_info,
                gamma,
                np.exp(m_current),
                iteration=evaluation_counter[0] + 1,
            )
            state_cache['m'] = np.array(m_current, copy=True)
            state_cache['state'] = state

        cost_history.append(state['objective'])
        cost_tar_history.append(state['cost_tar'])
        cost_reg_history.append(state['cost_reg'])
        gradient_norms.append(np.linalg.norm(state['grad_m']))

        accepted_iterations = len(cost_history) - 1
        if accepted_iterations > 0 and accepted_iterations % 10 == 0:
            print(f"  Iteration {accepted_iterations:4d}: "
                  f"Cost = {state['objective']:.6e}, "
                  f"||grad_m|| = {gradient_norms[-1]:.6e}")

    result = minimize(
        fun=objective_and_gradient,
        x0=m_vec0,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': max_iter,
            'ftol': ftol,
            'gtol': gtol,
            'disp': False,
            'maxls': 50,
            'maxcor': 10,
        },
        callback=callback,
    )

    if state_cache['m'] is not None and np.array_equal(result.x, state_cache['m']):
        final_state = state_cache['state']
    else:
        final_state = _evaluate_inverse_state(
            mesh_info,
            bc_info,
            U_measured,
            material_info,
            gamma,
            np.exp(result.x),
            iteration=evaluation_counter[0] + 1,
        )

    E_final = final_state['E_vec']

    print(f"  Optimization completed: {result.message}")
    print(f"  Final modulus: min={E_final.min():.4f}, max={E_final.max():.4f}, mean={E_final.mean():.4f}")

    return {
        'E_final': E_final,
        'U_final': final_state['forw_U'],
        'cost_history': np.array(cost_history),
        'cost_tar_history': np.array(cost_tar_history),
        'cost_reg_history': np.array(cost_reg_history),
        'grad_norm_history': np.array(gradient_norms),
        'gradient_norms': np.array(gradient_norms),
        'cond_H_history': np.ones(len(gradient_norms)),
        'n_iterations': int(result.nit),
        'converged': bool(result.success),
        'message': str(result.message),
        'final_cost': final_state['objective'],
        'residual_norm': final_state['residual_norm'],
        'regularization_norm': final_state['regularization_norm'],
        'final_gradient': final_state['grad_m'],
        'final_grad_tar': final_state['grad_tar_m'],
        'final_grad_reg': final_state['grad_reg_m'],
    }
