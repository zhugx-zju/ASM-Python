"""
Forward FEM solver for FGM plane stress problems.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import factorized


class FEMInfo:
    """
    Finite element information storage.
    """

    def __init__(self, mesh_info, material_info, bc_info):
        """
        Initialize FEM information.

        Args:
            mesh_info: MeshInfo object
            material_info: MaterialInfo object
            bc_info: Boundary condition dictionary
        """
        self.mesh_info = mesh_info
        self.material_info = material_info
        self.bc_info = bc_info

        self.B_gauss = mesh_info.B_gauss
        self.det_j_gauss = mesh_info.gauss_det_j
        self.D0 = None

        self.K = None
        self._solve_free = None

        self.fix_dof = np.asarray(bc_info['fixdof'], dtype=np.int32)
        free_mask = np.ones(mesh_info.n_dof, dtype=bool)
        free_mask[self.fix_dof] = False
        self.free_dof = np.flatnonzero(free_mask)


def _get_gauss_modulus(mesh_info, material_info):
    """Interpolate the nodal modulus field to the Gauss points."""
    E_vec = material_info.get_current_modulus()
    G_ele = E_vec[mesh_info.ele_nods_id - 1]
    return (G_ele @ mesh_info.gauss_N.T).T


def get_fgm_ke(mesh_info, material_info, D0):
    """
    Calculate element stiffness matrices for all elements.

    Args:
        mesh_info: MeshInfo object
        material_info: MaterialInfo object
        D0: Base elasticity matrix [3, 3]

    Returns:
        ke: Element stiffness matrices [n_el, 8, 8]
    """
    E_gauss = _get_gauss_modulus(mesh_info, material_info)
    weights = E_gauss * mesh_info.gauss_det_j * mesh_info.gauss_w[:, None]
    return np.einsum('geia,ij,gejb,ge->eab', mesh_info.B_gauss, D0, mesh_info.B_gauss, weights)


def assemble_global_stiffness(mesh_info, ke):
    """
    Assemble the global stiffness matrix from element stiffness blocks.

    Args:
        mesh_info: MeshInfo object
        ke: Element stiffness matrices [n_el, 8, 8]

    Returns:
        K: Global stiffness matrix [n_dof, n_dof]
    """
    K = csr_matrix(
        (ke.ravel(), (mesh_info.global_row_idx.ravel(), mesh_info.global_col_idx.ravel())),
        shape=(mesh_info.n_dof, mesh_info.n_dof),
    )
    K.sum_duplicates()
    return K.tocsr()


def fem_assemble(mesh_info, material_info, bc_info):
    """
    Assemble FEM system information for the current modulus field.

    Args:
        mesh_info: MeshInfo object
        material_info: MaterialInfo object
        bc_info: Boundary condition dictionary

    Returns:
        fem_info: FEMInfo object with assembled matrices
    """
    fem_info = FEMInfo(mesh_info, material_info, bc_info)
    fem_info.D0 = material_info.get_elasticity_matrix()
    ke = get_fgm_ke(mesh_info, material_info, fem_info.D0)
    fem_info.K = assemble_global_stiffness(mesh_info, ke)
    return fem_info


def _get_free_solver(fem_info):
    """Return the cached sparse solve callable for the free DOFs."""
    if fem_info._solve_free is None:
        K_free = fem_info.K[fem_info.free_dof][:, fem_info.free_dof].tocsc()
        fem_info._solve_free = factorized(K_free)
    return fem_info._solve_free


def solve_system(fem_info, rhs):
    """
    Solve a linear system with the current stiffness matrix.

    Args:
        fem_info: FEMInfo object
        rhs: Right-hand side vector [n_dof]

    Returns:
        solution: Solution vector [n_dof]
    """
    solver = _get_free_solver(fem_info)
    solution = np.zeros(fem_info.mesh_info.n_dof)
    solution[fem_info.free_dof] = solver(np.asarray(rhs)[fem_info.free_dof])
    return solution


def forward_solver(fem_info):
    """
    Solve the forward FEM problem for displacement.

    Args:
        fem_info: FEMInfo object

    Returns:
        U: Displacement vector [n_dof]
    """
    return solve_system(fem_info, fem_info.bc_info['force'])


def compute_reaction_forces(fem_info, U):
    """
    Compute reaction forces at fixed DOFs.

    Args:
        fem_info: FEMInfo object
        U: Displacement vector [n_dof]

    Returns:
        RF: Reaction force vector [n_dof]
    """
    return fem_info.K @ U - fem_info.bc_info['force']
