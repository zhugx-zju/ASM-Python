"""
Mesh generation and management module for FGM plane stress problems.
"""

import numpy as np
from scipy.sparse import csr_matrix


def shape_fun_at_gauss():
    """
    Get shape function values at Gauss integration points.

    Returns:
        N: Shape function values [4, 4]
        dNs: Derivatives with respect to s [4, 4]
        dNt: Derivatives with respect to t [4, 4]
        w: Gauss weights [4, 1]
    """
    gpt = 1.0 / np.sqrt(3.0)

    s = gpt * np.array([[-1], [1], [1], [-1]])
    t = gpt * np.array([[-1], [-1], [1], [1]])
    w = np.ones((4, 1))

    N1 = (1 - s) * (1 - t) / 4.0
    N2 = (1 + s) * (1 - t) / 4.0
    N3 = (1 + s) * (1 + t) / 4.0
    N4 = (1 - s) * (1 + t) / 4.0

    N1s = -(1 - t) / 4.0
    N2s = (1 - t) / 4.0
    N3s = (1 + t) / 4.0
    N4s = -(1 + t) / 4.0

    N1t = -(1 - s) / 4.0
    N2t = -(1 + s) / 4.0
    N3t = (1 + s) / 4.0
    N4t = (1 - s) / 4.0

    N = np.hstack([N1, N2, N3, N4])
    dNs = np.hstack([N1s, N2s, N3s, N4s])
    dNt = np.hstack([N1t, N2t, N3t, N4t])

    return N, dNs, dNt, w


class MeshInfo:
    """
    Mesh information class for a rectangular domain with uniform quad elements.
    """

    def __init__(self, geo_l, geo_h, nel_x, nel_y):
        """
        Initialize mesh information.

        Args:
            geo_l: Length of the domain
            geo_h: Height of the domain
            nel_x: Number of elements in x direction
            nel_y: Number of elements in y direction
        """
        self.geo_l = geo_l
        self.geo_h = geo_h
        self.nel_x = nel_x
        self.nel_y = nel_y

        self.el = geo_l / nel_x
        self.eh = geo_h / nel_y
        self.ele_size = np.array([self.el, self.eh])

        self.nods_x = nel_x + 1
        self.nods_y = nel_y + 1
        self.n_nod = self.nods_x * self.nods_y
        self.n_el = nel_x * nel_y
        self.n_dof = 2 * self.n_nod

        x = np.linspace(0, geo_l, self.nods_x)
        y = np.linspace(0, geo_h, self.nods_y)
        plot_x, plot_y = np.meshgrid(x, y)

        # Nodes are stored row-by-row on the plotting grid:
        # y increases between rows and x increases within each row.
        self.coord = np.column_stack([
            plot_x.T.ravel('F'),
            plot_y.T.ravel('F'),
        ])
        self.X = self.coord[:, 0]
        self.Y = self.coord[:, 1]

        self.plot_x = plot_x
        self.plot_y = plot_y

        mesh_nods = np.arange(1, self.n_nod + 1).reshape(self.nods_x, self.nods_y, order='F')
        mesh_nods = mesh_nods.astype(np.int32)

        self.ele_nods_id = (
            mesh_nods[:-1, :-1].ravel('F').reshape(-1, 1) +
            np.array([[0, 1, nel_x + 2, nel_x + 1]], dtype=np.int32)
        )
        self.x_ele = self.X[self.ele_nods_id - 1]
        self.y_ele = self.Y[self.ele_nods_id - 1]

        c_vec = (2 * mesh_nods[:-1, :-1] + 1).ravel('F').reshape(-1, 1)
        self.cMat = c_vec + np.array(
            [[-2, -1, 0, 1, 2 * nel_x + 2, 2 * nel_x + 3, 2 * nel_x, 2 * nel_x + 1]],
            dtype=np.int32,
        )
        self.ele_dof_id = self.cMat - 1

        self.M = None
        self.L_reg = None

        self._generate_sparse_indices()
        self._precompute_geometry()

    def _generate_sparse_indices(self):
        """Generate sparse-assembly indices for element matrices."""
        local_rows, local_cols = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
        self.global_row_idx = self.ele_dof_id[:, local_rows.ravel()]
        self.global_col_idx = self.ele_dof_id[:, local_cols.ravel()]

        ele_nods_id0 = self.ele_nods_id - 1
        node_rows, node_cols = np.meshgrid(np.arange(4), np.arange(4), indexing='ij')
        self.reg_row_idx = ele_nods_id0[:, node_rows.ravel()]
        self.reg_col_idx = ele_nods_id0[:, node_cols.ravel()]

    def _precompute_geometry(self):
        """Precompute all mesh-dependent, material-independent quadrature data."""
        N, dNs, dNt, w = shape_fun_at_gauss()

        n_gauss = N.shape[0]
        n_el = self.n_el

        self.gauss_N = N
        self.gauss_dNs = dNs
        self.gauss_dNt = dNt
        self.gauss_w = w[:, 0]

        self.gauss_det_j = np.empty((n_gauss, n_el))
        self.gauss_abs_det_j = np.empty((n_gauss, n_el))
        self.gauss_dNdx = np.empty((n_gauss, n_el, 4))
        self.gauss_dNdy = np.empty((n_gauss, n_el, 4))
        self.B_gauss = np.zeros((n_gauss, n_el, 3, 8))
        self.gauss_mass_local = np.empty((n_gauss, 8, 8))

        x_ele_t = self.x_ele.T
        y_ele_t = self.y_ele.T

        for ig in range(n_gauss):
            N_i = N[ig, :]
            dNs_i = dNs[ig, :]
            dNt_i = dNt[ig, :]

            dxds = dNs_i @ x_ele_t
            dxdt = dNt_i @ x_ele_t
            dyds = dNs_i @ y_ele_t
            dydt = dNt_i @ y_ele_t
            det_j = dxds * dydt - dyds * dxdt

            dNdx = (dydt[:, None] * dNs_i - dyds[:, None] * dNt_i) / det_j[:, None]
            dNdy = (-dxdt[:, None] * dNs_i + dxds[:, None] * dNt_i) / det_j[:, None]

            self.gauss_det_j[ig] = det_j
            self.gauss_abs_det_j[ig] = np.abs(det_j)
            self.gauss_dNdx[ig] = dNdx
            self.gauss_dNdy[ig] = dNdy

            self.B_gauss[ig, :, 0, 0::2] = dNdx
            self.B_gauss[ig, :, 1, 1::2] = dNdy
            self.B_gauss[ig, :, 2, 0::2] = dNdy
            self.B_gauss[ig, :, 2, 1::2] = dNdx

            Nm = np.zeros((2, 8))
            Nm[0, 0::2] = N_i
            Nm[1, 1::2] = N_i
            self.gauss_mass_local[ig] = Nm.T @ Nm

    def assemble_mass_matrix(self):
        """
        Assemble the global mass matrix for the mesh.

        Returns:
            M: Sparse mass matrix [n_dof, n_dof]
        """
        weights = self.gauss_abs_det_j * self.gauss_w[:, None]
        M_ele = np.einsum('gij,ge->eij', self.gauss_mass_local, weights)

        M = csr_matrix(
            (M_ele.ravel(), (self.global_row_idx.ravel(), self.global_col_idx.ravel())),
            shape=(self.n_dof, self.n_dof),
        )
        M.sum_duplicates()
        self.M = M.tocsr()
        return self.M

    def assemble_regularization_matrix(self):
        """
        Assemble the quadratic-form matrix for the Tikhonov regularization.

        Returns:
            L_reg: Sparse matrix [n_nod, n_nod]
        """
        weights = self.gauss_abs_det_j * self.gauss_w[:, None]
        L_ele = np.einsum('gea,geb,ge->eab', self.gauss_dNdx, self.gauss_dNdx, weights)
        L_ele += np.einsum('gea,geb,ge->eab', self.gauss_dNdy, self.gauss_dNdy, weights)

        L_reg = csr_matrix(
            (L_ele.ravel(), (self.reg_row_idx.ravel(), self.reg_col_idx.ravel())),
            shape=(self.n_nod, self.n_nod),
        )
        L_reg.sum_duplicates()
        self.L_reg = L_reg.tocsr()
        return self.L_reg

    def get_regularization_matrix(self):
        """Return the preassembled regularization matrix."""
        if self.L_reg is None:
            self.assemble_regularization_matrix()
        return self.L_reg


def get_body_force_load(mesh_info, bf_vec):
    """
    Calculate the load vector due to body forces.

    For the current formulation this is exactly the mass-matrix action.

    Args:
        mesh_info: MeshInfo object
        bf_vec: Body force vector [n_dof]

    Returns:
        F: Load vector [n_dof]
    """
    if mesh_info.M is None:
        mesh_info.assemble_mass_matrix()
    return np.asarray(mesh_info.M @ bf_vec).ravel()


def setup_boundary_conditions(mesh_info, geo_l, geo_h, F_tot):
    """
    Setup boundary conditions for the FGM plane stress problem.

    Args:
        mesh_info: MeshInfo object
        geo_l: Geometry length
        geo_h: Geometry height
        F_tot: Total force magnitude

    Returns:
        bc_info: Dictionary containing boundary conditions
    """
    coord = mesh_info.coord
    n_dof = mesh_info.n_dof

    fix_edge = np.where(coord[:, 0] == 0)[0]
    fix_dofs_x = 2 * fix_edge
    fix_dofs_y = np.array([2 * fix_edge[0] + 1])
    fix_dofs = np.sort(np.concatenate([fix_dofs_x, fix_dofs_y]))

    force = np.zeros(n_dof)
    load_points = np.where(coord[:, 0] == geo_l)[0]
    load_dofs = 2 * load_points

    F_density = F_tot / geo_h
    F_mag = F_density * mesh_info.eh

    force[load_dofs] = F_mag
    force[load_dofs[0]] -= F_mag / 2
    force[load_dofs[-1]] -= F_mag / 2

    return {
        'fixdof': fix_dofs,
        'force': force,
    }
