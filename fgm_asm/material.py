"""
Material property management module for FGM problems.
Handles material information updates during iterations.
"""

import numpy as np


class MaterialInfo:
    """
    Material information class for FGM.
    Stores material properties and modulus history during iterations.
    """

    def __init__(self, nu, dis_type='exp', alpha=0.0, beta=0.0, store_history=False):
        """
        Initialize material information.

        Args:
            nu: Poisson's ratio
            dis_type: Distribution type ('bil' for bilinear, 'exp' for exponential)
            alpha: Material gradient parameter in x direction
            beta: Material gradient parameter in y direction
        """
        self.nu = nu
        self.dis_type = dis_type
        self.alpha = alpha
        self.beta = beta
        self.store_history = store_history

        # Iteration tracking
        self.iter = 1
        self.E_ini_vec = None
        self.current_E_vec = None
        self.E_his_vec = {}

    def update(self, iter_E_vec, iteration=None):
        """
        Update modulus values after iteration.

        Args:
            iter_E_vec: Modulus vector [n_nod]
            iteration: Iteration number (if None, uses current iteration)
        """
        if iteration is None:
            iteration = self.iter

        iter_E_vec = np.asarray(iter_E_vec)

        if self.E_ini_vec is None:
            self.E_ini_vec = iter_E_vec.copy()

        self.iter = iteration
        self.current_E_vec = iter_E_vec
        if self.store_history:
            self.E_his_vec[iteration] = iter_E_vec.copy()

    def get_current_modulus(self):
        """
        Get current iteration modulus vector.

        Returns:
            E_vec: Current modulus vector [n_nod]
        """
        if self.current_E_vec is not None:
            return self.current_E_vec
        return self.E_his_vec[self.iter]

    def get_modulus_field(self, coord):
        """
        Generate modulus field from coordinates based on distribution type.

        Args:
            coord: Node coordinates [n_nod, 2]

        Returns:
            E_field: Modulus field at nodes [n_nod]
        """
        X = coord[:, 0]
        Y = coord[:, 1]

        if self.dis_type == 'bil':
            # Bilinear distribution: E(x,y) = 1 + alpha*x + beta*y
            E_field = 1.0 + self.alpha * X + self.beta * Y
        elif self.dis_type == 'exp':
            # Exponential distribution: E(x,y) = exp(alpha*x + beta*y)
            E_field = np.exp(self.alpha * X + self.beta * Y)
        else:
            raise ValueError(f"Unknown distribution type: {self.dis_type}")

        return E_field

    def get_elasticity_matrix(self):
        """
        Get plane stress elasticity matrix D0 (without modulus).

        Returns:
            D0: Elasticity matrix [3, 3]
        """
        nu = self.nu
        D0 = np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - nu)]
        ]) / (1.0 - nu**2)

        return D0


def generate_fgm_modulus(mesh_info, dis_type='exp', Ex=2.0, Ey=2.0):
    """
    Generate FGM modulus field for forward problem.

    Args:
        mesh_info: MeshInfo object
        dis_type: Distribution type ('bil' or 'exp')
        Ex: Target modulus ratio at x = geo_l
        Ey: Target modulus ratio at y = geo_h

    Returns:
        E_field: Modulus field at nodes
        material_info: MaterialInfo object
    """
    geo_l = mesh_info.geo_l
    geo_h = mesh_info.geo_h

    # Calculate alpha and beta
    if dis_type == 'bil':
        alpha = (Ex - 1.0) / geo_l
        beta = (Ey - 1.0) / geo_h
    else:  # 'exp'
        alpha = np.log(Ex) / geo_l
        beta = np.log(Ey) / geo_h

    # Create material info
    material_info = MaterialInfo(nu=0.3, dis_type=dis_type, alpha=alpha, beta=beta)

    # Generate modulus field
    E_field = material_info.get_modulus_field(mesh_info.coord)

    # Reshape to the plotting grid [nods_y, nods_x].
    # Mesh nodes are stored row-by-row in increasing y, then x.
    E_field_2d = E_field.reshape(mesh_info.nods_y, mesh_info.nods_x)

    return E_field_2d, material_info
