"""
Tikhonov regularization module for inverse problems.
"""

import numpy as np


def get_tikhonov_regularization(mesh_info, material_info):
    """
    Calculate the Tikhonov regularization value.

    Args:
        mesh_info: MeshInfo object
        material_info: MaterialInfo object

    Returns:
        tik_reg: Tikhonov regularization value
    """
    L_reg = mesh_info.get_regularization_matrix()
    E_vec = material_info.get_current_modulus()
    return float(E_vec @ (L_reg @ E_vec))


def get_tikhonov_gradient(mesh_info, material_info):
    """
    Calculate the gradient of the Tikhonov regularization with respect to modulus.

    Args:
        mesh_info: MeshInfo object
        material_info: MaterialInfo object

    Returns:
        tik_reg_grad: Gradient vector [n_nod]
    """
    L_reg = mesh_info.get_regularization_matrix()
    E_vec = material_info.get_current_modulus()
    return 2.0 * np.asarray(L_reg @ E_vec).ravel()
