"""
Visualization and plotting helpers for FGM forward and inverse workflows.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


# Set global font to Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Times New Roman'
rcParams['mathtext.it'] = 'Times New Roman:italic'
rcParams['mathtext.bf'] = 'Times New Roman:bold'


def _coerce_save_path(save_path):
    """Convert an optional save path into a ``Path``."""
    if save_path is None:
        return None
    return Path(save_path)


def _extract_displacement_fields(mesh_info, U):
    """Reshape the nodal displacement vector into plotting fields."""
    ux = reshape_nodal_values_for_plot(mesh_info, U[0::2])
    uy = reshape_nodal_values_for_plot(mesh_info, U[1::2])
    return ux, uy


def _save_figure(fig, save_path, stem, formats=('png', 'pdf'), dpi=1200):
    """Save a figure when an output directory is provided."""
    save_path = _coerce_save_path(save_path)
    if save_path is None:
        return

    for ext in formats:
        kwargs = {'bbox_inches': 'tight'}
        if ext in {'png', 'pdf'}:
            kwargs['dpi'] = dpi
        fig.savefig(save_path / f'{stem}.{ext}', **kwargs)


def create_smooth_contour(x, y, z, ax, levels=100, cmap='viridis', colorbar_label=None):
    """
    Create smooth contour plot using ``pcolormesh`` for better smoothness.

    Args:
        x: X coordinates (2D mesh)
        y: Y coordinates (2D mesh)
        z: Z values (2D mesh)
        ax: Matplotlib axis
        levels: Kept for compatibility with older callers
        cmap: Colormap
        colorbar_label: Kept for compatibility with older callers

    Returns:
        Plot object
    """
    del levels, colorbar_label
    return ax.pcolormesh(x, y, z, cmap=cmap, shading='gouraud')


def reshape_nodal_values_for_plot(mesh_info, nodal_values):
    """
    Map nodal values back to the plotting grid using node coordinates.

    This avoids relying on reshape/transposition assumptions and guarantees
    that nodal vectors and modulus fields use the same coordinate layout.

    Args:
        mesh_info: MeshInfo object
        nodal_values: Nodal values [n_nod]

    Returns:
        field_2d: Values arranged on the plotting grid [nods_y, nods_x]
    """
    field_2d = np.empty_like(mesh_info.plot_x, dtype=np.asarray(nodal_values).dtype)
    x_idx = np.rint(mesh_info.X / mesh_info.el).astype(int)
    y_idx = np.rint(mesh_info.Y / mesh_info.eh).astype(int)
    field_2d[y_idx, x_idx] = nodal_values
    return field_2d


def plot_modulus_distribution(mesh_info, E_field, save_path=None):
    """
    Plot modulus distribution in a single figure.

    When save_path is provided, this also exports a clean version without
    title or colorbar in PNG and SVG formats.

    Args:
        mesh_info: MeshInfo object
        E_field: Modulus field [nods_y, nods_x]
        save_path: Path to save figure (optional)

    Returns:
        fig: The matplotlib figure
    """
    fig = plt.figure(figsize=(8, 7))
    ax = plt.subplot(1, 1, 1)

    im = create_smooth_contour(mesh_info.plot_x, mesh_info.plot_y, E_field, ax, cmap='viridis')
    ax.set_title('Modulus Distribution E(x,y)', fontsize=15, fontweight='bold')
    ax.axis('equal')
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.0405, pad=0.04)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path, 'modulus_distribution')

        fig_clean = plt.figure(figsize=(8, 7))
        ax_clean = plt.subplot(1, 1, 1)
        create_smooth_contour(mesh_info.plot_x, mesh_info.plot_y, E_field, ax_clean, cmap='viridis')
        ax_clean.axis('equal')
        ax_clean.axis('off')
        plt.tight_layout()
        _save_figure(
            fig_clean,
            save_path,
            'modulus_distribution_plain',
            formats=('png', 'svg'),
        )
        plt.close(fig_clean)
        print(f"  Modulus distribution figure saved to {_coerce_save_path(save_path)}")

    return fig


def plot_displacement_fields(mesh_info, U, save_path=None):
    """
    Plot displacement fields in one figure with two subplots.

    Args:
        mesh_info: MeshInfo object
        U: Displacement vector [n_dof]
        save_path: Path to save figure (optional)

    Returns:
        fig: The matplotlib figure
    """
    ux, uy = _extract_displacement_fields(mesh_info, U)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    im1 = create_smooth_contour(mesh_info.plot_x, mesh_info.plot_y, ux, axes[0], cmap='viridis')
    axes[0].set_title('X-Displacement $u_x$', fontsize=15, fontweight='bold')
    axes[0].axis('equal')
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.0405, pad=0.04)
    cbar1.ax.tick_params(labelsize=11)

    im2 = create_smooth_contour(mesh_info.plot_x, mesh_info.plot_y, uy, axes[1], cmap='viridis')
    axes[1].set_title('Y-Displacement $u_y$', fontsize=15, fontweight='bold')
    axes[1].axis('equal')
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.0405, pad=0.04)
    cbar2.ax.tick_params(labelsize=11)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path, 'displacement_fields')
        print(f"  Displacement fields figure saved to {_coerce_save_path(save_path)}")

    return fig


def plot_single_displacement_field(mesh_info, U, component='ux', save_path=None):
    """
    Plot a single displacement component in its own figure without
    title or colorbar.

    Args:
        mesh_info: MeshInfo object
        U: Displacement vector [n_dof]
        component: Displacement component to plot ('ux' or 'uy')
        save_path: Path to save figure (optional)

    Returns:
        fig: The matplotlib figure
    """
    ux, uy = _extract_displacement_fields(mesh_info, U)

    component = component.lower()
    if component == 'ux':
        field = ux
        filename = 'displacement_ux'
    elif component == 'uy':
        field = uy
        filename = 'displacement_uy'
    else:
        raise ValueError("component must be 'ux' or 'uy'")

    fig = plt.figure(figsize=(8, 7))
    ax = plt.subplot(1, 1, 1)

    create_smooth_contour(mesh_info.plot_x, mesh_info.plot_y, field, ax, cmap='viridis')
    ax.axis('equal')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path, filename, formats=('png', 'pdf', 'svg'))
        print(f"  Single displacement figure saved to {_coerce_save_path(save_path) / filename}")

    return fig


def plot_reconstruction_results(mesh_info, E_true, E_reconstructed, errors,
                                noise_level, save_path=None, filename_stem=None):
    """
    Plot reconstruction results in one figure with three subplots.

    Args:
        mesh_info: MeshInfo object
        E_true: True modulus field [nods_y, nods_x]
        E_reconstructed: Reconstructed modulus field [n_nod]
        errors: Error metrics dictionary
        noise_level: Noise level used
        save_path: Path to save figure (optional)
        filename_stem: Optional output filename stem without extension

    Returns:
        fig: The matplotlib figure
    """
    E_recon_2d = reshape_nodal_values_for_plot(mesh_info, E_reconstructed)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im1 = create_smooth_contour(mesh_info.plot_x, mesh_info.plot_y, E_true, axes[0], cmap='viridis')
    axes[0].set_title('True Modulus Distribution', fontsize=15, fontweight='bold')
    axes[0].axis('equal')
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.0405, pad=0.04)
    cbar1.ax.tick_params(labelsize=11)

    im2 = create_smooth_contour(mesh_info.plot_x, mesh_info.plot_y, E_recon_2d, axes[1], cmap='viridis')
    axes[1].set_title(
        f'Reconstructed Modulus (Noise: {noise_level*100:.0f}%)',
        fontsize=15,
        fontweight='bold',
    )
    axes[1].axis('equal')
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.0405, pad=0.04)
    cbar2.ax.tick_params(labelsize=11)

    error_field = reshape_nodal_values_for_plot(mesh_info, errors['rel_error_field'])
    im3 = create_smooth_contour(mesh_info.plot_x, mesh_info.plot_y, error_field, axes[2], cmap='hot')
    axes[2].set_title(
        f'Relative Error (MAE: {errors["mae"]:.2f}%, RMSE: {errors["rmse"]:.4f})',
        fontsize=15,
        fontweight='bold',
    )
    axes[2].axis('equal')
    axes[2].axis('off')
    cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.0405, pad=0.04)
    cbar3.set_label('Error (%)', fontsize=12)
    cbar3.ax.tick_params(labelsize=11)

    plt.tight_layout()

    if save_path:
        stem = filename_stem or f'reconstruction_results_noise_{noise_level*100:.0f}pct'
        _save_figure(fig, save_path, stem)
        print(f"  Reconstruction results figure saved to {_coerce_save_path(save_path)}")

    return fig


def plot_iteration_history(results, save_path=None, noise_level=0.0, filename_stem=None):
    """
    Plot iteration history in one figure with three subplots.

    Args:
        results: Optimization results dictionary
        save_path: Path to save figure (optional)
        noise_level: Noise level used (for filename)
        filename_stem: Optional output filename stem without extension

    Returns:
        fig: The matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    iterations = np.arange(len(results['cost_history']))
    cond_h_history = results.get('cond_H_history')
    if cond_h_history is None:
        cond_h_history = np.ones(len(results['grad_norm_history']))
    else:
        cond_h_history = np.asarray(cond_h_history)

    axes[0].semilogy(iterations, results['cost_history'], 'b-', linewidth=2.5, label='Total Cost')
    axes[0].semilogy(iterations, results['cost_tar_history'], 'r--', linewidth=2, label='Data Misfit')
    axes[0].semilogy(iterations, results['cost_reg_history'], 'g-.', linewidth=2, label='Regularization')
    axes[0].set_xlabel('Iteration', fontsize=13)
    axes[0].set_ylabel('Objective Function Value', fontsize=13)
    axes[0].set_title('Convergence History', fontsize=15, fontweight='bold')
    axes[0].legend(fontsize=11, loc='best')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].tick_params(labelsize=11)

    axes[1].semilogy(iterations, results['grad_norm_history'], 'k-', linewidth=2.5)
    axes[1].set_xlabel('Iteration', fontsize=13)
    axes[1].set_ylabel('Gradient Norm', fontsize=13)
    axes[1].set_title('Gradient Norm Evolution', fontsize=15, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].tick_params(labelsize=11)

    if not np.all(cond_h_history == 1):
        axes[2].semilogy(iterations, cond_h_history, 'm-', linewidth=2.5)
        axes[2].set_ylabel('Condition Number', fontsize=13)
        axes[2].set_title('Approximate Hessian Condition Number', fontsize=15, fontweight='bold')
    elif len(results['grad_norm_history']) > 0:
        relative_grad_norm = results['grad_norm_history'] / (results['grad_norm_history'][0] + 1e-15)
        axes[2].semilogy(iterations, relative_grad_norm, 'm-', linewidth=2.5)
        axes[2].set_ylabel('Relative Gradient Norm', fontsize=13)
        axes[2].set_title('Relative Gradient Norm Evolution', fontsize=15, fontweight='bold')
    else:
        axes[2].text(
            0.5,
            0.5,
            'No gradient history available',
            ha='center',
            va='center',
            transform=axes[2].transAxes,
        )
        axes[2].set_ylabel('Relative Gradient Norm', fontsize=13)
        axes[2].set_title('Relative Gradient Norm Evolution', fontsize=15, fontweight='bold')

    axes[2].set_xlabel('Iteration', fontsize=13)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].tick_params(labelsize=11)

    plt.tight_layout()

    if save_path:
        stem = filename_stem or f'iteration_history_noise_{noise_level*100:.0f}pct'
        _save_figure(fig, save_path, stem)
        print(f"  Iteration history figure saved to {_coerce_save_path(save_path)}")

    return fig


def plot_gradient_field(mesh_info, results, noise_level=0.0, save_path=None, filename_stem=None):
    """
    Plot gradient field in one figure with three subplots.

    Uses diverging colormap (RdBu_r) with zero centered at white/light color
    to better visualize positive and negative gradients.

    Args:
        mesh_info: MeshInfo object
        results: Optimization results dictionary
        noise_level: Noise level used (for filename)
        save_path: Path to save figure (optional)
        filename_stem: Optional output filename stem without extension

    Returns:
        fig: The matplotlib figure, or None if gradient data is not available
    """
    from matplotlib.colors import TwoSlopeNorm

    if 'final_gradient' not in results or results['final_gradient'] is None:
        print("  Warning: Gradient data not available in results")
        return None

    grad_total = reshape_nodal_values_for_plot(mesh_info, results['final_gradient'])
    grad_tar = reshape_nodal_values_for_plot(
        mesh_info, results.get('final_grad_tar', np.zeros_like(results['final_gradient']))
    )
    grad_reg = reshape_nodal_values_for_plot(
        mesh_info, results.get('final_grad_reg', np.zeros_like(results['final_gradient']))
    )

    vmax_total = np.nanpercentile(np.abs(grad_total), 99)
    vmax_tar_only = np.nanpercentile(np.abs(grad_tar), 99)
    vmax_reg_only = np.nanpercentile(np.abs(grad_reg), 99)
    vmax_shared = max(vmax_tar_only, vmax_reg_only)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    norm_total = TwoSlopeNorm(vmin=-vmax_total, vcenter=0.0, vmax=vmax_total)
    im1 = axes[0].pcolormesh(
        mesh_info.plot_x, mesh_info.plot_y, grad_total, shading='auto', cmap='RdBu_r', norm=norm_total
    )
    axes[0].set_title(r'Total Gradient $\nabla J$', fontsize=15, fontweight='bold')
    axes[0].axis('equal')
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.0405, pad=0.04)
    cbar1.ax.tick_params(labelsize=11)

    norm_tar = TwoSlopeNorm(vmin=-vmax_shared, vcenter=0.0, vmax=vmax_shared)
    im2 = axes[1].pcolormesh(
        mesh_info.plot_x, mesh_info.plot_y, grad_tar, shading='auto', cmap='RdBu_r', norm=norm_tar
    )
    axes[1].set_title(r'Data Misfit Gradient $\nabla J_{data}$', fontsize=15, fontweight='bold')
    axes[1].axis('equal')
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.0405, pad=0.04)
    cbar2.ax.tick_params(labelsize=11)

    norm_reg = TwoSlopeNorm(vmin=-vmax_shared, vcenter=0.0, vmax=vmax_shared)
    im3 = axes[2].pcolormesh(
        mesh_info.plot_x, mesh_info.plot_y, grad_reg, shading='auto', cmap='RdBu_r', norm=norm_reg
    )
    axes[2].set_title(r'Regularization Gradient $\nabla J_{reg}$', fontsize=15, fontweight='bold')
    axes[2].axis('equal')
    axes[2].axis('off')
    cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.0405, pad=0.04)
    cbar3.ax.tick_params(labelsize=11)

    plt.tight_layout()

    if save_path:
        stem = filename_stem or f'gradient_field_noise_{noise_level*100:.0f}pct'
        _save_figure(fig, save_path, stem)
        print(f"  Gradient field figure saved to {_coerce_save_path(save_path)}")

    return fig


def plot_reconstruction_comparison(mesh_info, E_true, scan_E_reconstructed,
                                   scan_errors, rerun_E_reconstructed, rerun_errors,
                                   noise_level, save_path=None, filename_stem='reconstruction_comparison'):
    """
    Plot a comparison between the L-curve scan optimum and the final rerun.

    Args:
        mesh_info: MeshInfo object
        E_true: True modulus field [nods_y, nods_x]
        scan_E_reconstructed: Reconstruction from the L-curve scan optimum [n_nod]
        scan_errors: Error dictionary for scan_E_reconstructed
        rerun_E_reconstructed: Reconstruction from the final rerun [n_nod]
        rerun_errors: Error dictionary for rerun_E_reconstructed
        noise_level: Noise level used
        save_path: Path to save figure (optional)
        filename_stem: Output filename stem without extension

    Returns:
        fig: The matplotlib figure
    """
    from matplotlib.colors import TwoSlopeNorm

    scan_field = reshape_nodal_values_for_plot(mesh_info, scan_E_reconstructed)
    rerun_field = reshape_nodal_values_for_plot(mesh_info, rerun_E_reconstructed)
    scan_error_field = reshape_nodal_values_for_plot(mesh_info, scan_errors['rel_error_field'])
    rerun_error_field = reshape_nodal_values_for_plot(mesh_info, rerun_errors['rel_error_field'])
    diff_field = rerun_field - scan_field

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    e_min = min(np.min(E_true), np.min(scan_field), np.min(rerun_field))
    e_max = max(np.max(E_true), np.max(scan_field), np.max(rerun_field))
    err_vmax = max(np.nanpercentile(scan_error_field, 99), np.nanpercentile(rerun_error_field, 99), 1e-12)
    diff_vmax = max(np.nanpercentile(np.abs(diff_field), 99), 1e-12)

    panels = [
        (axes[0, 0], E_true, 'True Modulus Distribution', 'viridis', {'vmin': e_min, 'vmax': e_max}),
        (
            axes[0, 1],
            scan_field,
            f'L-curve Scan Optimum\nMAE: {scan_errors["mae"]:.2f}%',
            'viridis',
            {'vmin': e_min, 'vmax': e_max},
        ),
        (
            axes[0, 2],
            rerun_field,
            f'Final Rerun\nMAE: {rerun_errors["mae"]:.2f}%',
            'viridis',
            {'vmin': e_min, 'vmax': e_max},
        ),
        (
            axes[1, 0],
            scan_error_field,
            f'Scan Error\nRMSE: {scan_errors["rmse"]:.4f}',
            'hot',
            {'vmin': 0.0, 'vmax': err_vmax},
        ),
        (
            axes[1, 1],
            rerun_error_field,
            f'Rerun Error\nRMSE: {rerun_errors["rmse"]:.4f}',
            'hot',
            {'vmin': 0.0, 'vmax': err_vmax},
        ),
        (
            axes[1, 2],
            diff_field,
            'Rerun - Scan',
            'RdBu_r',
            {'norm': TwoSlopeNorm(vmin=-diff_vmax, vcenter=0.0, vmax=diff_vmax)},
        ),
    ]

    for ax, field, title, cmap, kwargs in panels:
        im = ax.pcolormesh(mesh_info.plot_x, mesh_info.plot_y, field, shading='gouraud', cmap=cmap, **kwargs)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('equal')
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.0405, pad=0.04)
        cbar.ax.tick_params(labelsize=11)

    fig.suptitle(f'Reconstruction Comparison (Noise: {noise_level*100:.2f}%)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        _save_figure(fig, save_path, filename_stem)
        print(f"  Reconstruction comparison figure saved to {_coerce_save_path(save_path)}")

    return fig


def visualize_forward_results(mesh_info, E_field, U, config, save_path=None, show=True):
    """
    Visualize modulus distribution and displacement field for the forward problem.

    Args:
        mesh_info: MeshInfo object
        E_field: Modulus field [nods_y, nods_x]
        U: Displacement vector [n_dof]
        config: Unused retained argument for script compatibility
        save_path: Path to save figures (optional)
        show: Whether to display the figure interactively
    """
    del config
    ux, uy = _extract_displacement_fields(mesh_info, U)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im1 = create_smooth_contour(mesh_info.plot_x, mesh_info.plot_y, E_field, axes[0], cmap='viridis')
    axes[0].set_title('Modulus Distribution E(x,y)', fontsize=14, fontweight='bold')
    axes[0].axis('equal')
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.0405, pad=0.04)
    cbar1.ax.tick_params(labelsize=11)

    im2 = create_smooth_contour(mesh_info.plot_x, mesh_info.plot_y, ux, axes[1], cmap='viridis')
    axes[1].set_title('X-Displacement $u_x$', fontsize=14, fontweight='bold')
    axes[1].axis('equal')
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.0405, pad=0.04)
    cbar2.ax.tick_params(labelsize=11)

    im3 = create_smooth_contour(mesh_info.plot_x, mesh_info.plot_y, uy, axes[2], cmap='viridis')
    axes[2].set_title('Y-Displacement $u_y$', fontsize=14, fontweight='bold')
    axes[2].axis('equal')
    axes[2].axis('off')
    cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.0405, pad=0.04)
    cbar3.ax.tick_params(labelsize=11)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path, 'forward_results', dpi=600)
        print(f"Figure saved to {_coerce_save_path(save_path) / 'forward_results.png'}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_inverse_results(mesh_info, E_true, E_reconstructed, errors,
                              results, noise_level, save_path=None):
    """
    Visualize reconstruction results for the inverse problem.

    Creates three separate figures:
    1. Reconstruction results
    2. Iteration history
    3. Gradient field

    Args:
        mesh_info: MeshInfo object
        E_true: True modulus field [nods_y, nods_x]
        E_reconstructed: Reconstructed modulus field [n_nod]
        errors: Error metrics dictionary
        results: Optimization results
        noise_level: Noise level used
        save_path: Path to save figures
    """
    fig1 = plot_reconstruction_results(
        mesh_info,
        E_true,
        E_reconstructed,
        errors,
        noise_level,
        save_path=save_path,
        filename_stem=f'inverse_results_noise_{noise_level*100:.0f}pct',
    )
    fig2 = plot_iteration_history(
        results,
        save_path=save_path,
        noise_level=noise_level,
        filename_stem=f'iteration_history_noise_{noise_level*100:.0f}pct',
    )
    fig3 = plot_gradient_field(
        mesh_info,
        results,
        noise_level=noise_level,
        save_path=save_path,
        filename_stem=f'gradient_field_noise_{noise_level*100:.0f}pct',
    )
    return fig1, fig2, fig3
