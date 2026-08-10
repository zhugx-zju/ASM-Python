"""Shared numerical case runners for forward and inverse grid studies."""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from fgm_asm import (
    MaterialInfo,
    MeshInfo,
    add_noise_to_displacement,
    fem_assemble,
    forward_solver,
    generate_fgm_modulus,
    lbfgs_inverse_solver_scipy,
)
from fgm_asm.config_types import ForwardConfig, InverseConfig
from fgm_asm.mesh import setup_boundary_conditions
from grid_study.demo_data import generate_demo_case


def field_from_displacement(mesh_info: MeshInfo, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert an interleaved nodal displacement vector to [y, x] fields."""
    ux = np.asarray(U[0::2], dtype=float)
    uy = np.asarray(U[1::2], dtype=float)
    return (
        ux.reshape(mesh_info.nods_y, mesh_info.nods_x, order="C"),
        uy.reshape(mesh_info.nods_y, mesh_info.nods_x, order="C"),
    )


def interpolate_field(field: np.ndarray, source_mesh: MeshInfo, target_mesh: MeshInfo) -> np.ndarray:
    """Interpolate a structured nodal field to another structured mesh."""
    x_source = np.asarray(source_mesh.plot_x[0, :], dtype=float)
    y_source = np.asarray(source_mesh.plot_y[:, 0], dtype=float)
    interpolator = RegularGridInterpolator(
        (y_source, x_source), np.asarray(field, dtype=float), bounds_error=True
    )
    points = np.column_stack((target_mesh.Y, target_mesh.X))
    return interpolator(points).reshape(target_mesh.nods_y, target_mesh.nods_x)


def measure_case(func, repeats: int = 5):
    """Measure steady-state wall time and Python allocation peak separately."""
    if repeats < 1:
        raise ValueError(f"repeats must be positive, got {repeats}")

    func()
    timings = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = func()
        timings.append(time.perf_counter() - start)

    tracemalloc.start()
    try:
        func()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return value, float(np.median(timings)), float(peak_bytes / 1024**2)


def make_grf_reference(forward_config: ForwardConfig, nodes: int) -> dict:
    """Generate a reproducible GRF reference using an explicit configuration."""
    mesh = MeshInfo(
        forward_config.geo_l,
        forward_config.geo_h,
        int(nodes) - 1,
        int(nodes) - 1,
    )
    E_field, _ = generate_fgm_modulus(
        mesh,
        dis_type="grf",
        Ex=forward_config.Ex,
        Ey=forward_config.Ey,
        nu=forward_config.nu,
        grf_E_max=forward_config.grf_E_max,
        grf_sigma_g=forward_config.grf_sigma_g,
        grf_ell=forward_config.grf_ell,
        grf_seed=forward_config.grf_seed,
    )
    return {"mesh": mesh, "E_field": E_field}


def run_forward_case(
    dis_type: str,
    nodes: int,
    forward_config: ForwardConfig,
    grf_reference: dict | None = None,
    demo_data: dict | None = None,
) -> dict:
    """Run one measured forward case with an explicit forward configuration."""
    if nodes < 2:
        raise ValueError(f"nodes must be at least 2, got {nodes}")

    mesh = MeshInfo(
        forward_config.geo_l,
        forward_config.geo_h,
        int(nodes) - 1,
        int(nodes) - 1,
    )

    def solve():
        if demo_data is not None:
            demo_case = generate_demo_case(demo_data, mesh, forward_config)
            E_field = demo_case["E_field"]
            material_info = demo_case["material_info"]
        elif dis_type == "grf" and grf_reference is not None:
            material_info = MaterialInfo(nu=forward_config.nu, dis_type="grf")
            E_field = interpolate_field(grf_reference["E_field"], grf_reference["mesh"], mesh)
        else:
            E_field, material_info = generate_fgm_modulus(
                mesh,
                dis_type=dis_type,
                Ex=forward_config.Ex,
                Ey=forward_config.Ey,
                nu=forward_config.nu,
                grf_E_max=forward_config.grf_E_max,
                grf_sigma_g=forward_config.grf_sigma_g,
                grf_ell=forward_config.grf_ell,
                grf_seed=forward_config.grf_seed,
            )
        material_info.update(E_field.ravel(order="C"), iteration=1)
        bc_info = setup_boundary_conditions(
            mesh, forward_config.geo_l, forward_config.geo_h, forward_config.f_tot
        )
        fem_info = fem_assemble(mesh, material_info, bc_info)
        U = forward_solver(fem_info)
        return mesh, bc_info, E_field, U

    (mesh, bc_info, E_field, U), elapsed, peak_python_mb = measure_case(solve)
    ux, uy = field_from_displacement(mesh, U)
    case_config = forward_config.to_dict()
    case_config["nel_x"] = int(nodes - 1)
    case_config["nel_y"] = int(nodes - 1)
    if demo_data is not None:
        spec = demo_data["spec"]
        case_config["sample_index"] = int(spec["sample_index"])
        case_config["source_sample_index"] = int(spec["source_sample_index"])
        case_config["distribution_parameters"] = spec["parameters"]
    return {
        "dataset": dis_type,
        "nodes": int(nodes),
        "elements": int(nodes - 1),
        "mesh": mesh,
        "bc_info": bc_info,
        "E_field": np.asarray(E_field, dtype=float),
        "U": np.asarray(U, dtype=float),
        "ux": ux,
        "uy": uy,
        "elapsed_time_seconds": elapsed,
        "peak_python_memory_mb": peak_python_mb,
        "config": case_config,
    }


def run_inverse_case(
    case: dict,
    noise_percentage: float,
    gamma: float,
    inverse_config: InverseConfig,
    max_iter: int | None = None,
    U_measured: np.ndarray | None = None,
) -> tuple[dict, np.ndarray, float, float]:
    """Run one inverse case using an explicit inverse configuration."""
    mesh = case["mesh"]
    if mesh.M is None:
        mesh.assemble_mass_matrix()
    noise_level = float(noise_percentage) / 100.0
    if U_measured is None:
        U_measured = add_noise_to_displacement(case["U"], noise_level, seed=42)
    else:
        U_measured = np.asarray(U_measured, dtype=float)

    solve_max_iter = inverse_config.max_iter if max_iter is None else int(max_iter)
    tracemalloc.start()
    start = time.perf_counter()
    try:
        result = lbfgs_inverse_solver_scipy(
            mesh_info=mesh,
            bc_info=case["bc_info"],
            U_measured=U_measured,
            E_init=None,
            gamma=gamma,
            E_min=inverse_config.E_min,
            E_max=inverse_config.E_max,
            max_iter=solve_max_iter,
            ftol=inverse_config.ftol,
            gtol=inverse_config.gtol,
            nu=inverse_config.nu,
        )
    finally:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return result, U_measured, float(time.perf_counter() - start), float(peak_bytes / 1024**2)


__all__ = [
    "field_from_displacement",
    "interpolate_field",
    "make_grf_reference",
    "measure_case",
    "run_forward_case",
    "run_inverse_case",
]
