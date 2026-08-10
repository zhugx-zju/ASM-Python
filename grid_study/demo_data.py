"""Load compact demo distributions and their source-generation metadata.

The imported 40 x 40 arrays are retained as reference data. Grid studies use
the parameter record in each case's metadata to generate the target modulus
field, while the measured displacement arrays are available for validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from fgm_asm import MeshInfo
from fgm_asm.config_types import ForwardConfig
from grid_study.distributions import generate_demo_modulus, get_distribution_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_ROOT = REPO_ROOT / "demo_distributions"
DEFAULT_DEMO_SPEC = DEFAULT_DEMO_ROOT / "demo_cases.json"
DEMO_NODES = 40
DEMO_CASES = {
    "bil": "sample_600",
    "exp": "sample_200",
    "grf": "sample_410",
}


def _as_scalar(value):
    """Convert a NumPy scalar to a JSON-friendly Python scalar."""
    return value.item() if hasattr(value, "item") else value


def _source_mesh(spec: dict) -> MeshInfo:
    """Create the source mesh from the imported case metadata."""
    parameters = spec.get("parameters", {})
    geometry = parameters.get("geometry")
    if geometry is None or len(geometry) != 2:
        raise ValueError("Demo metadata must contain parameters.geometry=[length, height]")
    source_nodes = int(spec.get("source_nodes", DEMO_NODES))
    return MeshInfo(
        float(geometry[0]),
        float(geometry[1]),
        source_nodes - 1,
        source_nodes - 1,
    )


def _interpolate_array(field: np.ndarray, source_mesh: MeshInfo, target_mesh: MeshInfo) -> np.ndarray:
    x_source = np.asarray(source_mesh.plot_x[0, :], dtype=float)
    y_source = np.asarray(source_mesh.plot_y[:, 0], dtype=float)
    interpolator = RegularGridInterpolator(
        (y_source, x_source), np.asarray(field, dtype=float), bounds_error=True
    )
    points = np.column_stack((target_mesh.Y, target_mesh.X))
    return interpolator(points).reshape(target_mesh.nods_y, target_mesh.nods_x)


def _interleave(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (np.asarray(ux, dtype=float).ravel(order="C"), np.asarray(uy, dtype=float).ravel(order="C"))
    ).ravel()


def load_demo_dataset(
    dataset: str,
    root: Path | str = DEFAULT_DEMO_ROOT,
) -> dict:
    """Load one imported demo distribution and all available noise cases."""
    normalized = str(dataset).strip().lower()
    if normalized not in DEMO_CASES:
        raise ValueError(f"Unsupported demo dataset: {dataset!r}")

    root = Path(root)
    case_dir = root / normalized / DEMO_CASES[normalized]
    spec_path = root / "demo_cases.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"Demo parameter file was not found: {spec_path}")
    demo_manifest = json.loads(spec_path.read_text(encoding="utf-8"))
    try:
        spec = demo_manifest["cases"][normalized]
    except KeyError as exc:
        raise KeyError(f"Demo parameter file has no case for {normalized!r}") from exc
    target_path = case_dir / "true_modulus.npy"
    metadata_path = case_dir / "metadata.json"
    if not target_path.exists():
        raise FileNotFoundError(
            f"Demo data for {normalized!r} was not found at {case_dir}. "
            "The repository must contain its local demo_distributions package."
        )

    target = np.asarray(np.load(target_path), dtype=float)
    if target.shape != (DEMO_NODES, DEMO_NODES):
        raise ValueError(f"Expected a {DEMO_NODES}x{DEMO_NODES} modulus field, got {target.shape}")

    noise_cases = {}
    for displacement_path in sorted(case_dir.glob("noise_*/measured_displacement.npz")):
        with np.load(displacement_path) as data:
            ux = np.asarray(data["ux"], dtype=float)
            uy = np.asarray(data["uy"], dtype=float)
            if ux.shape != target.shape or uy.shape != target.shape:
                raise ValueError(f"Displacement shape mismatch in {displacement_path}")
            if "U_measured_C" in data:
                measured = np.asarray(data["U_measured_C"], dtype=float)
            else:
                measured = _interleave(ux, uy)
            if "noise_percentage" in data:
                noise_percentage = float(_as_scalar(data["noise_percentage"]))
                noise = noise_percentage / 100.0
            else:
                noise = float(_as_scalar(data["noise_level"]))
                noise_percentage = 100.0 * noise
            noise_cases[noise_percentage] = {
                "ux": ux,
                "uy": uy,
                "U_measured": measured,
                "noise_percentage": noise_percentage,
                "noise_level": noise,
                "sample_index": _as_scalar(data["sample_index"]),
                "sample_seed": _as_scalar(data["sample_seed"]),
            }

    if not noise_cases:
        raise FileNotFoundError(f"No measured displacement files were found below {case_dir}")

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "dataset": normalized,
        "case_dir": case_dir,
        "mesh": _source_mesh(spec),
        "E_field": target,
        "noise_cases": noise_cases,
        "metadata": metadata,
        "spec": spec,
        "spec_path": spec_path,
    }


def generate_demo_case(
    demo_data: dict,
    target_mesh: MeshInfo,
    forward_config: ForwardConfig,
) -> dict:
    """Generate one selected demo sample on ``target_mesh``.

    This is the production path for grid studies. It does not interpolate the
    imported modulus field for BIL/EXP; those cases are evaluated from their
    stored distribution coefficients. GRF recovers an equivalent latent
    realization from the stored 40 x 40 field and evaluates its MATLAB-style
    RBF representation on each target mesh because the original per-sample
    random vector is not included in the compact demo package.
    """
    E_field, material_info = generate_demo_modulus(
        demo_data,
        target_mesh,
        nu=forward_config.nu,
    )
    spec = get_distribution_spec(demo_data)
    return {
        "E_field": E_field,
        "material_info": material_info,
        "sample_index": spec.sample_index,
        "source_sample_index": spec.source_sample_index,
        "distribution_spec": spec,
    }


def interpolate_demo_case(demo_data: dict, target_mesh: MeshInfo, noise_percentage: float = 0.0) -> dict:
    """Interpolate imported reference arrays for validation/inspection only.

    Production forward and inverse grid studies use :func:`generate_demo_case`
    instead, so this helper is intentionally kept out of those workflows.
    """
    noise_key = float(noise_percentage)
    available = demo_data["noise_cases"]
    if noise_key not in available:
        available_text = ", ".join(f"{key:g}" for key in sorted(available))
        raise KeyError(f"Noise level {noise_key:g}% is unavailable; available levels: {available_text}")
    source_noise = available[noise_key]
    ux = _interpolate_array(source_noise["ux"], demo_data["mesh"], target_mesh)
    uy = _interpolate_array(source_noise["uy"], demo_data["mesh"], target_mesh)
    return {
        "E_field": _interpolate_array(demo_data["E_field"], demo_data["mesh"], target_mesh),
        "ux": ux,
        "uy": uy,
        "U_measured": _interleave(ux, uy),
        "noise_percentage": noise_key,
        "sample_index": source_noise["sample_index"],
        "sample_seed": source_noise["sample_seed"],
    }
