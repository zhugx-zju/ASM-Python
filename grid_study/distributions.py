"""Parameter-driven demo material distributions for grid studies.

The demo samples originate from the MATLAB batch generators in the parent
U-Net project.  The fixed-test-set index used by ``asm_unet_compare`` is kept
separate from the original batch index so that a sample is never mistaken for
an inverse parameter or a random seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from fgm_asm import MaterialInfo, MeshInfo


@dataclass(frozen=True)
class DistributionSpec:
    """Source parameters for one selected demo sample."""

    dataset: str
    sample_index: int
    source_sample_index: int
    source_nodes: int
    distribution: str
    alpha: float | None = None
    beta: float | None = None
    e_max: float | None = None
    sigma_g: float | None = None
    ell: float | None = None
    jitter: float = 1e-6


def _interpolate(field: np.ndarray, source_mesh: MeshInfo, target_mesh: MeshInfo) -> np.ndarray:
    x_source = np.asarray(source_mesh.plot_x[0, :], dtype=float)
    y_source = np.asarray(source_mesh.plot_y[:, 0], dtype=float)
    interpolator = RegularGridInterpolator(
        (y_source, x_source), np.asarray(field, dtype=float), bounds_error=True
    )
    points = np.column_stack((target_mesh.Y, target_mesh.X))
    return interpolator(points).reshape(target_mesh.nods_y, target_mesh.nods_x)


def _source_mesh(demo_data: dict) -> MeshInfo:
    return demo_data["mesh"]


def _spec_from_metadata(demo_data: dict) -> DistributionSpec:
    metadata = demo_data.get("spec", demo_data.get("metadata", {}))
    parameters = metadata.get("parameters", {})
    distribution = str(
        metadata.get("dataset", metadata.get("source_distribution", demo_data["dataset"]))
    ).strip().lower()
    return DistributionSpec(
        dataset=str(demo_data["dataset"]).strip().lower(),
        sample_index=int(metadata["sample_index"]),
        source_sample_index=int(metadata["source_sample_index"]),
        source_nodes=int(metadata.get("source_nodes", 40)),
        distribution=distribution,
        alpha=parameters.get("alpha"),
        beta=parameters.get("beta"),
        e_max=parameters.get("E_max"),
        sigma_g=parameters.get("sigma_g"),
        ell=parameters.get("ell"),
        jitter=float(parameters.get("jitter", 1e-6)),
    )


def get_distribution_spec(demo_data: dict) -> DistributionSpec:
    """Return the source parameter record for a loaded demo case."""
    return _spec_from_metadata(demo_data)


def _generate_linear_or_exponential(
    spec: DistributionSpec,
    mesh: MeshInfo,
    nu: float,
) -> tuple[np.ndarray, MaterialInfo]:
    if spec.alpha is None or spec.beta is None:
        raise ValueError(f"Missing alpha/beta parameters for {spec.dataset} demo")
    material = MaterialInfo(
        nu=nu,
        dis_type=spec.distribution,
        alpha=float(spec.alpha),
        beta=float(spec.beta),
    )
    field = material.get_modulus_field(mesh.coord).reshape(mesh.nods_y, mesh.nods_x)
    return field, material


def _generate_grf_from_reference(
    demo_data: dict,
    spec: DistributionSpec,
    mesh: MeshInfo,
    nu: float,
) -> tuple[np.ndarray, MaterialInfo]:
    """Evaluate the selected GRF realization on another structured mesh.

    The original batch generator stores a separate Gaussian vector for each
    sample and mesh.  Only its 40 x 40 realization is available in this repo,
    so the realization is transported in latent Gaussian space.  This keeps
    ``E_max``, ``sigma_g``, ``ell`` and the selected sample fixed while avoiding
    interpolation of the modulus field itself.  The source mesh reproduces the
    imported 40 x 40 field exactly up to floating-point roundoff.
    """
    if spec.e_max is None or spec.sigma_g is None or spec.ell is None:
        raise ValueError(f"Missing GRF parameters for {spec.dataset} demo")
    reference = np.asarray(demo_data["E_field"], dtype=float)
    normalized = np.clip(2.0 * reference / float(spec.e_max) - 1.0, -1.0 + 1e-12, 1.0 - 1e-12)
    latent = np.arctanh(normalized) / float(spec.sigma_g)
    latent_target = _interpolate(latent, _source_mesh(demo_data), mesh)
    field = float(spec.e_max) * (
        np.tanh(float(spec.sigma_g) * latent_target) + 1.0
    ) / 2.0
    return field, MaterialInfo(nu=nu, dis_type="grf")


def generate_demo_modulus(
    demo_data: dict,
    mesh: MeshInfo,
    nu: float,
) -> tuple[np.ndarray, MaterialInfo]:
    """Generate one demo sample's modulus field on ``mesh`` from its parameters."""
    spec = get_distribution_spec(demo_data)
    if spec.distribution in {"bil", "exp"}:
        return _generate_linear_or_exponential(spec, mesh, nu)
    if spec.distribution == "grf":
        return _generate_grf_from_reference(demo_data, spec, mesh, nu)
    raise ValueError(f"Unsupported demo distribution: {spec.distribution!r}")


def validate_demo_modulus(
    demo_data: dict,
    nu: float,
    atol: float = 5e-7,
    rtol: float = 1e-7,
) -> dict:
    """Compare parameter-generated and imported fields on the source mesh."""
    generated, _ = generate_demo_modulus(demo_data, _source_mesh(demo_data), nu=nu)
    reference = np.asarray(demo_data["E_field"], dtype=float)
    difference = generated - reference
    scale = max(float(np.max(np.abs(reference))), 1.0)
    max_abs = float(np.max(np.abs(difference)))
    return {
        "dataset": demo_data["dataset"],
        "sample_index": int(get_distribution_spec(demo_data).sample_index),
        "source_sample_index": int(get_distribution_spec(demo_data).source_sample_index),
        "source_sample_number_matlab": int(
            demo_data.get("spec", {}).get("source_sample_number_matlab", 0)
        ),
        "max_absolute_difference": max_abs,
        "relative_max_difference": max_abs / scale,
        "matches": bool(np.allclose(generated, reference, atol=atol, rtol=rtol)),
    }


__all__ = [
    "DistributionSpec",
    "generate_demo_modulus",
    "get_distribution_spec",
    "validate_demo_modulus",
]
