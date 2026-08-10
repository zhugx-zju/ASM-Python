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
    batch_size: int | None = None
    shuffle_seed: int | None = None


def _rbf_matrix(
    left_coordinates: np.ndarray,
    right_coordinates: np.ndarray,
    ell: float,
) -> np.ndarray:
    differences = (
        np.asarray(left_coordinates, dtype=float)[:, None, :]
        - np.asarray(right_coordinates, dtype=float)[None, :, :]
    )
    squared_distance = np.sum(differences * differences, axis=2)
    return np.exp(-squared_distance / (2.0 * float(ell) ** 2))


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
        batch_size=parameters.get("batch_size"),
        shuffle_seed=parameters.get("shuffle_seed"),
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
    """Evaluate one fixed MATLAB-style GRF realization on ``mesh``.

    ``GRF_Generate.m`` forms a Gaussian field with a full RBF covariance
    matrix, then applies the tanh map.  The compact demo package stores the
    resulting source field, not the random vector.  The inverse tanh map and
    source covariance recover an equivalent realization, which is then
    evaluated on every target mesh with the same RBF coefficients.
    """
    if spec.e_max is None or spec.sigma_g is None or spec.ell is None:
        raise ValueError(f"Missing GRF parameters for {spec.dataset} demo")
    source_mesh = _source_mesh(demo_data)
    reference = np.asarray(demo_data["E_field"], dtype=float)
    normalized = np.clip(
        2.0 * reference / float(spec.e_max) - 1.0,
        -1.0 + 1e-12,
        1.0 - 1e-12,
    )
    latent_source = np.arctanh(normalized) / float(spec.sigma_g)

    realization = demo_data.get("_grf_realization")
    if realization is None:
        source_coordinates = np.asarray(source_mesh.coord, dtype=float)
        covariance = _rbf_matrix(source_coordinates, source_coordinates, spec.ell)
        covariance.flat[:: covariance.shape[0] + 1] += float(spec.jitter)
        latent_vector = latent_source.ravel(order="C")
        cholesky = np.linalg.cholesky(covariance)
        random_vector = np.linalg.solve(cholesky, latent_vector)
        rbf_weights = np.linalg.solve(covariance, latent_vector)
        realization = {
            "source_coordinates": source_coordinates,
            "latent_source": latent_source,
            "random_vector": random_vector,
            "rbf_weights": rbf_weights,
        }
        demo_data["_grf_realization"] = realization

    # Preserve the imported MATLAB field exactly at its source resolution.
    if (
        mesh.nods_x == source_mesh.nods_x
        and mesh.nods_y == source_mesh.nods_y
        and np.allclose(mesh.coord, source_mesh.coord)
    ):
        return reference.copy(), MaterialInfo(nu=nu, dis_type="grf")

    target_coordinates = np.asarray(mesh.coord, dtype=float)
    latent_target = np.empty(target_coordinates.shape[0], dtype=float)
    source_coordinates = realization["source_coordinates"]
    rbf_weights = realization["rbf_weights"]
    chunk_size = 2048
    for start in range(0, target_coordinates.shape[0], chunk_size):
        stop = min(start + chunk_size, target_coordinates.shape[0])
        kernel = _rbf_matrix(
            target_coordinates[start:stop],
            source_coordinates,
            spec.ell,
        )
        latent_target[start:stop] = kernel @ rbf_weights
    latent_target = latent_target.reshape(mesh.nods_y, mesh.nods_x, order="C")
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
