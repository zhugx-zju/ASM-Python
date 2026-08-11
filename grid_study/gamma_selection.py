"""Batch gamma selection via L-curve for grid convergence studies.

This module provides tools to select regularization parameters for each noise
level on a reference mesh before running multi-mesh convergence tests.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from fgm_asm import find_optimal_gamma_lcurve
from fgm_asm.config_types import ForwardConfig, LCurveConfig
from fgm_asm.utils import add_noise_to_displacement
from grid_study.case_runner import run_forward_case
from grid_study.demo_data import DEFAULT_DEMO_ROOT, load_demo_dataset


DEFAULT_NOISE_PERCENTAGE = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
NOISE_SEED = 42


def select_gamma_for_dataset(
    dataset: str,
    reference_nodes: int,
    forward_config: ForwardConfig,
    lcurve_config: LCurveConfig,
    noise_percentage: tuple[float, ...] = DEFAULT_NOISE_PERCENTAGE,
    output_dir: Path | str = "results/grid_study/gamma_selection",
    demo_root: Path | str | None = DEFAULT_DEMO_ROOT,
) -> dict:
    """Select gamma for each noise level using L-curve analysis on a reference mesh.

    Args:
        dataset: Dataset type (bil/exp/grf)
        reference_nodes: Nodes per direction for the reference mesh
        forward_config: Forward problem configuration
        lcurve_config: L-curve analysis configuration
        noise_percentage: Tuple of noise levels in percentage
        output_dir: Directory to save gamma selection results
        demo_root: Root directory for demo data, None for analytical fields

    Returns:
        Dictionary containing gamma values and metadata for each noise level
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized = str(dataset).strip().lower()
    if normalized not in ("bil", "exp", "grf"):
        raise ValueError(f"Dataset must be bil/exp/grf, got {dataset!r}")

    print(f"\n{'='*70}")
    print(f"Gamma Selection for {normalized.upper()} at {reference_nodes}x{reference_nodes} mesh")
    print(f"{'='*70}\n")

    # Run forward problem once on the reference mesh
    print(f"Running forward problem on {reference_nodes}x{reference_nodes} mesh...")
    demo_data = load_demo_dataset(normalized, demo_root) if demo_root is not None else None

    case = run_forward_case(
        normalized,
        reference_nodes,
        forward_config=forward_config,
        demo_data=demo_data,
    )

    # Assemble mass matrix once
    case["mesh"].assemble_mass_matrix()

    gamma_results = {}

    # Run L-curve for each noise level
    for noise in noise_percentage:
        print(f"\n{'-'*70}")
        print(f"Processing noise level: {noise}%")
        print(f"{'-'*70}")

        noise_level = float(noise) / 100.0
        U_measured = add_noise_to_displacement(case["U"], noise_level, seed=NOISE_SEED)

        tracemalloc.start()
        start_time = time.time()
        try:
            gamma_optimal, lcurve_results = find_optimal_gamma_lcurve(
                mesh_info=case["mesh"],
                bc_info=case["bc_info"],
                U_measured=U_measured,
                config=forward_config,
                gamma_min=lcurve_config.gamma_min,
                gamma_max=lcurve_config.gamma_max,
                n_gamma=lcurve_config.n_gamma,
                E_min=lcurve_config.E_min,
                E_max=lcurve_config.E_max,
                max_iter=lcurve_config.max_iter,
                ftol=lcurve_config.ftol,
                gtol=lcurve_config.gtol,
            )
        finally:
            _, peak_memory_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        elapsed_time = time.time() - start_time

        optimal_idx = lcurve_results["optimal_idx"]
        scan_result = lcurve_results["all_results"][optimal_idx]

        gamma_results[float(noise)] = {
            "gamma": float(gamma_optimal),
            "optimal_idx": int(optimal_idx),
            "converged": bool(scan_result["converged"]),
            "iterations": int(scan_result["n_iterations"]),
            "elapsed_time_seconds": float(elapsed_time),
            "peak_memory_mb": float(peak_memory_bytes / 1024**2),
        }

        print(f"  Selected gamma: {gamma_optimal:.6e}")
        print(f"  Converged: {scan_result['converged']}")
        print(f"  Iterations: {scan_result['n_iterations']}")
        print(f"  Time: {elapsed_time:.2f} seconds")

    # Save results to JSON
    result_data = {
        "dataset": normalized,
        "reference_nodes": int(reference_nodes),
        "noise_percentage": list(noise_percentage),
        "noise_seed": NOISE_SEED,
        "gamma_selection": gamma_results,
        "lcurve_config": lcurve_config.to_dict(),
        "forward_config": case["config"],
    }

    output_path = output_dir / f"gamma_selection_{normalized}_nodes{reference_nodes}.json"
    output_path.write_text(json.dumps(result_data, indent=2), encoding="utf-8")

    print(f"\n{'='*70}")
    print(f"Gamma selection complete for {normalized.upper()}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}\n")

    return result_data


def load_gamma_by_noise(gamma_file: Path | str) -> dict[float, float]:
    """Load a ``{noise_percentage: gamma}`` mapping from a saved selection file."""
    gamma_file = Path(gamma_file)
    data = json.loads(gamma_file.read_text(encoding="utf-8"))
    return {
        float(noise): float(entry["gamma"])
        for noise, entry in data["gamma_selection"].items()
    }


__all__ = [
    "DEFAULT_NOISE_PERCENTAGE",
    "load_gamma_by_noise",
    "select_gamma_for_dataset",
]
