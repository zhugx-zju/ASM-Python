"""Adaptive gamma reuse-with-fallback strategy for mesh x noise inverse studies.

For each mesh, one gamma is selected via L-curve on a representative ("seed")
noise level, then reused across every other noise level on that same mesh.
Any noise level whose reconstruction quality falls below a threshold when
reusing the seed gamma is individually re-solved with its own L-curve search.
This keeps the total L-curve search count far below a full mesh x noise
sweep while still catching cases where the reused gamma is a poor fit.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from fgm_asm import find_optimal_gamma_lcurve
from fgm_asm.config_types import ForwardConfig, InverseConfig, LCurveConfig
from fgm_asm.results_io import save_inverse_results
from fgm_asm.utils import add_noise_to_displacement
from grid_study.case_runner import run_forward_case, run_inverse_case
from grid_study.demo_data import DEFAULT_DEMO_ROOT, load_demo_dataset
from grid_study.study_io import write_csv_rows
from grid_study.study_metrics import inverse_metrics


DEFAULT_DATASETS = ("bil", "exp", "grf")
DEFAULT_NODES = (4, 10, 20, 40, 80, 100, 200)
DEFAULT_NOISE_PERCENTAGE = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
DEFAULT_SEED_NOISE = 4.0
DEFAULT_QUALITY_THRESHOLD = 0.4
GRF_REFERENCE_NODES = 200
NOISE_SEED = 42


def _run_lcurve(mesh, bc_info, U_measured, forward_config, lcurve_config):
    """Run one L-curve scan and return (gamma, scan_result, elapsed, peak_mb)."""
    tracemalloc.start()
    start = time.time()
    try:
        gamma_optimal, lcurve_results = find_optimal_gamma_lcurve(
            mesh_info=mesh,
            bc_info=bc_info,
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
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    elapsed = time.time() - start
    optimal_idx = lcurve_results["optimal_idx"]
    scan_result = lcurve_results["all_results"][optimal_idx]
    return float(gamma_optimal), scan_result, elapsed, float(peak_bytes / 1024**2)


def run_case_with_reuse_fallback(
    case: dict,
    noise_percentage: tuple[float, ...],
    seed_noise: float,
    forward_config: ForwardConfig,
    inverse_config: InverseConfig,
    lcurve_config: LCurveConfig,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    max_iter: int | None = None,
) -> dict:
    """Run one mesh's noise sweep using seed-gamma reuse with fallback re-search.

    Returns a dict with per-noise metrics rows and a summary of which noise
    levels reused the seed gamma versus triggered an individual re-search.
    """
    if seed_noise not in noise_percentage:
        raise ValueError(f"seed_noise {seed_noise} must be included in noise_percentage")

    mesh = case["mesh"]
    if mesh.M is None:
        mesh.assemble_mass_matrix()
    max_iter = int(inverse_config.max_iter if max_iter is None else max_iter)

    seed_noise_level = float(seed_noise) / 100.0
    U_seed = add_noise_to_displacement(case["U"], seed_noise_level, seed=NOISE_SEED)
    seed_gamma, seed_scan, seed_lcurve_time, seed_lcurve_mb = _run_lcurve(
        mesh, case["bc_info"], U_seed, forward_config, lcurve_config
    )

    rows = []
    reuse_log = []

    for noise in noise_percentage:
        noise_level = float(noise) / 100.0
        if noise == seed_noise:
            U_measured = U_seed
        else:
            U_measured = add_noise_to_displacement(case["U"], noise_level, seed=NOISE_SEED)

        result, U_measured, inverse_elapsed, peak_memory_mb = run_inverse_case(
            case, noise, seed_gamma, inverse_config, max_iter=max_iter, U_measured=U_measured
        )
        row = inverse_metrics(
            case, result, U_measured, noise, seed_gamma, inverse_elapsed, peak_memory_mb
        )

        reused = True
        resolved_gamma = seed_gamma
        if row["relative_l2_E"] > quality_threshold or not row["converged"]:
            reused = False
            resolved_gamma, rescan_result, rescan_time, rescan_mb = _run_lcurve(
                mesh, case["bc_info"], U_measured, forward_config, lcurve_config
            )
            result, U_measured, inverse_elapsed, peak_memory_mb = run_inverse_case(
                case, noise, resolved_gamma, inverse_config, max_iter=max_iter, U_measured=U_measured
            )
            row = inverse_metrics(
                case, result, U_measured, noise, resolved_gamma, inverse_elapsed, peak_memory_mb
            )
            row["lcurve_rescan_time_seconds"] = rescan_time
            row["lcurve_rescan_peak_memory_mb"] = rescan_mb

        row["gamma_source"] = "seed_reuse" if reused else "individual_rescan"
        row["seed_gamma"] = seed_gamma
        row["seed_noise_percentage"] = float(seed_noise)
        rows.append(row)
        reuse_log.append({
            "noise_percentage": float(noise),
            "gamma_source": row["gamma_source"],
            "gamma_used": resolved_gamma,
            "relative_l2_E": row["relative_l2_E"],
            "converged": row["converged"],
        })

    return {
        "nodes": int(case["nodes"]),
        "seed_noise": float(seed_noise),
        "seed_gamma": seed_gamma,
        "seed_lcurve_time_seconds": seed_lcurve_time,
        "seed_lcurve_peak_memory_mb": seed_lcurve_mb,
        "seed_scan_converged": bool(seed_scan["converged"]),
        "quality_threshold": float(quality_threshold),
        "rows": rows,
        "reuse_log": reuse_log,
    }


def run_adaptive_grid_study(
    dataset: str,
    forward_config: ForwardConfig,
    inverse_config: InverseConfig,
    lcurve_config: LCurveConfig,
    nodes_list: tuple[int, ...] = DEFAULT_NODES,
    noise_percentage: tuple[float, ...] = DEFAULT_NOISE_PERCENTAGE,
    seed_noise: float = DEFAULT_SEED_NOISE,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    output_dir: Path | str = "results/grid_study/inverse_adaptive",
    max_iter: int | None = None,
    demo_root: Path | str | None = DEFAULT_DEMO_ROOT,
) -> Path:
    """Run the seed-gamma-reuse-with-fallback study across every mesh for one dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = str(dataset).strip().lower()
    if normalized not in DEFAULT_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset!r}")

    demo_data = load_demo_dataset(normalized, demo_root) if demo_root is not None else None

    all_rows = []
    mesh_summaries = []

    for nodes in nodes_list:
        print(f"\n{'='*70}")
        print(f"Mesh {nodes}x{nodes}: seed noise {seed_noise}% L-curve, then reuse+fallback")
        print(f"{'='*70}")

        case = run_forward_case(
            normalized,
            nodes,
            forward_config=forward_config,
            demo_data=demo_data,
        )

        mesh_result = run_case_with_reuse_fallback(
            case,
            noise_percentage,
            seed_noise,
            forward_config,
            inverse_config,
            lcurve_config,
            quality_threshold=quality_threshold,
            max_iter=max_iter,
        )

        mesh_dir = output_dir / normalized / f"nodes_{nodes}"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        for row in mesh_result["rows"]:
            noise = row["noise_percentage"]
            noise_dir = mesh_dir / f"noise_{noise:g}"
            noise_dir.mkdir(parents=True, exist_ok=True)
            (noise_dir / "metrics.json").write_text(
                json.dumps(row, indent=2), encoding="utf-8"
            )
        (mesh_dir / "reuse_summary.json").write_text(
            json.dumps(
                {
                    "nodes": mesh_result["nodes"],
                    "seed_noise": mesh_result["seed_noise"],
                    "seed_gamma": mesh_result["seed_gamma"],
                    "seed_lcurve_time_seconds": mesh_result["seed_lcurve_time_seconds"],
                    "seed_lcurve_peak_memory_mb": mesh_result["seed_lcurve_peak_memory_mb"],
                    "seed_scan_converged": mesh_result["seed_scan_converged"],
                    "quality_threshold": mesh_result["quality_threshold"],
                    "reuse_log": mesh_result["reuse_log"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        all_rows.extend(mesh_result["rows"])
        mesh_summaries.append({
            "nodes": nodes,
            "seed_gamma": mesh_result["seed_gamma"],
            "reuse_log": mesh_result["reuse_log"],
        })

        n_rescan = sum(1 for entry in mesh_result["reuse_log"] if entry["gamma_source"] == "individual_rescan")
        print(f"Mesh {nodes}x{nodes} complete: seed gamma={mesh_result['seed_gamma']:.4e}, "
              f"{n_rescan}/{len(noise_percentage)} noise levels needed individual re-search")

    metrics_path = write_csv_rows(all_rows, output_dir / f"{normalized}_adaptive_gamma_metrics.csv")
    manifest = {
        "study": "asm_inverse_grid_adaptive_gamma_reuse",
        "dataset": normalized,
        "nodes": list(nodes_list),
        "noise_percentage": list(noise_percentage),
        "seed_noise": float(seed_noise),
        "quality_threshold": float(quality_threshold),
        "noise_seed": NOISE_SEED,
        "forward_config": forward_config.to_dict(),
        "inverse_config": inverse_config.to_dict(),
        "lcurve_config": lcurve_config.to_dict(),
        "mesh_summaries": mesh_summaries,
        "metrics_file": metrics_path.name,
    }
    (output_dir / f"{normalized}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nSaved adaptive-gamma metrics to {metrics_path}")
    return metrics_path


__all__ = [
    "DEFAULT_DATASETS",
    "DEFAULT_NODES",
    "DEFAULT_NOISE_PERCENTAGE",
    "DEFAULT_QUALITY_THRESHOLD",
    "DEFAULT_SEED_NOISE",
    "run_adaptive_grid_study",
    "run_case_with_reuse_fallback",
]
