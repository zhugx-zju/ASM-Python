"""Post-process independently completed ASM inverse grid cases.

The script never runs an inverse solve and never selects gamma.  It reads an
explicit manifest of result directories, computes common metrics, interpolates
the finest-grid reconstruction to coarser grids, and creates convergence plots.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _field(values, nodes: int) -> np.ndarray:
    array = np.asarray(values, dtype=float).squeeze()
    if array.shape == (nodes, nodes):
        return array
    if array.size == nodes * nodes:
        return array.reshape(nodes, nodes, order="C")
    raise ValueError(f"Cannot reshape field of shape {array.shape} for {nodes}x{nodes} nodes")


def _relative_l1(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sum(np.abs(pred - target)) / (np.sum(np.abs(target)) + 1e-15))


def _interpolate(field: np.ndarray, source_nodes: int, target_nodes: int) -> np.ndarray:
    source_axis = np.linspace(0.0, 1.0, source_nodes)
    target_axis = np.linspace(0.0, 1.0, target_nodes)
    interpolator = RegularGridInterpolator((source_axis, source_axis), field)
    yy, xx = np.meshgrid(target_axis, target_axis, indexing="ij")
    return interpolator(np.column_stack((yy.ravel(), xx.ravel()))).reshape(target_nodes, target_nodes)


def _read_config_snapshot(result_dir: Path) -> dict:
    snapshot = result_dir / "config.py"
    if not snapshot.exists():
        return {}
    values = {}
    for line in snapshot.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, raw_value = line.split("=", 1)
        try:
            values[key.strip()] = ast.literal_eval(raw_value.strip())
        except (SyntaxError, ValueError):
            continue
    return values


def _resolve_result_dir(manifest_dir: Path, item: dict) -> Path:
    path = Path(item["result_dir"])
    if not path.is_absolute():
        path = (manifest_dir / path).resolve()
    if (path / "inverse_results.pkl").exists():
        return path
    noise_percent = item.get("noise_percent")
    if noise_percent is not None:
        tag = f"noise_{float(noise_percent):.2f}pct"
        candidate = path / tag
        if (candidate / "inverse_results.pkl").exists():
            return candidate
    raise FileNotFoundError(f"inverse_results.pkl not found under {path}")


def _load_case(manifest_dir: Path, item: dict) -> dict:
    result_dir = _resolve_result_dir(manifest_dir, item)
    with (result_dir / "inverse_results.pkl").open("rb") as handle:
        data = pickle.load(handle)
    nodes = int(item["nodes"])
    true_field = _field(data["E_true"], nodes)
    pred_field = _field(data["E_reconstructed"], nodes)
    results = data.get("results", {})
    snapshot = _read_config_snapshot(result_dir)
    noise_percent = item.get("noise_percent")
    if noise_percent is None and data.get("noise_level") is not None:
        noise_percent = 100.0 * float(data["noise_level"])
    return {
        "dataset": str(item["dataset"]).lower(),
        "nodes": nodes,
        "noise_percent": float(noise_percent or 0.0),
        "result_dir": str(result_dir),
        "true": true_field,
        "pred": pred_field,
        "relative_l1_true": _relative_l1(pred_field, true_field),
        "mae_percent": float(np.mean(np.abs(pred_field - true_field) / (np.abs(true_field) + 1e-15)) * 100.0),
        "rmse": float(np.sqrt(np.mean((pred_field - true_field) ** 2))),
        "gamma": data.get("gamma_used", snapshot.get("GAMMA_USED", "")),
        "iterations": data.get("n_iterations", results.get("n_iterations", "")),
        "converged": results.get("converged", ""),
        "elapsed_time_seconds": data.get("elapsed_time_total_seconds", snapshot.get("ELAPSED_TIME_TOTAL_SECONDS", "")),
        "lcurve_time_seconds": snapshot.get("LCURVE_SCAN_TIME_SECONDS", ""),
        "peak_python_memory_mb": data.get("peak_python_memory_mb", snapshot.get("PEAK_PYTHON_MEMORY_MB", "")),
        "peak_lcurve_python_memory_mb": data.get(
            "peak_lcurve_python_memory_mb",
            snapshot.get("PEAK_LCURVE_PYTHON_MEMORY_MB", ""),
        ),
    }


def _write_csv(rows: list[dict], output_dir: Path) -> Path:
    path = output_dir / "inverse_grid_metrics.csv"
    columns = [
        "dataset", "nodes", "noise_percent", "relative_l1_true",
        "relative_l1_to_reference", "mae_percent", "rmse", "gamma",
        "iterations", "converged", "elapsed_time_seconds", "lcurve_time_seconds",
        "peak_python_memory_mb", "peak_lcurve_python_memory_mb",
        "result_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in columns} for row in rows)
    return path


def _plot(rows: list[dict], output_dir: Path) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    groups = sorted({(row["dataset"], row["noise_percent"]) for row in rows})
    for dataset, noise in groups:
        subset = sorted(
            (row for row in rows if row["dataset"] == dataset and row["noise_percent"] == noise),
            key=lambda row: row["nodes"],
        )
        label = f"{dataset.upper()}, {noise:g}%"
        nodes = [row["nodes"] for row in subset]
        axes[0].plot(nodes, [row["relative_l1_true"] for row in subset], "o-", label=label)
        axes[0].plot(nodes, [row["relative_l1_to_reference"] for row in subset], "s--", alpha=0.8)
        axes[1].plot(nodes, [float(row["elapsed_time_seconds"]) if row["elapsed_time_seconds"] != "" else np.nan for row in subset], "o-", label=label)
        axes[2].plot(
            nodes,
            [float(row["peak_python_memory_mb"]) if row["peak_python_memory_mb"] != "" else np.nan for row in subset],
            "o-",
            label=label,
        )

    axes[0].set_xlabel("Nodes per direction")
    axes[0].set_ylabel("Relative L1 error / difference")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].set_xlabel("Nodes per direction")
    axes[1].set_ylabel("Inverse solve time (s)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8, frameon=False)

    axes[2].set_xlabel("Nodes per direction")
    axes[2].set_ylabel("Peak Python memory (MB)")
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend(fontsize=8, frameon=False)

    fig.suptitle("ASM Inverse Mesh Influence")
    fig.tight_layout()
    png_path = output_dir / "inverse_grid_convergence.png"
    pdf_path = output_dir / "inverse_grid_convergence.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def postprocess(manifest_path: Path | str, output_dir: Path | str | None = None) -> Path:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_dir = manifest_path.parent
    output_dir = Path(output_dir) if output_dir is not None else manifest_dir / "postprocessed"
    if not output_dir.is_absolute():
        output_dir = (manifest_dir / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [_load_case(manifest_dir, item) for item in manifest["cases"]]
    reference_nodes = int(manifest.get("reference_nodes", max(case["nodes"] for case in cases)))
    references = {
        (case["dataset"], case["noise_percent"]): case
        for case in cases
        if case["nodes"] == reference_nodes
    }
    if not references:
        raise ValueError(f"No reference cases found at nodes={reference_nodes}")

    for case in cases:
        reference = references.get((case["dataset"], case["noise_percent"]))
        if reference is None:
            raise ValueError(
                f"Missing reference case for dataset={case['dataset']}, "
                f"noise={case['noise_percent']:g}% at nodes={reference_nodes}"
            )
        reference_on_grid = _interpolate(reference["pred"], reference_nodes, case["nodes"])
        case["relative_l1_to_reference"] = _relative_l1(reference_on_grid, case["pred"])

    metrics_path = _write_csv(cases, output_dir)
    figure_paths = _plot(cases, output_dir)
    summary = {
        "reference_nodes": reference_nodes,
        "cases": [
            {
                key: value
                for key, value in case.items()
                if key not in {"true", "pred"}
            }
            for case in cases
        ],
        "metrics_file": str(metrics_path),
        "figures": [str(path) for path in figure_paths],
    }
    (output_dir / "postprocess_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Saved inverse grid metrics to {metrics_path}")
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Explicit manifest of independently completed inverse cases")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    postprocess(args.manifest, args.output)


if __name__ == "__main__":
    main()
