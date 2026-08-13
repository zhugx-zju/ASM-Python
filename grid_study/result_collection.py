"""Collect completed independent demo inverse cases without rerunning solvers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_INVERSE_DEMO_ROOT = Path("results/grid_study/inverse_demo")


def _nodes_from_directory(directory: Path) -> tuple[int, int, bool]:
    """Parse current ``nodes_NxM`` or legacy square ``nodes_N`` names."""
    if not directory.name.startswith("nodes_"):
        raise ValueError(f"Invalid inverse-case mesh directory: {directory}")
    token = directory.name.removeprefix("nodes_")
    if "x" in token:
        parts = token.split("x")
        if len(parts) != 2:
            raise ValueError(f"Invalid inverse-case mesh directory: {directory}")
        nodesx, nodesy = (int(part) for part in parts)
        return nodesx, nodesy, True
    nodes = int(token)
    return nodes, nodes, False


def collect_demo_inverse_results(
    project_root: Path | str,
    results_root: Path | str = DEFAULT_INVERSE_DEMO_ROOT,
) -> list[dict[str, Any]]:
    """Read every completed case's ``metrics.json`` in stable path order."""
    root = Path(results_root)
    if not root.is_absolute():
        root = Path(project_root) / root
    selected: dict[tuple[str, int, int, float], tuple[bool, dict[str, Any]]] = {}
    for metrics_path in sorted(root.glob("*/nodes_*/noise_*/metrics.json")):
        nodesx, nodesy, uses_current_layout = _nodes_from_directory(
            metrics_path.parent.parent
        )
        row = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
        dataset = str(row.get("dataset", metrics_path.parents[2].name)).lower()
        noise = float(row.get(
            "noise_percentage",
            metrics_path.parent.name.removeprefix("noise_"),
        ))
        row["dataset"] = dataset
        row["nodesx"] = nodesx
        row["nodesy"] = nodesy
        if nodesx == nodesy:
            row["nodes"] = nodesx
        row["noise_percentage"] = noise
        row["case_dir"] = str(metrics_path.parent)
        key = (dataset, nodesx, nodesy, noise)
        previous = selected.get(key)
        if previous is None or (uses_current_layout and not previous[0]):
            selected[key] = (uses_current_layout, row)
    return [
        selected[key][1]
        for key in sorted(selected, key=lambda value: (value[0], value[1], value[2], value[3]))
    ]


def write_demo_inverse_summary(
    project_root: Path | str,
    results_root: Path | str = DEFAULT_INVERSE_DEMO_ROOT,
    filename: str = "inverse_demo_metrics.csv",
) -> Path:
    """Write one CSV row per independently completed demo inverse case."""
    root = Path(results_root)
    if not root.is_absolute():
        root = Path(project_root) / root
    rows = collect_demo_inverse_results(project_root, root)
    if not rows:
        raise FileNotFoundError(f"No completed demo inverse cases were found below {root}")
    preferred = [
        "dataset",
        "nodesx",
        "nodesy",
        "noise_percentage",
        "gamma",
        "relative_l1_E",
        "relative_l2_E",
        "relative_linf_E",
        "mae_E_percent",
        "rmse_E",
        "converged",
        "iterations",
        "inverse_elapsed_time_seconds",
        "lcurve_elapsed_time_seconds",
        "peak_python_memory_mb",
        "case_dir",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    columns = [key for key in preferred if any(key in row for row in rows)] + remaining
    output_path = root / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


__all__ = ["collect_demo_inverse_results", "write_demo_inverse_summary"]
