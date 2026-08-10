"""Persistence helpers shared by grid-study workflows."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np

from fgm_asm import MeshInfo


def write_csv_rows(rows: list[dict], path: Path, columns: list[str] | None = None) -> Path:
    """Write a list of scalar result rows to CSV."""
    if not rows:
        raise ValueError("Cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if columns is None else columns
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in columns} for row in rows)
    return path


def save_forward_case(case: dict, path: Path) -> None:
    """Save the numerical part of one forward grid-study case."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {
                "dataset": case["dataset"],
                "nodes": case["nodes"],
                "elements": case["elements"],
                "E_true": case["E_field"],
                "U_clean": case["U"],
                "bc_info": case["bc_info"],
                "config": case["config"],
                "forward_elapsed_time_seconds": case["elapsed_time_seconds"],
                "forward_peak_python_memory_mb": case["peak_python_memory_mb"],
            },
            handle,
        )


def load_forward_case(path: Path) -> dict:
    """Load a forward case and rebuild geometry-only mesh data."""
    with path.open("rb") as handle:
        saved = pickle.load(handle)
    config = saved["config"]
    nodes = int(saved["nodes"])
    mesh = MeshInfo(config["geo_l"], config["geo_h"], nodes - 1, nodes - 1)
    return {
        "dataset": saved["dataset"],
        "nodes": nodes,
        "elements": int(saved["elements"]),
        "mesh": mesh,
        "bc_info": saved["bc_info"],
        "E_field": np.asarray(saved["E_true"], dtype=float),
        "U": np.asarray(saved["U_clean"], dtype=float),
        "config": config,
        "elapsed_time_seconds": float(saved["forward_elapsed_time_seconds"]),
        "peak_python_memory_mb": float(saved["forward_peak_python_memory_mb"]),
    }


__all__ = ["load_forward_case", "save_forward_case", "write_csv_rows"]
