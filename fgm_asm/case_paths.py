"""Stable output paths for independently runnable demo inverse cases."""

from __future__ import annotations

from pathlib import Path


def format_number_token(value: float) -> str:
    """Format an integer-like number without a decimal point."""
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def resolve_demo_inverse_case_dir(
    project_root: Path | str,
    dataset: str,
    nodesx: int,
    nodesy: int,
    noise_level: float,
    output_root: Path | str = "results/grid_study/inverse_demo",
) -> Path:
    """Return ``dataset/nodes_NxM/noise_P`` for one demo inverse case."""
    root = Path(output_root)
    if not root.is_absolute():
        root = Path(project_root) / root
    return (
        root
        / str(dataset).strip().lower()
        / f"nodes_{int(nodesx)}x{int(nodesy)}"
        / f"noise_{format_number_token(noise_level)}"
    )


__all__ = ["format_number_token", "resolve_demo_inverse_case_dir"]
