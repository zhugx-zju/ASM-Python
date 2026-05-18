"""
Shared script-level helpers for loading problem data and resolving result paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .results_io import find_forward_data_path, load_forward_data


def load_latest_forward_problem(search_root: Path | str = ".") -> tuple[Path, dict[str, Any]]:
    """Load the latest available forward-problem dataset."""
    forward_data_path, _ = find_forward_data_path(search_root)
    forward_data = load_forward_data(forward_data_path)
    return forward_data_path, forward_data


def resolve_results_dir(forward_data_path: Path | str, forward_data: dict[str, Any]) -> Path:
    """Resolve the base output directory associated with a forward dataset."""
    forward_data_path = Path(forward_data_path)
    folder_name = forward_data.get("folder_name")
    if folder_name:
        return Path(folder_name)
    return forward_data_path.parent
