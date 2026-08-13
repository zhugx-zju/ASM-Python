"""CSV persistence shared by forward grid-study workflows."""

from __future__ import annotations

import csv
from pathlib import Path


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


__all__ = ["write_csv_rows"]
