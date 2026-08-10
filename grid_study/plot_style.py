"""Shared plotting styles for grid-study figures."""

from __future__ import annotations

import numpy as np
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter


DATASET_STYLES = {
    "bil": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "exp": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    "grf": {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
}

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "Times New Roman"
rcParams["mathtext.it"] = "Times New Roman:italic"
rcParams["axes.linewidth"] = 1.0


def compact_tick(value: float, _position: int) -> str:
    """Format ticks without trailing zeros or unnecessary decimal places."""
    if abs(value) < 1e-12:
        return "0"
    return f"{value:g}"


def style_axes(ax) -> None:
    """Apply the common grid-study axis style."""
    ax.grid(False)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=False,
        right=False,
        width=0.9,
        length=4.5,
    )
    ax.xaxis.set_major_formatter(FuncFormatter(compact_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(compact_tick))


__all__ = ["DATASET_STYLES", "compact_tick", "style_axes"]
