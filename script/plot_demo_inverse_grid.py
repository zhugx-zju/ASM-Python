"""Generate inverse mesh-study figures from completed BIL metrics."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grid_study.inverse_grid_plots import plot_inverse_grid_study


METRICS_PATH = (
    REPO_ROOT
    / "results"
    / "grid_study"
    / "inverse_demo"
    / "bil_adaptive_gamma_metrics.csv"
)
OUTPUT_DIR = METRICS_PATH.parent / "summary"


plot_inverse_grid_study(
    metrics_path=METRICS_PATH,
    output_dir=OUTPUT_DIR,
    dataset="bil",
    nodes=(10, 20, 40, 80, 100, 200),
)
