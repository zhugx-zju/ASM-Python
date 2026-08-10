"""Run the repository-local forward-problem mesh convergence study."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grid_study.forward_convergence import (
    DEFAULT_DATASETS,
    DEFAULT_NODES,
    run_forward_grid_study,
)
from grid_study.demo_data import DEFAULT_DEMO_ROOT
import config as cfg


DATASETS = DEFAULT_DATASETS
NODES = DEFAULT_NODES
OUTPUT_DIR = REPO_ROOT / "results" / "grid_study" / "forward_demo"

run_forward_grid_study(
    forward_config=cfg.get_forward_config(),
    datasets=DATASETS,
    nodes_list=NODES,
    output_dir=OUTPUT_DIR,
    demo_root=DEFAULT_DEMO_ROOT,
)
