"""Run the original parameter-driven forward mesh convergence study."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grid_study.forward_convergence import run_forward_grid_study
import config as cfg


# This path uses EX and EY from config.py for BIL/EXP and the GRF settings
# from the same forward configuration.
DATASETS = ("bil", "exp", "grf")
NODES = (4, 10, 20, 30, 40, 60, 80, 100, 200)
OUTPUT_DIR = REPO_ROOT / "results" / "grid_study" / "forward_original"

run_forward_grid_study(
    forward_config=cfg.get_forward_config(),
    datasets=DATASETS,
    nodes_list=NODES,
    output_dir=OUTPUT_DIR,
    demo_root=None,
)
