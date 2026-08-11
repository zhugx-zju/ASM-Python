"""Run batch gamma selection for BIL demo dataset at 40x40 reference mesh."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grid_study.gamma_selection import (
    DEFAULT_NOISE_PERCENTAGE,
    select_gamma_for_dataset,
)
from grid_study.demo_data import DEFAULT_DEMO_ROOT
import config as cfg


DATASET = "bil"
REFERENCE_NODES = 40
NOISE_PERCENTAGE = DEFAULT_NOISE_PERCENTAGE
OUTPUT_DIR = REPO_ROOT / "results" / "grid_study" / "gamma_selection"

select_gamma_for_dataset(
    dataset=DATASET,
    reference_nodes=REFERENCE_NODES,
    forward_config=cfg.get_forward_config(),
    lcurve_config=cfg.get_lcurve_config(),
    noise_percentage=NOISE_PERCENTAGE,
    output_dir=OUTPUT_DIR,
    demo_root=DEFAULT_DEMO_ROOT,
)
