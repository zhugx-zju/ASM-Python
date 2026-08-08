"""Run the forward-problem mesh convergence study."""

from grid_study.forward_convergence import (
    DEFAULT_DATASETS,
    DEFAULT_NODES,
    run_forward_grid_study,
)
from script import REPO_ROOT


run_forward_grid_study(
    datasets=DEFAULT_DATASETS,
    nodes_list=DEFAULT_NODES,
    output_dir=REPO_ROOT / "results" / "grid_study" / "forward",
)
