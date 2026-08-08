"""Run the forward-problem mesh convergence study."""

from grid_study.forward_convergence import (
    DEFAULT_DATASETS,
    DEFAULT_NODES,
    run_forward_grid_study,
)


run_forward_grid_study(
    datasets=DEFAULT_DATASETS,
    nodes_list=DEFAULT_NODES,
    output_dir="results/grid_study/forward",
)
