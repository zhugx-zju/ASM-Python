# ASM grid study

This study separates forward mesh convergence from inverse-result
post-processing.

## Forward problem

The forward workflow may be run in batch because no regularization parameter is
selected. It evaluates the same analytical BIL/EXP field on 10 x 10, 20 x 20,
30 x 30, 40 x 40, 60 x 60, 80 x 80, and 100 x 100 nodal meshes, solves the
forward FEM problem, and compares the coarser displacement fields with the
100 x 100 result after interpolation.

Run from `ASM-Python`:

```powershell
python forward_grid_convergence.py
```

Outputs are written below:

```text
results/grid_study/forward/
|-- bil/nodes_10/forward_result.pkl
|-- bil/nodes_20/forward_result.pkl
|-- bil/nodes_30/forward_result.pkl
|-- bil/nodes_40/forward_result.pkl
|-- bil/nodes_60/forward_result.pkl
|-- bil/nodes_80/forward_result.pkl
|-- bil/nodes_100/forward_result.pkl
|-- exp/...
|-- forward_grid_metrics.csv
|-- forward_edge_profiles.csv
|-- forward_edge_bil_ux.(png|pdf)
|-- forward_edge_bil_uy.(png|pdf)
|-- forward_edge_exp_ux.(png|pdf)
|-- forward_edge_exp_uy.(png|pdf)
`-- forward_grid_convergence.(png|pdf)
```

Each edge-profile figure contains one modulus distribution and one displacement
component, with all mesh densities overlaid on the common physical boundary
`x=L`. Each mesh is interpolated to the same 181-point physical `y` coordinate,
so the curves can be compared directly rather than by unrelated node indices.

## Inverse problem

Inverse cases must be completed independently because each grid may require its
own L-curve/gamma selection. The postprocessor never runs ASM and never
changes gamma. After the result directories are available, copy
`inverse_cases.example.json`, replace the paths, and call:

```python
from grid_study.inverse_postprocess import postprocess

postprocess("path/to/inverse_cases.json")
```

The postprocessor reads `inverse_results.pkl` and the saved configuration
snapshot from each explicit result directory. It computes the global relative
L1 error against that grid's true field and the relative difference to the
interpolated reference-grid reconstruction. It also reports the selected gamma,
iterations, elapsed time, and peak Python allocation measured by the single-case
inverse scripts.
