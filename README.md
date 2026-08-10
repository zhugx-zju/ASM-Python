# FGM ASM Solver

This repository contains a small finite element codebase for forward and inverse analysis of 2D functionally graded materials (FGMs) under plane-stress assumptions.

The project covers:

- structured quadrilateral mesh generation
- forward displacement analysis
- modulus-field reconstruction from displacement data
- Tikhonov regularization and gradient checking
- L-curve based regularization-parameter selection
- plotting and export utilities for saved results

## Branch Strategy

This repository is managed as a research codebase with one stable baseline branch and independent exploratory branches.

- `main`: baseline branch for the current small-strain, linear-elastic, force-controlled workflow
- `disp_linear`: displacement-controlled branch under small-strain linear elasticity
- `disp_nonlinear`: displacement-controlled branch for finite-strain and nonlinear constitutive development

The intended workflow is:

- keep `main` stable and easy to rerun
- let each research branch evolve independently
- only bring clearly reusable fixes or utilities back to `main`

See [docs/branch_scope.md](docs/branch_scope.md) for the lightweight branch rules and [docs/validation.md](docs/validation.md) for the validation-record template.

## Model Summary

- Geometry: rectangular 2D domain
- Elements: 4-node bilinear quadrilateral elements
- Integration: 2x2 Gauss quadrature
- Material field: nodal Young's modulus with bilinear, exponential, or single-case GRF variation
- Boundary conditions: left edge constrained, distributed load applied on the right edge
- Inverse objective: displacement misfit plus Tikhonov smoothness regularization on the modulus field

## Dependencies

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

Current required packages:

- `numpy>=1.21.0`
- `scipy>=1.7.0`
- `matplotlib>=3.4.0`

## Current Default Configuration

The main defaults are defined in `config.py`, which now returns typed configuration objects:

- geometry: `9.0 x 9.0`
- mesh: `39 x 39`
- total load: `0.01`
- Poisson ratio: `0.3`
- modulus targets: `Ex = 4.0`, `Ey = 0.25`
- distribution type: `bil`
- GRF defaults (used when `DIS_TYPE = 'grf'`): `E_max = 8.0`, `sigma_g = 1.0`, `ell = 25.0 mm`, `seed = 123`
- default regularization parameter: `1e-6`

## Typical Workflow

1. Edit `config.py` if you want to change geometry, mesh density, material distribution, or inverse settings.
2. Run the forward problem:

```bash
python -m script.forward_job
```

3. Run one of the inverse workflows:

Recommended, because it includes automatic gamma selection and a final independent rerun with the selected `gamma`:

```bash
python -m script.inverse_l_curve
```

Alternative single-gamma L-BFGS-B workflow:

```bash
python -m script.inverse_main
```

4. Regenerate plots from saved result files if needed:

```bash
python -m script.plot_forward_results
python -m script.plot_inverse_results
python -m script.plot_lcurve
```

## Output Files

The main forward and inverse workflows continue to write generated case folders
to the repository root. These outputs remain ignored by Git.

The scripts save results into a folder named like:

```text
Geo_9x9_Mesh_39x39_Alpha_0.3333_Beta_-0.0833_Gamma_1e-06
```

Inverse-result files are now organized in a noise-specific subfolder, for example:

```text
Geo_9x9_Mesh_39x39_Alpha_0.3333_Beta_-0.0833_Gamma_1e-06/
|-- noise_0.00pct/
|   |-- inverse_results.pkl
|   |-- lcurve_analysis.pkl
|   |-- reconstruction_results.png
|   |-- iteration_history.png
|   |-- gradient_field.png
|   |-- reconstruction_comparison.png
|   |-- lcurve.png
|   `-- curvature_vs_gamma.png
```

Typical generated files include:

- `forward_problem_data.pkl`
- `forward_results.png`
- `forward_results.pdf`
- `noise_XX.XXpct/inverse_results.pkl`
- `noise_XX.XXpct/lcurve_analysis.pkl`
- `noise_XX.XXpct/reconstruction_results.png`
- `noise_XX.XXpct/iteration_history.png`
- `noise_XX.XXpct/gradient_field.png`
- `noise_XX.XXpct/reconstruction_comparison.png`
- additional exported figures from the plotting scripts

Most plotting utilities automatically search for the most recently modified result folder that matches the `Geo_*_Mesh_*_Alpha_*_Beta_*_Gamma_*` pattern.

The entry points resolve the repository root from their own location, so they
write and search the same root-level case folders even when launched with
`python -m script.<module>`.

## Repository Layout

### Top-level Scripts

| File | Purpose |
| --- | --- |
| `config.py` | Central configuration entry point for forward, inverse, and L-curve runs. It now returns typed config objects instead of loose dictionaries. |
| `script/forward_job.py` | Main forward-analysis entry point. Generates the mesh, builds the modulus field, applies boundary conditions, solves the FE system, and saves data and plots. |
| `script/inverse_main.py` | Single-gamma inverse solver entry point based on the shared SciPy L-BFGS-B implementation in `fgm_asm/inverse_solver.py`. Saves results into noise-specific subdirectories. |
| `script/inverse_l_curve.py` | Inverse solver entry point using SciPy L-BFGS-B together with L-curve based gamma selection, followed by a fresh rerun from the default initialization using the selected `gamma`. |
| `script/check_regularization_gradient.py` | Finite-difference verification script for the analytical Tikhonov gradient. Useful when changing the regularization code. |
| `script/plot_forward_results.py` | Reloads saved forward results and exports standalone forward-problem figures. |
| `script/plot_inverse_results.py` | Reloads saved inverse results and exports reconstruction, convergence, gradient, and scan-vs-rerun comparison plots. |
| `script/plot_lcurve.py` | Reloads saved L-curve data and exports the L-curve and curvature figures. |
| `grid_study/demo_data.py` | Loads demo data and source-generation metadata; exposes parameter-driven mesh generation and reference-data validation. |
| `grid_study/distributions.py` | Generates selected BIL/EXP fields from `alpha/beta` and selected GRF fields from stored GRF parameters. |
| `grid_study/inverse_convergence.py` | Runs a resumable fixed-gamma inverse mesh/noise study after gamma is selected on an explicit reference grid. |
| `docs/branch_scope.md` | Lightweight branch-management note describing the role of `main`, `disp_linear`, and `disp_nonlinear`. |
| `docs/validation.md` | Lightweight validation checklist and record template for each research branch. |

### Package Modules

| File | Purpose |
| --- | --- |
| `fgm_asm/__init__.py` | Package export surface for the main solver utilities. |
| `fgm_asm/config_types.py` | Dataclass-based configuration contracts for forward, inverse, and L-curve workflows, plus coercion helpers for backward-compatible loading. |
| `fgm_asm/mesh.py` | Structured mesh generation, quadrature shape functions, sparse indexing, cached geometry-only FE data, mass-matrix assembly, regularization-matrix assembly, body-force loading, and `setup_boundary_conditions`. |
| `fgm_asm/material.py` | Material container class and utilities for generating bilinear, exponential, or GRF modulus fields. |
| `fgm_asm/grf.py` | Reproducible single-case RBF Gaussian random-field generation on structured meshes. |
| `fgm_asm/fem_forward.py` | FE assembly and forward/adjoint linear solves with reusable stiffness-factorization support. |
| `fgm_asm/regularization.py` | Matrix-based Tikhonov regularization value and gradient with respect to nodal modulus variables. |
| `fgm_asm/inverse_solver.py` | Adjoint-based gradient terms and the shared SciPy L-BFGS-B inverse solver with cached objective-state reuse. |
| `fgm_asm/l_curve.py` | Gamma sweep, curvature-based L-curve corner detection, and L-curve plotting. |
| `fgm_asm/results_io.py` | Centralized result discovery, path conventions, pickle save/load helpers, and noise-specific output-directory management. |
| `fgm_asm/workflows.py` | Shared script-level helpers for loading the latest forward dataset and resolving result directories. |
| `fgm_asm/utils.py` | Small numerical helpers for adding displacement noise and computing reconstruction errors. |
| `fgm_asm/visualization.py` | Unified plotting module for both workflow-level visualization and reusable figure helpers used by the standalone plotting scripts. |

## Forward Problem

`script/forward_job.py` performs the following steps:

1. reads the forward configuration from `config.py`
2. creates a structured rectangular mesh
3. generates the FGM modulus field
4. applies the boundary conditions from `fgm_asm.mesh.setup_boundary_conditions`
5. assembles the FE system and solves for displacement
6. saves the results as a pickle file
7. exports forward-result figures

Set `DIS_TYPE = 'grf'` in `config.py` to use the single-case Gaussian
random-field generator. It follows `GRF_Generate.m` using an RBF covariance,
the `tanh` map to `[0, E_max]`, and a reproducible seed. The batch dataset
dimension and `E_max` shuffle from the MATLAB data-generation script are not
part of this repository's one-case forward workflow.

## Imported Demo Distributions

The repository also contains a compact, solver-facing copy of the three demo
cases from `comparison_sample_GN`:

```text
demo_distributions/
|-- demo_cases.json
|-- bil/sample_600/true_modulus.npy
|-- exp/sample_200/true_modulus.npy
`-- grf/sample_410/true_modulus.npy
```

Each case has `noise_0`, `noise_2`, `noise_4`, `noise_6`, `noise_8`, and
`noise_10` measured-displacement files. `demo_cases.json` is the single source
of truth for sample indices, original batch indices, BIL/EXP coefficients,
GRF parameters, geometry, and noise seeds. Legacy ASM predictions, L-curve
files, and U-Net outputs are deliberately excluded.

Run the self-contained demo-data forward grid study by directly running:

```text
script/forward_grid_convergence.py
```

To run the original parameter-driven BIL/EXP grid study instead, set
`EX` and `EY` in `config.py`, then directly run:

```text
script/forward_original_grid_convergence.py
```

This keeps the original `10, 20, 30, 40, 60, 80, 100` node cases and writes
its results to `results/grid_study/forward_original`.

The demo runner uses `demo_distributions` from this repository by default.
The original runner sets `demo_root=None`, so BIL and EXP fields are evaluated
from the `EX` and `EY` values on every target mesh, while the GRF field is
generated from the configured GRF parameters without loading demo data.

The inverse grid study also uses the repository copy by default. For every
target mesh it solves the forward problem for the parameter-generated true
field and creates that mesh's requested noise level from the clean
displacement. The imported displacement files remain reference data for the
40 x 40 demo case. It still requires an explicit `--reference-nodes` and a
gamma selected beforehand on that reference grid:

```bash
python -m grid_study.inverse_convergence --reference-nodes 40 --gamma 1e-6
```

## Inverse Problem Options

### `script/inverse_l_curve.py`

This is the most complete end-to-end inverse workflow in the current repository:

- loads the latest forward result set
- assembles the mass matrix
- uses a single configured noise level per run
- sweeps `gamma` values on a log scale
- runs warm-started SciPy L-BFGS-B solves
- selects the best `gamma` using maximum L-curve curvature
- reruns the inverse problem from the default initialization with the selected `gamma`
- saves both the L-curve scan data and the final rerun result
- stores the scan-optimal solution inside `inverse_results.pkl` so it can be compared with the final rerun

### `script/inverse_main.py`

This script uses the same SciPy L-BFGS-B implementation as the L-curve workflow, but with a fixed user-specified `gamma`.

It saves each noise case into its own `noise_XX.XXpct` subdirectory. The current repository is still more polished around the L-curve workflow, so `inverse_l_curve.py` remains the safer default entry point.

## Plotting

`script/plot_inverse_results.py` now generates:

- the final rerun reconstruction figure
- iteration-history plots for the final rerun
- gradient-field plots for the final rerun
- a comparison figure between the L-curve scan optimum and the final rerun with the selected `gamma`

The comparison plot is useful when you want to see whether warm-start continuation during the L-curve sweep materially changes the selected solution compared with a clean rerun from the default initialization.

## Notes

- The codebase now separates concerns more explicitly.
- `config.py` defines typed runtime configuration objects.
- `results_io.py` owns result-file naming, directory layout, and pickle persistence.
- `workflows.py` owns shared top-level script loading logic.
- `visualization.py` is the single plotting module for reusable plot helpers and script-facing visualization entry points.
- `mesh.py` now precomputes geometry-only quadrature data once per mesh and assembles the regularization matrix once for reuse.
- `fem_forward.py` reuses one stiffness factorization for both forward and adjoint solves inside the same inverse-state evaluation.
- `inverse_solver.py` uses one cached L-BFGS-B evaluation path and avoids duplicate forward solves in callbacks and L-curve post-processing.
- `setup_boundary_conditions` now lives in `fgm_asm/mesh.py`.
- `script/forward_job.py` saves figures without blocking on an interactive Matplotlib window.
- `script/inverse_l_curve.py` processes one noise value per run. If `config.py` provides multiple values, the script currently uses the first one.
- Plotting scripts rely on saved pickle files rather than rerunning the solver.
- Result files are stored as Python pickle objects, which is convenient for internal reuse but not intended as a stable interchange format.
- For research branching, keep each branch's scope, output layout, and validation notes documented even if the code stays largely independent.

## Suggested Starting Point

If you are new to this repository, start here:

```bash
python -m script.forward_job
python -m script.inverse_l_curve
python -m script.plot_forward_results
python -m script.plot_inverse_results
python -m script.plot_lcurve
```
