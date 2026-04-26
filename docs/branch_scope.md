# Branch Scope

This repository is used primarily for research. The preferred strategy is:

- keep `main` as the stable baseline
- develop new physics and loading modes in separate branches
- avoid premature unification unless repeated maintenance pain justifies it

## Active Branch Roles

| Branch | Role | Loading | Kinematics | Constitutive Model |
| --- | --- | --- | --- | --- |
| `main` | Current baseline and reference branch | Force-controlled | Small strain | Linear elastic |
| `disp_linear` | Displacement-controlled linear extension | Displacement-controlled | Small strain | Linear elastic |
| `disp_nonlinear` | Nonlinear research branch | Displacement-controlled | Finite strain | Nonlinear constitutive model(s) |

## Lightweight Rules

1. `main` should always stay runnable and easy to reproduce.
2. Each branch should focus on one research direction only.
3. Branches may evolve independently in solver details, file layout, and saved outputs.
4. Only clearly reusable fixes should be moved back to `main`.
5. Each branch should keep its own short `README` section or note covering scope, assumptions, entry points, outputs, and validation status.

## Suggested Branch Names

- `main`
- `disp_linear`
- `disp_nonlinear`

If more branches are added later, prefer short names based on the main research variable, for example:

- `force_nonlinear`
- `plasticity_tests`
- `hyperelastic_benchmark`

## Output Directory Convention

Recommended top-level result layout:

```text
results/
|-- main/
|-- disp_linear/
`-- disp_nonlinear/
```

Recommended case-folder pattern inside each branch:

```text
results/<branch_name>/<case_name>/
```

Examples:

```text
results/main/Geo_9x9_Mesh_39x39_Alpha_0.1111_Beta_-0.0556_Gamma_1e-06/
results/disp_linear/DispTop_0p10_Geo_9x9_Mesh_39x39/
results/disp_nonlinear/FiniteStrain_NH_DispTop_0p10_Geo_9x9_Mesh_39x39/
```

## Recommended README Sections Per Branch

Each branch README should ideally include:

1. `Scope`
2. `Physics`
3. `Entry Points`
4. `Outputs`
5. `Differences From Main`
6. `Validation Status`

## Merge Guidance

- Merge or cherry-pick to `main`:
  - bug fixes
  - plotting and IO improvements
  - performance optimizations that do not change the branch physics
  - documentation improvements
- Keep branch-local:
  - branch-specific constitutive models
  - branch-specific nonlinear solvers
  - branch-specific load-step logic
  - branch-specific experiment scripts
