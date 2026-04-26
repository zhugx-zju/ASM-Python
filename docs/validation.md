# Validation Notes

This file is a lightweight validation checklist and record template for the research branches in this repository.

## How To Use

- Keep the checklist concise.
- Record only the tests that matter for the current branch.
- Prefer short statements with dates, case names, and outcome summaries.
- If a test fails, keep the failure note instead of deleting it.

## Common Record Template

Use this structure when documenting a branch:

```text
Branch:
Date:
Code revision:

Scope:

Validation cases:
- Case:
  Purpose:
  Expected behavior:
  Observed behavior:
  Status:

Open issues:
- ...
```

## Recommended Minimum Validation By Branch

### `main`

Suggested checks:

- forward linear solve runs successfully on the default case
- reaction forces balance the applied force
- regularization gradient finite-difference check passes
- inverse baseline case reruns reproducibly with the saved configuration
- plotting scripts can reload the latest saved results

### `disp_linear`

Suggested checks:

- prescribed-displacement boundary condition is applied to the intended DOFs
- reaction-force output is reasonable and sign-consistent
- for a simple comparable case, displacement-controlled and force-controlled results are mutually consistent
- small changes in prescribed displacement scale the response linearly
- saved output files and plots are still readable by the branch plotting scripts

### `disp_nonlinear`

Suggested checks:

- load stepping runs without solver breakdown on a simple benchmark
- Newton or nonlinear iterations show consistent residual reduction
- constitutive update behaves correctly on a single-element or single-point benchmark
- finite-strain implementation reproduces the linear branch in the small-deformation limit when applicable
- reaction force versus imposed displacement curve is physically plausible
- state-variable updates are repeatable after reruns from the same input

## Current Status Template

### `main`

- Status: baseline branch
- Last checked:
- Notes:

### `disp_linear`

- Status: planned / in development
- Last checked:
- Notes:

### `disp_nonlinear`

- Status: planned / in development
- Last checked:
- Notes:
