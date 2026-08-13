"""Manual, independently runnable demo inverse cases.

Edit ``CASES`` to select one or more distribution, mesh, and noise
combinations. Numerical work, saving, and plotting live in ``fgm_asm``.
"""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config as cfg
from fgm_asm.demo_inverse import run_demo_inverse_cases


CASES = [
    {
        "dataset": "bil",
        "nodesx": 40,
        "nodesy": 40,
        "noise_level": 0.0,
        "gamma": None,
        "enable_lcurve": True,
        "lcurve_points": 50,
        "lcurve_gamma_min": 1e-10,
        "lcurve_gamma_max": 1e-8,
        "max_iter": 2000,
        "ftol": 1e-30,
        "gtol": 1e-12,
        "overwrite": False,
    },
]


run_demo_inverse_cases(
    project_root=REPO_ROOT,
    forward_config=cfg.get_forward_config(),
    inverse_config=cfg.get_inverse_config(),
    lcurve_config=cfg.get_lcurve_config(),
    cases=CASES,
)
