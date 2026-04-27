"""
Typed configuration objects used by the FGM solver scripts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class ForwardConfig:
    """Configuration for the forward problem."""

    geo_l: float
    geo_h: float
    nel_x: int
    nel_y: int
    disp_amp: float
    Ex: float
    Ey: float
    nu: float
    dis_type: str

    def to_dict(self) -> dict:
        """Return a plain dictionary version for logging or serialization."""
        return asdict(self)

    def output_folder_name(self, alpha: float, beta: float, gamma: float) -> str:
        """Build the default results folder name."""
        return (
            f"Geo_{int(self.geo_l)}x{int(self.geo_h)}_"
            f"Mesh_{self.nel_x}x{self.nel_y}_"
            f"Alpha_{alpha:.4f}_Beta_{beta:.4f}_"
            f"Gamma_{gamma:.0e}"
        )


@dataclass(frozen=True)
class InverseConfig:
    """Configuration for the inverse problem."""

    gamma: float
    E_min: float
    E_max: float
    max_iter: int
    ftol: float
    gtol: float
    noise_levels: tuple[float, ...]
    nu: float

    @property
    def primary_noise_level(self) -> float:
        """Return the first configured noise level."""
        if not self.noise_levels:
            raise ValueError("noise_levels cannot be empty")
        return self.noise_levels[0]

    def to_dict(self) -> dict:
        """Return a plain dictionary version for logging or serialization."""
        return asdict(self)


@dataclass(frozen=True)
class LCurveConfig:
    """Configuration for the L-curve workflow."""

    gamma_min: float
    gamma_max: float
    n_gamma: int
    E_min: float
    E_max: float
    max_iter: int
    ftol: float
    gtol: float
    nu: float

    def to_dict(self) -> dict:
        """Return a plain dictionary version for logging or serialization."""
        return asdict(self)


def normalize_noise_levels(noise_levels: float | Iterable[float]) -> tuple[float, ...]:
    """Normalize scalar or iterable noise definitions to a tuple of floats."""
    if isinstance(noise_levels, (int, float)):
        return (float(noise_levels),)
    return tuple(float(value) for value in noise_levels)


def coerce_forward_config(config: ForwardConfig | Mapping[str, object]) -> ForwardConfig:
    """Convert a mapping or config-like object to ``ForwardConfig``."""
    if isinstance(config, ForwardConfig):
        return config
    return ForwardConfig(
        geo_l=float(config["geo_l"]),
        geo_h=float(config["geo_h"]),
        nel_x=int(config["nel_x"]),
        nel_y=int(config["nel_y"]),
        disp_amp=float(config["disp_amp"]),
        Ex=float(config["Ex"]),
        Ey=float(config["Ey"]),
        nu=float(config["nu"]),
        dis_type=str(config["dis_type"]),
    )


def coerce_inverse_config(config: InverseConfig | Mapping[str, object]) -> InverseConfig:
    """Convert a mapping or config-like object to ``InverseConfig``."""
    if isinstance(config, InverseConfig):
        return config
    return InverseConfig(
        gamma=float(config["gamma"]),
        E_min=float(config["E_min"]),
        E_max=float(config["E_max"]),
        max_iter=int(config["max_iter"]),
        ftol=float(config["ftol"]),
        gtol=float(config["gtol"]),
        noise_levels=normalize_noise_levels(config["noise_levels"]),
        nu=float(config["nu"]),
    )


def coerce_lcurve_config(config: LCurveConfig | Mapping[str, object]) -> LCurveConfig:
    """Convert a mapping or config-like object to ``LCurveConfig``."""
    if isinstance(config, LCurveConfig):
        return config
    return LCurveConfig(
        gamma_min=float(config["gamma_min"]),
        gamma_max=float(config["gamma_max"]),
        n_gamma=int(config["n_gamma"]),
        E_min=float(config["E_min"]),
        E_max=float(config["E_max"]),
        max_iter=int(config["max_iter"]),
        ftol=float(config["ftol"]),
        gtol=float(config["gtol"]),
        nu=float(config["nu"]),
    )
