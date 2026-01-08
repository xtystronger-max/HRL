from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LayerComputeProfile:
    """Optional: measured per-layer compute time coefficients."""
    # time_s = coeff * params / compute_score
    coeff: float = 1e-12


@dataclass
class ProfileStore:
    """A minimal profile store. You can later replace this with real microbenchmarks."""
    default: LayerComputeProfile = field(default_factory=LayerComputeProfile)

    def layer_time_s(self, *, layer_params: int, device_group_compute_sum: float, coeff: Optional[float] = None) -> float:
        c = coeff if coeff is not None else self.default.coeff
        denom = max(1e-9, device_group_compute_sum)
        return float(c * layer_params / denom)
