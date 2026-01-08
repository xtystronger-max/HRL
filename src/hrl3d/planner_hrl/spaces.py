from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class HighAction:
    pp: int
    schedule: str
    microbatch: int


@dataclass
class LowAction:
    tp: int
    recompute: bool
    overlap: str  # placeholder, used in cost model extensions


@dataclass
class ActionSpace:
    pp_candidates: List[int]
    schedule_candidates: List[str]
    microbatch_candidates: List[int]
    tp_candidates: List[int]
    recompute_candidates: List[bool]
    overlap_candidates: List[str]
