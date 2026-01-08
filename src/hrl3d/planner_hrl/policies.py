from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..errors import OptionalDependencyError
from .spaces import ActionSpace, HighAction, LowAction


def _require_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception as e:
        raise OptionalDependencyError("HRL training requires torch. Install with: pip install -e '.[rl]'") from e


@dataclass
class CategoricalPolicy:
    """A simple multi-head categorical policy (REINFORCE)."""
    input_dim: int
    heads: List[int]              # number of categories per head
    hidden_dim: int = 128

    def __post_init__(self) -> None:
        torch = _require_torch()
        self.torch = torch
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_dim, sum(self.heads)),
        )

    def parameters(self):
        return self.net.parameters()

    def _split_logits(self, logits):
        torch = self.torch
        outs = []
        start = 0
        for h in self.heads:
            outs.append(logits[..., start : start + h])
            start += h
        return outs

    def sample(self, obs_vec: np.ndarray) -> Tuple[List[int], "object", float]:
        torch = self.torch
        x = torch.from_numpy(obs_vec).float().unsqueeze(0)
        logits = self.net(x)
        parts = self._split_logits(logits)
        actions: List[int] = []
        dists = []
        logp = 0.0
        for part in parts:
            dist = torch.distributions.Categorical(logits=part)
            a = dist.sample()
            actions.append(int(a.item()))
            logp = logp + float(dist.log_prob(a).item())
            dists.append(dist)
        return actions, dists, float(logp)

    def log_prob(self, obs_vec: np.ndarray, actions: List[int]) -> float:
        torch = self.torch
        x = torch.from_numpy(obs_vec).float().unsqueeze(0)
        logits = self.net(x)
        parts = self._split_logits(logits)
        lp = 0.0
        for part, a in zip(parts, actions):
            dist = torch.distributions.Categorical(logits=part)
            aa = torch.tensor([a], dtype=torch.long)
            lp = lp + dist.log_prob(aa).sum()
        return float(lp.item())


@dataclass
class HRLPolicies:
    action_space: ActionSpace
    input_dim: int

    def __post_init__(self) -> None:
        # High policy heads: pp, schedule, microbatch
        self.high = CategoricalPolicy(
            input_dim=self.input_dim,
            heads=[
                len(self.action_space.pp_candidates),
                len(self.action_space.schedule_candidates),
                len(self.action_space.microbatch_candidates),
            ],
        )
        # Low policy heads: tp, recompute, overlap
        self.low = CategoricalPolicy(
            input_dim=self.input_dim,
            heads=[
                len(self.action_space.tp_candidates),
                len(self.action_space.recompute_candidates),
                len(self.action_space.overlap_candidates),
            ],
        )

    def sample_high(self, obs: np.ndarray) -> Tuple[HighAction, List[int], float]:
        idxs, _, logp = self.high.sample(obs)
        pp = self.action_space.pp_candidates[idxs[0]]
        sched = self.action_space.schedule_candidates[idxs[1]]
        micro = self.action_space.microbatch_candidates[idxs[2]]
        return HighAction(pp=pp, schedule=sched, microbatch=micro), idxs, logp

    def sample_low(self, obs: np.ndarray, *, stage_group_size: int) -> Tuple[LowAction, List[int], float]:
        # Sample tp; enforce divisibility by stage_group_size by resampling a few times.
        for _ in range(8):
            idxs, _, logp = self.low.sample(obs)
            tp = self.action_space.tp_candidates[idxs[0]]
            if stage_group_size % tp == 0:
                recompute = self.action_space.recompute_candidates[idxs[1]]
                overlap = self.action_space.overlap_candidates[idxs[2]]
                return LowAction(tp=tp, recompute=recompute, overlap=overlap), idxs, logp

        # fallback: tp=1
        tp = 1
        recompute = False
        overlap = "none"
        return LowAction(tp=tp, recompute=recompute, overlap=overlap), [0, 0, 0], 0.0
