from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..cluster.grouping import DeviceGroup
from ..cluster.loader import cluster_from_dict
from ..cost_model.simulator import simulate_plan_cost
from ..errors import OptionalDependencyError
from ..io import dump_json, load_json, load_yaml, dump_yaml
from ..model.introspect import introspect_hf_causal_lm
from ..profile.profile_store import ProfileStore
from ..schemas import ClusterSpec, ModelMeta, Plan
from .obs import build_observation
from .plan_builder import GroupingConfig, build_plan_from_actions
from .policies import HRLPolicies, _require_torch
from .spaces import ActionSpace


@dataclass
class TrainConfig:
    seed: int = 123
    episodes: int = 200
    samples_per_episode: int = 1

    penalty_oom: float = 1e6
    penalty_invalid: float = 1e6
    entropy_bonus: float = 0.01
    baseline_momentum: float = 0.9

    sim: Dict[str, Any] = None

    pp_candidates: list[int] = None
    schedule_candidates: list[str] = None
    microbatch_candidates: list[int] = None
    tp_candidates: list[int] = None

    grouping: Dict[str, Any] = None


def _cfg(d: Dict[str, Any]) -> TrainConfig:
    tc = TrainConfig()
    tc.seed = int(d.get("seed", tc.seed))
    tc.episodes = int(d.get("episodes", tc.episodes))
    tc.samples_per_episode = int(d.get("samples_per_episode", tc.samples_per_episode))
    tc.penalty_oom = float(d.get("penalty_oom", tc.penalty_oom))
    tc.penalty_invalid = float(d.get("penalty_invalid", tc.penalty_invalid))
    tc.entropy_bonus = float(d.get("entropy_bonus", tc.entropy_bonus))
    tc.baseline_momentum = float(d.get("baseline_momentum", tc.baseline_momentum))

    tc.pp_candidates = list(d.get("pp_candidates", [1, 2, 4]))
    tc.schedule_candidates = list(d.get("schedule_candidates", ["1f1b", "gpipe"]))
    tc.microbatch_candidates = list(d.get("microbatch_candidates", [1, 2, 4, 8]))
    tc.tp_candidates = list(d.get("tp_candidates", [1, 2, 4]))

    tc.grouping = dict(d.get("grouping", {}) or {})
    tc.sim = dict(d.get("sim", {}) or {})
    return tc


def train_hrl(
    *,
    cluster: ClusterSpec,
    model: ModelMeta,
    planner_cfg: Dict[str, Any],
    out_dir: str | Path,
) -> None:
    torch = _require_torch()
    np.random.seed(int(planner_cfg.get("seed", 123)))
    random.seed(int(planner_cfg.get("seed", 123)))
    torch.manual_seed(int(planner_cfg.get("seed", 123)))

    tc = _cfg(planner_cfg)

    action_space = ActionSpace(
        pp_candidates=tc.pp_candidates,
        schedule_candidates=tc.schedule_candidates,
        microbatch_candidates=tc.microbatch_candidates,
        tp_candidates=tc.tp_candidates,
        recompute_candidates=[False, True],
        overlap_candidates=["none", "tp_dp", "pp_tp"],
    )

    grouping_cfg = GroupingConfig(
        min_group_size=int(tc.grouping.get("min_group_size", 1)),
        max_group_size=int(tc.grouping.get("max_group_size", 8)),
        prefer_same_vendor=bool(tc.grouping.get("prefer_same_vendor", True)),
        prefer_same_backend=True,
        enforce_uniform_nproc_per_node=bool(tc.grouping.get("enforce_uniform_nproc_per_node", True)),
        nproc_per_node=tc.grouping.get("nproc_per_node", None),
    )

    profile = ProfileStore()

    # Build a provisional group list to build observation; this is only to form a stable obs vector.
    from .plan_builder import build_stage_device_groups
    groups, _, _ = build_stage_device_groups(cluster, pp=max(tc.pp_candidates), grouping_cfg=grouping_cfg)
    obs = build_observation(cluster=cluster, model=model, groups=groups, use_gnn=bool(planner_cfg.get('use_gnn', False)), gnn_dim=int(planner_cfg.get('gnn_dim', 64))).vec
    policies = HRLPolicies(action_space=action_space, input_dim=int(obs.shape[0]))

    # optimizer over both policies
    optim = torch.optim.Adam(list(policies.high.parameters()) + list(policies.low.parameters()), lr=3e-4)

    baseline = None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = tc.sim or {}
    bytes_per_param = int(sim_cfg.get("bytes_per_param", 2))
    activation_bytes = int(sim_cfg.get("activation_bytes_per_token", 2))
    comm_alpha = float(sim_cfg.get("comm_alpha", 1.0))
    compute_alpha = float(sim_cfg.get("compute_alpha", 1.0))

    for ep in range(tc.episodes):
        # Re-build groups for current pp candidate later; observation uses updated best-effort groups.
        total_loss = 0.0
        rewards = []
        costs = []
        for _ in range(tc.samples_per_episode):
            # observation (use pp=max candidate to avoid changing dim)
            groups_obs, _, _ = build_stage_device_groups(cluster, pp=max(tc.pp_candidates), grouping_cfg=grouping_cfg)
            obs_vec = build_observation(cluster=cluster, model=model, groups=groups_obs, use_gnn=bool(planner_cfg.get('use_gnn', False)), gnn_dim=int(planner_cfg.get('gnn_dim', 64))).vec

            high, high_idxs, high_logp = policies.sample_high(obs_vec)
            # Determine stage group size implied by pp (used devices / pp)
            group_size = max(1, cluster.world_size() // max(1, high.pp))
            low, low_idxs, low_logp = policies.sample_low(obs_vec, stage_group_size=group_size)

            plan, groups_plan, _unused = build_plan_from_actions(
                cluster=cluster,
                model=model,
                pp=high.pp,
                schedule=high.schedule,
                microbatch=high.microbatch,
                tp=low.tp,
                recompute=low.recompute,
                overlap=low.overlap,
                grouping_cfg=grouping_cfg,
            )

            # simulate cost
            res = simulate_plan_cost(
                cluster=cluster,
                model=model,
                plan=plan,
                device_groups=groups_plan,
                profile=profile,
                bytes_per_param=bytes_per_param,
                activation_bytes_per_token=activation_bytes,
                comm_alpha=comm_alpha,
                compute_alpha=compute_alpha,
                penalty_invalid=tc.penalty_invalid,
                penalty_oom=tc.penalty_oom,
            )
            cost = float(res.iter_time_s)
            reward = -cost
            rewards.append(reward)
            costs.append(cost)

            # baseline
            if baseline is None:
                baseline = reward
            else:
                baseline = tc.baseline_momentum * baseline + (1.0 - tc.baseline_momentum) * reward

            adv = reward - float(baseline)

            # REINFORCE loss
            # Convert logp scalars to torch tensors for autograd: re-run log_prob
            hlp = policies.high.log_prob(obs_vec, high_idxs)
            llp = policies.low.log_prob(obs_vec, low_idxs)
            loss = -(hlp + llp) * adv

            total_loss = total_loss + loss

        optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(policies.high.parameters()) + list(policies.low.parameters()), 1.0)
        optim.step()

        if (ep + 1) % 10 == 0:
            dump_json(
                {
                    "episode": ep + 1,
                    "mean_cost": float(np.mean(costs)) if costs else None,
                    "best_cost": float(np.min(costs)) if costs else None,
                },
                out_dir / "train_log.json",
            )

    # save checkpoints
    torch.save({"state_dict": policies.high.net.state_dict(), "input_dim": policies.high.input_dim, "heads": policies.high.heads}, out_dir / "high.pt")
    torch.save({"state_dict": policies.low.net.state_dict(), "input_dim": policies.low.input_dim, "heads": policies.low.heads}, out_dir / "low.pt")
    dump_json({"action_space": action_space.__dict__, "grouping": tc.grouping, "sim": tc.sim}, out_dir / "meta.json")
