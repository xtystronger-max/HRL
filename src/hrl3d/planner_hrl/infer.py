from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ..cluster.loader import cluster_from_dict
from ..cost_model.simulator import simulate_plan_cost
from ..errors import OptionalDependencyError
from ..io import load_json, load_yaml
from ..profile.profile_store import ProfileStore
from ..schemas import ClusterSpec, ModelMeta, Plan
from .obs import build_observation
from .plan_builder import GroupingConfig, build_plan_from_actions, build_stage_device_groups
from .policies import HRLPolicies, _require_torch
from .spaces import ActionSpace


def load_policies(checkpoint_dir: str | Path, action_space: ActionSpace, input_dim: int) -> HRLPolicies:
    torch = _require_torch()
    ckpt_dir = Path(checkpoint_dir)

    policies = HRLPolicies(action_space=action_space, input_dim=input_dim)
    high = torch.load(ckpt_dir / "high.pt", map_location="cpu")
    low = torch.load(ckpt_dir / "low.pt", map_location="cpu")
    policies.high.net.load_state_dict(high["state_dict"])
    policies.low.net.load_state_dict(low["state_dict"])
    policies.high.net.eval()
    policies.low.net.eval()
    return policies


def plan_with_hrl(
    *,
    cluster: ClusterSpec,
    model: ModelMeta,
    planner_cfg: Dict[str, Any],
    checkpoint_dir: str | Path,
    num_samples: int = 64,
) -> Tuple[Plan, Dict[str, Any]]:
    tc = planner_cfg
    action_space = ActionSpace(
        pp_candidates=list(tc.get("pp_candidates", [1, 2, 4])),
        schedule_candidates=list(tc.get("schedule_candidates", ["1f1b", "gpipe"])),
        microbatch_candidates=list(tc.get("microbatch_candidates", [1, 2, 4, 8])),
        tp_candidates=list(tc.get("tp_candidates", [1, 2, 4])),
        recompute_candidates=[False, True],
        overlap_candidates=["none", "tp_dp", "pp_tp"],
    )
    grouping_cfg = GroupingConfig(
        min_group_size=int((tc.get("grouping", {}) or {}).get("min_group_size", 1)),
        max_group_size=int((tc.get("grouping", {}) or {}).get("max_group_size", 8)),
        prefer_same_vendor=bool((tc.get("grouping", {}) or {}).get("prefer_same_vendor", True)),
        prefer_same_backend=True,
        enforce_uniform_nproc_per_node=bool((tc.get("grouping", {}) or {}).get("enforce_uniform_nproc_per_node", True)),
        nproc_per_node=(tc.get("grouping", {}) or {}).get("nproc_per_node", None),
    )

    groups_obs, _, _ = build_stage_device_groups(cluster, pp=max(action_space.pp_candidates), grouping_cfg=grouping_cfg)
    obs_vec = build_observation(cluster=cluster, model=model, groups=groups_obs, use_gnn=bool(planner_cfg.get('use_gnn', False)), gnn_dim=int(planner_cfg.get('gnn_dim', 64))).vec
    policies = load_policies(checkpoint_dir, action_space, input_dim=int(obs_vec.shape[0]))

    profile = ProfileStore()
    sim_cfg = tc.get("sim", {}) or {}
    bytes_per_param = int(sim_cfg.get("bytes_per_param", 2))
    activation_bytes = int(sim_cfg.get("activation_bytes_per_token", 2))
    comm_alpha = float(sim_cfg.get("comm_alpha", 1.0))
    compute_alpha = float(sim_cfg.get("compute_alpha", 1.0))
    penalty_invalid = float(tc.get("penalty_invalid", 1e6))
    penalty_oom = float(tc.get("penalty_oom", 1e6))

    best_plan = None
    best_cost = None
    best_details = None

    for _ in range(num_samples):
        high, _, _ = policies.sample_high(obs_vec)
        group_size = max(1, cluster.world_size() // max(1, high.pp))
        low, _, _ = policies.sample_low(obs_vec, stage_group_size=group_size)

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
            penalty_invalid=penalty_invalid,
            penalty_oom=penalty_oom,
        )
        cost = float(res.iter_time_s)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_plan = plan
            best_details = res.details

    assert best_plan is not None
    best_plan.meta["sim_details"] = best_details or {}
    best_plan.meta["sim_cost_s"] = float(best_cost) if best_cost is not None else None
    return best_plan, {"best_cost_s": best_cost, "details": best_details}
