from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..schemas import ClusterSpec, DeviceSpec, ModelMeta, Plan
from ..cluster.grouping import DeviceGroup
from ..profile.profile_store import ProfileStore


@dataclass
class PlanSimResult:
    iter_time_s: float
    valid: bool
    oom: bool
    details: Dict[str, float]


def _pairwise_bw_min(cluster: ClusterSpec, devices: List[DeviceSpec]) -> float:
    if len(devices) <= 1:
        return float("inf")
    m = float("inf")
    for i in range(len(devices)):
        for j in range(i + 1, len(devices)):
            m = min(m, cluster.device_bw_gbps(devices[i], devices[j]))
    return m


def _activation_bytes_per_microbatch(model: ModelMeta, microbatch: int, bytes_per_token: int) -> int:
    hidden = int(model.hidden_size or 768)
    seqlen = int(model.seq_len or 1024)
    # very rough: activation tensor ~ microbatch * seqlen * hidden
    return int(microbatch * seqlen * hidden * bytes_per_token)


def estimate_stage_memory_gb(
    *,
    model: ModelMeta,
    stage_layer_params: int,
    microbatch: int,
    tp: int,
    bytes_per_param: int,
    activation_bytes_per_token: int,
    recompute: bool = False,
) -> float:
    # Params are sharded by TP
    param_bytes = int(stage_layer_params * bytes_per_param / max(1, tp))
    # Activations: very rough; reduced by recompute
    act_bytes = _activation_bytes_per_microbatch(model, microbatch, activation_bytes_per_token)
    act_bytes = int(act_bytes * (0.35 if recompute else 1.0))
    # optimizer states ignored in this simulated planner
    total = param_bytes + act_bytes
    return float(total) / (1024**3)


def simulate_plan_cost(
    *,
    cluster: ClusterSpec,
    model: ModelMeta,
    plan: Plan,
    device_groups: List[DeviceGroup],
    profile: ProfileStore,
    bytes_per_param: int = 2,
    activation_bytes_per_token: int = 2,
    comm_alpha: float = 1.0,
    compute_alpha: float = 1.0,
    penalty_invalid: float = 1e6,
    penalty_oom: float = 1e6,
) -> PlanSimResult:
    """Return a simulated iteration time in seconds (lower is better)."""
    details: Dict[str, float] = {}

    # basic validation
    try:
        plan.validate_basic(total_devices=cluster.world_size())
    except Exception:
        return PlanSimResult(iter_time_s=float(penalty_invalid), valid=False, oom=False, details={"invalid": 1.0})

    # per-stage compute time
    stage_times: List[float] = []
    stage_mem_gb: List[float] = []
    dp_comm_times: List[float] = []
    tp_comm_times: List[float] = []

    for st in plan.stages:
        grp = device_groups[st.device_group_id]
        # stage layer params
        layer_params = sum(model.per_layer_params[st.layer_start : st.layer_end + 1])
        comp_t = compute_alpha * sum(
            profile.layer_time_s(layer_params=p, device_group_compute_sum=grp.compute_sum)
            for p in model.per_layer_params[st.layer_start : st.layer_end + 1]
        )

        # TP comm per layer ~ params/tp (toy)
        bw_tp = max(1e-6, _pairwise_bw_min(cluster, grp.devices))
        tp_bytes = int(layer_params * bytes_per_param / max(1, st.tp))
        tp_t = comm_alpha * (tp_bytes * 8.0 / (bw_tp * 1e9))  # seconds (Gbps)
        # DP all-reduce ~ params
        bw_dp = max(1e-6, _pairwise_bw_min(cluster, grp.devices))
        dp_bytes = int(layer_params * bytes_per_param)
        dp_t = comm_alpha * (dp_bytes * 8.0 / (bw_dp * 1e9))

        stage_times.append(comp_t + tp_t + dp_t)
        tp_comm_times.append(tp_t)
        dp_comm_times.append(dp_t)

        # memory
        mem = estimate_stage_memory_gb(
            model=model,
            stage_layer_params=layer_params,
            microbatch=plan.microbatch_size,
            tp=st.tp,
            bytes_per_param=bytes_per_param,
            activation_bytes_per_token=activation_bytes_per_token,
            recompute=bool(plan.meta.get("recompute", False)),
        )
        stage_mem_gb.append(mem)

        if mem > grp.mem_min_gb and grp.mem_min_gb > 0:
            # OOM penalty
            return PlanSimResult(iter_time_s=float(penalty_oom), valid=True, oom=True, details={"oom": 1.0, "mem_gb": mem, "mem_limit_gb": grp.mem_min_gb})

    max_stage = max(stage_times) if stage_times else 0.0
    sum_stage = sum(stage_times)

    # Pipeline bubble model
    m = max(1, plan.microbatch_size)
    if plan.schedule.lower() == "1f1b":
        bubble = (plan.pp - 1) * max_stage / m
    else:  # gpipe-like
        bubble = (plan.pp - 1) * max_stage

    # activation communication between adjacent stages
    act_bytes = _activation_bytes_per_microbatch(model, plan.microbatch_size, activation_bytes_per_token)
    pp_comm = 0.0
    for i in range(plan.pp - 1):
        g0 = device_groups[plan.stages[i].device_group_id]
        g1 = device_groups[plan.stages[i + 1].device_group_id]
        # approximate bw between groups as min bw between any device pair across groups
        bw = float("inf")
        for a in g0.devices:
            for b in g1.devices:
                bw = min(bw, cluster.device_bw_gbps(a, b))
        if bw == float("inf"):
            bw = cluster.links.inter_node_gbps
        pp_comm += (act_bytes * 8.0 / (max(1e-6, bw) * 1e9))

    iter_time = sum_stage + bubble + pp_comm

    details.update(
        {
            "sum_stage_s": sum_stage,
            "max_stage_s": max_stage,
            "bubble_s": bubble,
            "pp_comm_s": pp_comm,
            "tp_comm_s": sum(tp_comm_times),
            "dp_comm_s": sum(dp_comm_times),
            "peak_mem_gb": max(stage_mem_gb) if stage_mem_gb else 0.0,
        }
    )
    return PlanSimResult(iter_time_s=float(iter_time), valid=True, oom=False, details=details)
