from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..schemas import Plan


def plan_to_megatron_args(plan: Plan) -> Dict[str, Any]:
    """
    Convert a plan into a Megatron-LM friendly argument dict.
    NOTE: Megatron-LM versions vary; treat this as a starting point.

    Returned keys are typical CLI arg names without leading '--'.
    """
    if not plan.stages:
        return {}

    pp = int(plan.pp)
    # All stages share same tp/dp in this simplified planner
    tp = int(plan.stages[0].tp)
    dp = int(plan.stages[0].dp)

    return {
        "pipeline-model-parallel-size": pp,
        "tensor-model-parallel-size": tp,
        "data-parallel-size": dp,
        "micro-batch-size": int(plan.microbatch_size),
        "pipeline-schedule": str(plan.schedule),
    }


def plan_to_rank_mapping(plan: Plan) -> Dict[str, Any]:
    """
    Provide an explicit mapping from device groups to (node, local_id).
    Use this to set CUDA_VISIBLE_DEVICES per node and to validate launch topology.
    """
    return {
        "device_groups": plan.device_groups,
        "launch": plan.meta.get("launch", {}),
        "unused_devices": plan.meta.get("unused_devices", []),
    }
