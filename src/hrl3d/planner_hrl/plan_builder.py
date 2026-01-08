from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..schemas import ClusterSpec, ModelMeta, PipelineStage, Plan, DeviceSpec, NodeSpec
from ..cluster.grouping import DeviceGroup, build_device_groups


@dataclass
class GroupingConfig:
    min_group_size: int = 1
    max_group_size: int = 8
    prefer_same_vendor: bool = True
    prefer_same_backend: bool = True

    # Torchrun constraint: nproc-per-node must be equal across nodes.
    # If enabled, we only use a subset of devices per node (=nproc_per_node) and a subset of nodes,
    # such that total_used_devices = nproc_per_node * nnodes is divisible by pp.
    enforce_uniform_nproc_per_node: bool = True
    nproc_per_node: Optional[int] = None


def _select_dominant_bucket(cluster: ClusterSpec) -> Tuple[str, str]:
    """Select a (vendor, backend) bucket to plan on (avoid mixing vendors by default)."""
    counts: Dict[Tuple[str, str], int] = {}
    compute: Dict[Tuple[str, str], float] = {}
    for d in cluster.all_devices():
        key = (d.vendor, d.backend)
        counts[key] = counts.get(key, 0) + 1
        compute[key] = compute.get(key, 0.0) + float(d.compute_score)
    best = None
    for k in counts:
        if best is None:
            best = k
        else:
            if counts[k] > counts[best] or (counts[k] == counts[best] and compute[k] > compute[best]):
                best = k
    return best if best is not None else ("unknown", "")


def _top_k_devices_per_node(cluster: ClusterSpec, vendor: str, backend: str, k: int, *, same_vendor: bool, same_backend: bool) -> Dict[str, List[DeviceSpec]]:
    out: Dict[str, List[DeviceSpec]] = {}
    for n in cluster.nodes:
        devs = []
        for d in n.devices:
            if same_vendor and d.vendor != vendor:
                continue
            if same_backend and d.backend != backend:
                continue
            devs.append(d)
        devs.sort(key=lambda x: (x.compute_score, x.mem_gb), reverse=True)
        if len(devs) >= k:
            out[n.name] = devs[:k]
    return out


def _choose_uniform_launch_set(cluster: ClusterSpec, *, pp: int, cfg: GroupingConfig, vendor: str, backend: str) -> Tuple[Dict[str, List[DeviceSpec]], int]:
    """
    Choose (selected_devices_per_node, nproc_per_node) under torchrun constraint.
    We search over nproc_per_node and node subset sizes to maximize total compute, while ensuring
    total_used_devices divisible by pp.
    """
    best_score = None
    best_selection: Dict[str, List[DeviceSpec]] = {}
    best_nproc = 0

    nproc_candidates = [cfg.nproc_per_node] if cfg.nproc_per_node else list(range(cfg.min_group_size, cfg.max_group_size + 1))
    for nproc in nproc_candidates:
        per_node = _top_k_devices_per_node(cluster, vendor, backend, nproc, same_vendor=cfg.prefer_same_vendor, same_backend=cfg.prefer_same_backend)
        if not per_node:
            continue
        # score each node by compute sum of selected devices
        node_scores = [(name, sum(d.compute_score for d in devs)) for name, devs in per_node.items()]
        node_scores.sort(key=lambda x: x[1], reverse=True)

        # try different nnodes counts
        for nnodes in range(1, len(node_scores) + 1):
            total_used = nproc * nnodes
            if total_used < pp:
                continue
            if total_used % pp != 0:
                continue
            chosen = dict((name, per_node[name]) for name, _ in node_scores[:nnodes])
            score = sum(s for _, s in node_scores[:nnodes])
            # Prefer more devices slightly to stabilize throughput scaling
            score = score * (1.0 + 0.02 * total_used)
            if best_score is None or score > best_score:
                best_score = score
                best_selection = chosen
                best_nproc = nproc

    return best_selection, best_nproc


def build_stage_device_groups(
    cluster: ClusterSpec,
    *,
    pp: int,
    grouping_cfg: GroupingConfig,
) -> Tuple[List[DeviceGroup], List[DeviceSpec], int]:
    """
    Build exactly `pp` stage groups with equal group size, and (optionally) torchrun-feasible
    uniform nproc-per-node across nodes. Nodes may have different device counts; unused devices are allowed.

    Returns (groups, unused_devices, stage_group_size).
    """
    assert pp >= 1
    total = cluster.world_size()
    if total <= 0:
        return [], [], 0

    vendor, backend = _select_dominant_bucket(cluster)

    # Step 1: choose the set of devices we will actually use for execution
    if grouping_cfg.enforce_uniform_nproc_per_node:
        selected_per_node, nproc = _choose_uniform_launch_set(cluster, pp=pp, cfg=grouping_cfg, vendor=vendor, backend=backend)
        if not selected_per_node or nproc <= 0:
            return [], cluster.all_devices(), 0
        selected_nodes = list(selected_per_node.keys())
        selected_devs: List[DeviceSpec] = []
        for n in selected_nodes:
            selected_devs.extend(selected_per_node[n])
        used_devices = selected_devs
    else:
        # no uniform constraint: just use dominant bucket
        used_devices = []
        for d in cluster.all_devices():
            if grouping_cfg.prefer_same_vendor and d.vendor != vendor:
                continue
            if grouping_cfg.prefer_same_backend and d.backend != backend:
                continue
            used_devices.append(d)

    used_total = len(used_devices)
    if used_total < pp:
        return [], cluster.all_devices(), 0

    # Stage group size must be equal: used_total divisible by pp
    if used_total % pp != 0:
        # drop some devices (still keep uniform per-node if enabled by dropping full nodes)
        if grouping_cfg.enforce_uniform_nproc_per_node:
            # drop the weakest node until divisible
            # (keeps nproc-per-node uniform)
            # compute node strengths
            per_node = {}
            for d in used_devices:
                per_node.setdefault(d.node, []).append(d)
            node_strengths = [(n, sum(dd.compute_score for dd in devs)) for n, devs in per_node.items()]
            node_strengths.sort(key=lambda x: x[1])  # weakest first
            while used_total % pp != 0 and node_strengths:
                n, _ = node_strengths.pop(0)
                used_devices = [d for d in used_devices if d.node != n]
                used_total = len(used_devices)
            if used_total < pp or used_total % pp != 0:
                return [], cluster.all_devices(), 0
        else:
            # drop weakest devices
            used_devices.sort(key=lambda x: (x.compute_score, x.mem_gb))
            while used_total % pp != 0 and used_devices:
                used_devices.pop(0)
                used_total -= 1
            if used_total < pp:
                return [], cluster.all_devices(), 0

    stage_group_size = used_total // pp
    if stage_group_size < grouping_cfg.min_group_size or stage_group_size > grouping_cfg.max_group_size:
        # clamp by dropping devices
        target = max(grouping_cfg.min_group_size, min(grouping_cfg.max_group_size, stage_group_size))
        used_total = target * pp
        if grouping_cfg.enforce_uniform_nproc_per_node:
            # recompute with that constraint by selecting nproc-per-node and nodes again
            grouping_cfg2 = GroupingConfig(**{**grouping_cfg.__dict__})
            grouping_cfg2.min_group_size = min(grouping_cfg.min_group_size, grouping_cfg.max_group_size)
            grouping_cfg2.max_group_size = grouping_cfg.max_group_size
            # force nproc to <= max_group_size
            # leave selection to algorithm; if fails, bail.
            selected_per_node, _nproc = _choose_uniform_launch_set(cluster, pp=pp, cfg=grouping_cfg, vendor=vendor, backend=backend)
            if not selected_per_node:
                return [], cluster.all_devices(), 0
            used_devices = []
            for n, devs in selected_per_node.items():
                used_devices.extend(devs)
            # re-check
            used_total = len(used_devices)
            if used_total < pp or used_total % pp != 0:
                return [], cluster.all_devices(), 0
            stage_group_size = used_total // pp
        else:
            # take top used_total devices
            used_devices.sort(key=lambda x: (x.compute_score, x.mem_gb), reverse=True)
            used_devices = used_devices[:used_total]
            stage_group_size = used_total // pp

    # Step 2: create a temp cluster with only used devices for grouping
    temp_nodes: List[NodeSpec] = []
    used_set = set((d.node, d.local_id, d.vendor, d.backend) for d in used_devices)
    for n in cluster.nodes:
        devs = [d for d in n.devices if (d.node, d.local_id, d.vendor, d.backend) in used_set]
        if devs:
            temp_nodes.append(NodeSpec(name=n.name, devices=devs))
    from ..schemas import ClusterSpec as CS
    temp_cluster = CS(nodes=temp_nodes, links=cluster.links)

    groups_all, unused_tmp = build_device_groups(
        temp_cluster,
        group_size=stage_group_size,
        prefer_same_vendor=grouping_cfg.prefer_same_vendor,
        prefer_same_backend=grouping_cfg.prefer_same_backend,
    )
    # groups_all should already be exactly pp, but keep defensive
    groups_all = sorted(groups_all, key=lambda g: (g.compute_sum, g.bw_min_gbps, g.mem_min_gb), reverse=True)
    groups = groups_all[:pp]
    # Re-index group ids
    for i, g in enumerate(groups):
        g.group_id = i

    # unused devices = devices not in groups + devices excluded by selection
    selected_in_groups = set((d.node, d.local_id, d.vendor, d.backend) for g in groups for d in g.devices)
    unused: List[DeviceSpec] = []
    for d in cluster.all_devices():
        key = (d.node, d.local_id, d.vendor, d.backend)
        if key not in selected_in_groups:
            unused.append(d)

    return groups, unused, stage_group_size


def _effective_group_capacity(g: DeviceGroup) -> float:
    # include comm and memory in a mild way
    return float(g.compute_sum) * (1.0 + float(g.bw_min_gbps) / 100.0) ** 0.25 * (1.0 + float(g.mem_min_gb) / 16.0) ** 0.25


def partition_layers_to_groups(model: ModelMeta, groups: List[DeviceGroup]) -> List[Tuple[int, int, int]]:
    """Return list of (layer_start, layer_end, group_id) for each stage in layer order."""
    pp = len(groups)
    if pp <= 0:
        return []
    caps = [_effective_group_capacity(g) for g in groups]
    # assign stages to groups (fastest gets more layers)
    order = sorted(range(pp), key=lambda i: caps[i], reverse=True)

    layer_params = model.per_layer_params
    total_params = sum(layer_params)
    total_cap = sum(caps) if sum(caps) > 0 else float(pp)

    targets = [total_params * (caps[i] / total_cap) for i in order]

    spans: List[Tuple[int, int, int]] = []
    cur = 0
    for si, gid in enumerate(order):
        if si == pp - 1:
            spans.append((cur, model.num_layers - 1, gid))
            break
        min_left = pp - si - 1
        max_end = model.num_layers - min_left - 1
        best_end = cur
        best_diff = None
        s = 0
        for end in range(cur, max_end + 1):
            s += layer_params[end]
            diff = abs(s - targets[si])
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_end = end
        spans.append((cur, best_end, gid))
        cur = best_end + 1

    return sorted(spans, key=lambda x: x[0])


def build_plan_from_actions(
    *,
    cluster: ClusterSpec,
    model: ModelMeta,
    pp: int,
    schedule: str,
    microbatch: int,
    tp: int,
    recompute: bool,
    overlap: str,
    grouping_cfg: GroupingConfig,
) -> Tuple[Plan, List[DeviceGroup], List[DeviceSpec]]:
    groups, unused, group_size = build_stage_device_groups(cluster, pp=pp, grouping_cfg=grouping_cfg)
    if not groups or group_size <= 0 or len(groups) != pp:
        plan = Plan(
            pp=pp,
            stages=[],
            microbatch_size=microbatch,
            schedule=schedule,
            device_groups=[],
            meta={"invalid_reason": "cannot_build_groups"},
        )
        return plan, [], cluster.all_devices()

    spans = partition_layers_to_groups(model, groups)

    # ensure tp divides group_size
    if group_size % max(1, tp) != 0:
        tp = 1
    dp = group_size // max(1, tp)

    stages: List[PipelineStage] = []
    for sid, (ls, le, gid) in enumerate(spans):
        stages.append(
            PipelineStage(
                stage_id=sid,
                layer_start=int(ls),
                layer_end=int(le),
                device_group_id=int(gid),
                tp=int(tp),
                dp=int(dp),
            )
        )

    # torchrun launch hints
    per_node_used = {}
    for g in groups:
        for d in g.devices:
            per_node_used.setdefault(d.node, set()).add(d.local_id)
    nproc_per_node = min(len(v) for v in per_node_used.values()) if per_node_used else 0

    plan = Plan(
        pp=pp,
        stages=stages,
        microbatch_size=int(microbatch),
        schedule=str(schedule),
        device_groups=[g.to_index_list() for g in groups],
        meta={
            "group_size": group_size,
            "vendor": groups[0].vendor if groups else None,
            "backend": groups[0].backend if groups else None,
            "recompute": bool(recompute),
            "overlap": str(overlap),
            "unused_devices": [(d.node, d.local_id) for d in unused],
            "launch": {
                "torchrun_nproc_per_node": nproc_per_node,
                "torchrun_nnodes": len(per_node_used),
                "per_node_visible_devices": {k: sorted(list(v)) for k, v in per_node_used.items()},
            },
        },
    )
    return plan, groups, unused
