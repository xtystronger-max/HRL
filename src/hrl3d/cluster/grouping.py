from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..schemas import ClusterSpec, DeviceSpec


@dataclass
class DeviceGroup:
    group_id: int
    devices: List[DeviceSpec]

    # Derived stats
    compute_sum: float
    mem_min_gb: float
    bw_min_gbps: float
    bw_avg_gbps: float

    vendor: str
    backend: str

    def to_index_list(self) -> List[Tuple[str, int]]:
        return [(d.node, d.local_id) for d in self.devices]


def _group_bandwidth_stats(cluster: ClusterSpec, devs: Sequence[DeviceSpec]) -> Tuple[float, float]:
    if len(devs) <= 1:
        return (float("inf"), float("inf"))
    bws: List[float] = []
    for i in range(len(devs)):
        for j in range(i + 1, len(devs)):
            bw = cluster.device_bw_gbps(devs[i], devs[j])
            bws.append(float(bw))
    if not bws:
        return (float("inf"), float("inf"))
    return (min(bws), sum(bws) / len(bws))


def _group_score(
    *,
    compute_sum: float,
    mem_min_gb: float,
    bw_min_gbps: float,
    bw_avg_gbps: float,
    w_compute: float = 1.0,
    w_mem: float = 0.2,
    w_bw: float = 0.2,
) -> float:
    # Conservative: min-mem and min-bw dominate feasibility; avg-bw influences comm overlap.
    return w_compute * compute_sum + w_mem * mem_min_gb + w_bw * (0.5 * bw_min_gbps + 0.5 * bw_avg_gbps)


def build_device_groups(
    cluster: ClusterSpec,
    *,
    group_size: int,
    prefer_same_vendor: bool = True,
    prefer_same_backend: bool = True,
) -> Tuple[List[DeviceGroup], List[DeviceSpec]]:
    """
    Build equal-size device groups (stage groups). Nodes may have different device counts.
    Groups may span nodes.

    Returns (groups, unused_devices).
    """
    assert group_size >= 1
    devices = cluster.all_devices()

    # Partition by (vendor, backend) if requested. This prevents invalid cross-vendor collectives.
    buckets: Dict[Tuple[str, str], List[DeviceSpec]] = {}
    for d in devices:
        key = (d.vendor if prefer_same_vendor else "*", d.backend if prefer_same_backend else "*")
        buckets.setdefault(key, []).append(d)

    groups: List[DeviceGroup] = []
    unused: List[DeviceSpec] = []

    gid = 0
    for (vendor_key, backend_key), bucket in buckets.items():
        # sort by compute_score desc, then memory desc
        remaining = sorted(bucket, key=lambda x: (x.compute_score, x.mem_gb), reverse=True)

        while len(remaining) >= group_size:
            # seed with the strongest device
            g_devs: List[DeviceSpec] = [remaining.pop(0)]
            while len(g_devs) < group_size and remaining:
                # greedy pick that maximizes group score after adding candidate
                best_idx = None
                best_score = None
                for idx, cand in enumerate(remaining):
                    trial = g_devs + [cand]
                    bw_min, bw_avg = _group_bandwidth_stats(cluster, trial)
                    compute_sum = sum(d.compute_score for d in trial)
                    mem_min = min(d.mem_gb for d in trial) if trial else 0.0
                    score = _group_score(
                        compute_sum=compute_sum,
                        mem_min_gb=mem_min,
                        bw_min_gbps=bw_min,
                        bw_avg_gbps=bw_avg,
                    )
                    if best_score is None or score > best_score:
                        best_score = score
                        best_idx = idx
                if best_idx is None:
                    break
                g_devs.append(remaining.pop(best_idx))

            bw_min, bw_avg = _group_bandwidth_stats(cluster, g_devs)
            compute_sum = sum(d.compute_score for d in g_devs)
            mem_min = min(d.mem_gb for d in g_devs) if g_devs else 0.0
            groups.append(
                DeviceGroup(
                    group_id=gid,
                    devices=g_devs,
                    compute_sum=compute_sum,
                    mem_min_gb=mem_min,
                    bw_min_gbps=bw_min if bw_min != float("inf") else cluster.links.intra_node_default_gbps,
                    bw_avg_gbps=bw_avg if bw_avg != float("inf") else cluster.links.intra_node_default_gbps,
                    vendor=vendor_key if vendor_key != "*" else "mixed",
                    backend=backend_key if backend_key != "*" else "mixed",
                )
            )
            gid += 1

        # leftover in this bucket unused
        unused.extend(remaining)

    return groups, unused
