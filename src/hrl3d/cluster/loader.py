from __future__ import annotations

from typing import Any, Dict, List

from ..schemas import ClusterSpec, DeviceSpec, LinksSpec, NodeSpec


def cluster_from_dict(d: Dict[str, Any]) -> ClusterSpec:
    nodes: List[NodeSpec] = []
    for n in d.get("nodes", []):
        name = str(n["name"])
        devs: List[DeviceSpec] = []
        for dev in n.get("devices", []):
            devs.append(
                DeviceSpec(
                    node=name,
                    local_id=int(dev["local_id"]),
                    vendor=str(dev.get("vendor", "unknown")),
                    backend=str(dev.get("backend", "")),
                    name=str(dev.get("name", "")),
                    mem_gb=float(dev.get("mem_gb", 0.0)),
                    cc_major=dev.get("cc_major"),
                    cc_minor=dev.get("cc_minor"),
                    sm_count=dev.get("sm_count"),
                    clock_rate_khz=dev.get("clock_rate_khz"),
                    uuid=dev.get("uuid"),
                    compute_score=float(dev.get("compute_score", 1.0)),
                )
            )
        nodes.append(NodeSpec(name=name, devices=devs))

    links_d = d.get("links", {})
    links = LinksSpec(
        inter_node_gbps=float(links_d.get("inter_node_gbps", 10.0)),
        intra_node_default_gbps=float(links_d.get("intra_node_default_gbps", 100.0)),
        intra_node_device_matrix_gbps=links_d.get("intra_node_device_matrix_gbps", {}) or {},
    )
    return ClusterSpec(nodes=nodes, links=links)
