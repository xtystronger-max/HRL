from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DeviceSpec:
    node: str
    local_id: int
    vendor: str = "unknown"   # e.g., nvidia, huawei
    backend: str = ""         # e.g., cuda, npu
    name: str = ""
    mem_gb: float = 0.0

    # Optional compute descriptors
    cc_major: Optional[int] = None
    cc_minor: Optional[int] = None
    sm_count: Optional[int] = None
    clock_rate_khz: Optional[int] = None
    uuid: Optional[str] = None

    # A scalar proxy for compute throughput (higher => faster)
    compute_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NodeSpec:
    name: str
    devices: List[DeviceSpec]


@dataclass
class LinksSpec:
    inter_node_gbps: float = 10.0
    intra_node_default_gbps: float = 100.0

    # Optional: per-node device bandwidth matrices (Gbps), indexed by local_id
    intra_node_device_matrix_gbps: Dict[str, List[List[float]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClusterSpec:
    nodes: List[NodeSpec]
    links: LinksSpec

    def all_devices(self) -> List[DeviceSpec]:
        out: List[DeviceSpec] = []
        for n in self.nodes:
            out.extend(n.devices)
        return out

    def world_size(self) -> int:
        return len(self.all_devices())

    def get_node(self, name: str) -> NodeSpec:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(name)

    def device_bw_gbps(self, a: DeviceSpec, b: DeviceSpec) -> float:
        """Return estimated point-to-point bandwidth between two devices."""
        if a.node == b.node:
            mat = self.links.intra_node_device_matrix_gbps.get(a.node)
            if mat and a.local_id < len(mat) and b.local_id < len(mat[a.local_id]):
                v = float(mat[a.local_id][b.local_id])
                if v > 0:
                    return v
            return float(self.links.intra_node_default_gbps)
        return float(self.links.inter_node_gbps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {"name": n.name, "devices": [d.to_dict() for d in n.devices]}
                for n in self.nodes
            ],
            "links": self.links.to_dict(),
        }


@dataclass
class ModelMeta:
    model_id: str
    total_params: int
    num_layers: int
    per_layer_params: List[int]
    hidden_size: Optional[int] = None
    num_heads: Optional[int] = None
    seq_len: Optional[int] = None
    vocab_size: Optional[int] = None

    # optional: embedding/head params
    embedding_params: Optional[int] = None
    lm_head_params: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineStage:
    stage_id: int
    layer_start: int
    layer_end: int          # inclusive
    device_group_id: int
    tp: int
    dp: int

    def num_layers(self) -> int:
        return self.layer_end - self.layer_start + 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Plan:
    pp: int
    stages: List[PipelineStage]
    microbatch_size: int
    schedule: str                 # '1f1b' or 'gpipe'
    device_groups: List[List[Tuple[str, int]]]  # list of (node, local_id) per group

    meta: Dict[str, Any] = field(default_factory=dict)

    
def validate_basic(self, *, total_devices: int, allow_unused: bool = True) -> None:
    if self.pp <= 0:
        raise ValueError("pp must be positive")
    if len(self.stages) != self.pp:
        raise ValueError(f"stages length {len(self.stages)} != pp {self.pp}")

    used = sum(len(g) for g in self.device_groups)
    if allow_unused:
        if used > total_devices:
            raise ValueError(f"device_groups total devices {used} > cluster devices {total_devices}")
    else:
        if used != total_devices:
            raise ValueError(f"device_groups total devices {used} != cluster devices {total_devices}")

    # All device groups must be equal-size
    group_size = len(self.device_groups[0]) if self.device_groups else 0
    for g in self.device_groups:
        if len(g) != group_size:
            raise ValueError("All device groups must have equal size")

    for s in self.stages:
        if not (0 <= s.device_group_id < len(self.device_groups)):
            raise ValueError("Invalid stage.device_group_id")
        if s.tp <= 0 or s.dp <= 0:
            raise ValueError("tp/dp must be positive")
        if s.tp * s.dp != group_size:
            raise ValueError("tp*dp must equal device group size")

    # Check stages cover layers contiguously
    spans = [(s.layer_start, s.layer_end) for s in self.stages]
    spans.sort()
    if spans and spans[0][0] != 0:
        raise ValueError("First stage must start at layer 0")
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        if a1 + 1 != b0:
            raise ValueError("Stages must be contiguous without gaps/overlap")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pp": self.pp,
            "microbatch_size": self.microbatch_size,
            "schedule": self.schedule,
            "device_groups": self.device_groups,
            "stages": [s.to_dict() for s in self.stages],
            "meta": self.meta,
        }
