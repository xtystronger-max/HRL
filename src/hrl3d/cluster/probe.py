from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..errors import OptionalDependencyError
from ..schemas import ClusterSpec, DeviceSpec, LinksSpec, NodeSpec


def _try_import_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None


def _probe_cuda_devices_with_torch(node_name: str) -> List[DeviceSpec]:
    torch = _try_import_torch()
    if torch is None or not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return []
    devices: List[DeviceSpec] = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        cc = torch.cuda.get_device_capability(i)
        # A simple compute proxy: SMs * clock (kHz)
        compute_score = float(getattr(p, "multi_processor_count", 1) * getattr(p, "clock_rate", 1))
        devices.append(
            DeviceSpec(
                node=node_name,
                local_id=i,
                vendor="nvidia",
                backend="cuda",
                name=str(getattr(p, "name", "")),
                mem_gb=float(getattr(p, "total_memory", 0) / (1024**3)),
                cc_major=int(cc[0]),
                cc_minor=int(cc[1]),
                sm_count=int(getattr(p, "multi_processor_count", 0)),
                clock_rate_khz=int(getattr(p, "clock_rate", 0)),
                uuid=None,
                compute_score=compute_score,
            )
        )
    return devices


def _probe_nvidia_smi(node_name: str) -> List[DeviceSpec]:
    """Fallback probing using nvidia-smi if torch isn't available."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,uuid,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []

    devices: List[DeviceSpec] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        idx, name, mem_mb, uuid, cc = parts[0], parts[1], parts[2], parts[3], parts[4]
        cc_major, cc_minor = None, None
        m = cc.split(".")
        if len(m) == 2 and m[0].isdigit() and m[1].isdigit():
            cc_major, cc_minor = int(m[0]), int(m[1])
        mem_gb = float(mem_mb) / 1024.0
        devices.append(
            DeviceSpec(
                node=node_name,
                local_id=int(idx),
                vendor="nvidia",
                backend="cuda",
                name=name,
                mem_gb=mem_gb,
                cc_major=cc_major,
                cc_minor=cc_minor,
                uuid=uuid,
                compute_score=1.0,
            )
        )
    return devices


def _probe_ascend_torch_npu(node_name: str) -> List[DeviceSpec]:
    """Best-effort Ascend probing via torch_npu."""
    try:
        import torch_npu  # type: ignore
    except Exception:
        return []

    # torch_npu API varies; keep defensive.
    try:
        n = int(torch_npu.npu.device_count())
    except Exception:
        return []

    devices: List[DeviceSpec] = []
    for i in range(n):
        name = ""
        try:
            name = str(torch_npu.npu.get_device_name(i))
        except Exception:
            name = "ascend"
        mem_gb = 0.0
        # Try to infer memory if possible
        try:
            # Some builds expose mem_get_info(device) -> (free, total) in bytes
            free_b, total_b = torch_npu.npu.mem_get_info(i)  # type: ignore
            mem_gb = float(total_b) / (1024**3)
        except Exception:
            pass

        devices.append(
            DeviceSpec(
                node=node_name,
                local_id=i,
                vendor="huawei",
                backend="npu",
                name=name,
                mem_gb=mem_gb,
                compute_score=1.0,
            )
        )
    return devices


def probe_local_cluster(
    *,
    node_name: Optional[str] = None,
    inter_node_gbps: float = 10.0,
    intra_node_default_gbps: float = 100.0,
    measure_intra_matrix: bool = True,
    size_mb: int = 256,
    iters: int = 30,
    warmup: int = 5,
) -> ClusterSpec:
    """Probe the current machine and return a 1-node ClusterSpec."""
    node_name = node_name or os.uname().nodename

    devices = _probe_cuda_devices_with_torch(node_name)
    if not devices:
        devices = _probe_ascend_torch_npu(node_name)
    if not devices:
        devices = _probe_nvidia_smi(node_name)

    links = LinksSpec(
        inter_node_gbps=float(inter_node_gbps),
        intra_node_default_gbps=float(intra_node_default_gbps),
        intra_node_device_matrix_gbps={},
    )
    cluster = ClusterSpec(nodes=[NodeSpec(name=node_name, devices=devices)], links=links)

    if measure_intra_matrix and any(d.backend == "cuda" for d in devices):
        try:
            mat = measure_intra_node_bandwidth_cuda(devices, size_mb=size_mb, iters=iters, warmup=warmup)
            cluster.links.intra_node_device_matrix_gbps[node_name] = mat
        except OptionalDependencyError:
            pass
        except Exception:
            # Do not fail probing just because the benchmark fails.
            pass
    return cluster


def measure_intra_node_bandwidth_cuda(
    devices: List[DeviceSpec],
    *,
    size_mb: int = 256,
    iters: int = 30,
    warmup: int = 5,
) -> List[List[float]]:
    torch = _try_import_torch()
    if torch is None or not torch.cuda.is_available():
        raise OptionalDependencyError("torch+cuda is required to measure intra-node bandwidth")

    n = len(devices)
    # allocate per device
    num_bytes = int(size_mb * 1024 * 1024)
    numel = num_bytes // 4  # float32
    tensors: List[Tuple[int, "torch.Tensor"]] = []
    for d in devices:
        if d.backend != "cuda":
            raise OptionalDependencyError("Bandwidth measurement only implemented for CUDA devices")
        torch.cuda.set_device(d.local_id)
        t = torch.empty(numel, device=f"cuda:{d.local_id}", dtype=torch.float32)
        tensors.append((d.local_id, t))

    def copy_time(src_id: int, dst_id: int) -> float:
        torch.cuda.synchronize(src_id)
        torch.cuda.synchronize(dst_id)
        # warmup
        for _ in range(warmup):
            _ = tensors[dst_id][1].copy_(tensors[src_id][1])
        torch.cuda.synchronize(src_id)
        torch.cuda.synchronize(dst_id)

        start = time.time()
        for _ in range(iters):
            _ = tensors[dst_id][1].copy_(tensors[src_id][1])
        torch.cuda.synchronize(src_id)
        torch.cuda.synchronize(dst_id)
        end = time.time()
        return (end - start) / max(1, iters)

    mat: List[List[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i][j] = 0.0
                continue
            dt = copy_time(i, j)
            gbps = (num_bytes / dt) / (1024**3) * 8.0  # bytes/s -> Gbps
            mat[i][j] = float(gbps)
    return mat


def save_cluster_json(cluster: ClusterSpec, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cluster.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
