from __future__ import annotations

from typing import Dict, List


def torchrun_command(
    *,
    nnodes: int,
    nproc_per_node: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
    entrypoint: str,
    entry_args: List[str],
) -> List[str]:
    cmd = [
        "torchrun",
        f"--nnodes={nnodes}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        f"--nproc_per_node={nproc_per_node}",
        entrypoint,
    ]
    cmd.extend(entry_args)
    return cmd


def visible_devices_env(per_node_visible_devices: Dict[str, List[int]], node_name: str) -> Dict[str, str]:
    devs = per_node_visible_devices.get(node_name, [])
    return {"CUDA_VISIBLE_DEVICES": ",".join(str(x) for x in devs)}
