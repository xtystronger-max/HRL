from __future__ import annotations

import argparse
import os
from typing import Any, Dict

from hrl3d.cluster.probe import probe_local_cluster
from hrl3d.io import dump_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output cluster JSON path")
    parser.add_argument("--inter-node-gbps", type=float, default=10.0)
    parser.add_argument("--intra-node-default-gbps", type=float, default=100.0)
    parser.add_argument("--no-matrix", action="store_true")
    args = parser.parse_args()

    # torchrun will set RANK/WORLD_SIZE
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))

    local = probe_local_cluster(
        node_name=os.uname().nodename,
        inter_node_gbps=args.inter_node_gbps,
        intra_node_default_gbps=args.intra_node_default_gbps,
        measure_intra_matrix=(not args.no_matrix),
    ).to_dict()

    # Simple gather via file system: each rank writes its json. Rank0 merges.
    # This is robust without assuming torch.distributed availability.
    tmp_dir = os.path.join(os.path.dirname(args.out), ".probe_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"rank{rank}.json")
    dump_json(local, tmp_path)

    if rank == 0:
        # wait for others
        import time
        deadline = time.time() + 300
        while True:
            files = [f for f in os.listdir(tmp_dir) if f.endswith(".json")]
            if len(files) >= world:
                break
            if time.time() > deadline:
                raise RuntimeError("Timed out waiting for other ranks")
            time.sleep(1)

        merged: Dict[str, Any] = {"nodes": [], "links": local.get("links", {})}
        # Keep inter/intra defaults from rank0; matrices are per node.
        matrices = merged["links"].get("intra_node_device_matrix_gbps", {}) or {}
        merged["links"]["intra_node_device_matrix_gbps"] = matrices

        for r in range(world):
            d = __import__("json").loads(open(os.path.join(tmp_dir, f"rank{r}.json"), "r", encoding="utf-8").read())
            merged["nodes"].extend(d.get("nodes", []))
            mats = d.get("links", {}).get("intra_node_device_matrix_gbps", {}) or {}
            matrices.update(mats)

        dump_json(merged, args.out)
        # optionally cleanup
        # shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
