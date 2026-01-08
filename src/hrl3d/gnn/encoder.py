from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..errors import OptionalDependencyError
from ..schemas import ClusterSpec


def _require_torch_geo():
    try:
        import torch  # type: ignore
        from torch_geometric.data import Data  # type: ignore
        from torch_geometric.nn import GINEConv, global_mean_pool  # type: ignore
        return torch, Data, GINEConv, global_mean_pool
    except Exception as e:
        raise OptionalDependencyError(
            "GNN encoding requires torch and torch-geometric. Install with: pip install -e '.[rl,gnn]'"
        ) from e


@dataclass
class ClusterGNNEncoder:
    hidden_dim: int = 64
    out_dim: int = 64
    num_layers: int = 3

    def __post_init__(self) -> None:
        torch, Data, GINEConv, global_mean_pool = _require_torch_geo()
        self.torch = torch
        self.Data = Data
        self.global_mean_pool = global_mean_pool

        mlp = lambda: torch.nn.Sequential(
            torch.nn.Linear(1, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.in_proj = torch.nn.Linear(4, self.hidden_dim)
        self.convs = torch.nn.ModuleList([GINEConv(nn=mlp()) for _ in range(self.num_layers)])
        self.proj = torch.nn.Linear(self.hidden_dim, self.out_dim)

    def encode(self, cluster: ClusterSpec) -> np.ndarray:
        torch = self.torch
        Data = self.Data
        # map each device to a global index
        devices = cluster.all_devices()
        idx = {(d.node, d.local_id): i for i, d in enumerate(devices)}
        n = len(devices)
        if n == 0:
            return np.zeros((self.out_dim,), dtype=np.float32)

        # node features (standardized lightly)
        vendor_map: Dict[str, float] = {}
        backend_map: Dict[str, float] = {}
        def vid(x: str) -> float:
            if x not in vendor_map:
                vendor_map[x] = float(len(vendor_map) + 1)
            return vendor_map[x]
        def bid(x: str) -> float:
            if x not in backend_map:
                backend_map[x] = float(len(backend_map) + 1)
            return backend_map[x]

        x_list = []
        for d in devices:
            x_list.append([float(d.compute_score), float(d.mem_gb), vid(d.vendor), bid(d.backend)])
        x = torch.tensor(x_list, dtype=torch.float32)

        # edges: complete graph with edge_attr=bw, but that's O(n^2); use thresholding by bandwidth.
        edge_index = []
        edge_attr = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                di = devices[i]
                dj = devices[j]
                bw = float(cluster.device_bw_gbps(di, dj))
                # Keep only meaningful edges
                if bw <= 0:
                    continue
                edge_index.append([i, j])
                edge_attr.append([bw])

        if not edge_index:
            # isolated: return pooled projection of node features
            h = torch.zeros((n, self.hidden_dim), dtype=torch.float32)
            emb = self.proj(h).mean(dim=0)
            return emb.detach().cpu().numpy().astype(np.float32)

        edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t)
        h = self.in_proj(data.x)

        for conv in self.convs:
            h = conv(h, data.edge_index, data.edge_attr)
            h = torch.relu(h)
        h = self.proj(h)

        batch = torch.zeros((n,), dtype=torch.long)
        pooled = self.global_mean_pool(h, batch)[0]
        return pooled.detach().cpu().numpy().astype(np.float32)
