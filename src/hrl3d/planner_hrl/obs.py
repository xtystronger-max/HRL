from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..schemas import ClusterSpec, ModelMeta
from ..cluster.grouping import DeviceGroup


@dataclass
class Observation:
    vec: np.ndarray


def build_observation(
    *,
    cluster: ClusterSpec,
    model: ModelMeta,
    groups: List[DeviceGroup],
    use_gnn: bool = False,
    gnn_dim: int = 64,
) -> Observation:
    """A compact numerical observation vector for policy networks."""
    world = float(cluster.world_size())
    num_nodes = float(len(cluster.nodes))
    # model stats
    total_params = float(model.total_params)
    num_layers = float(model.num_layers)
    per_layer = np.array(model.per_layer_params, dtype=np.float64) if model.per_layer_params else np.zeros(1, dtype=np.float64)
    per_layer_mean = float(per_layer.mean()) if per_layer.size else 0.0
    per_layer_max = float(per_layer.max()) if per_layer.size else 0.0
    per_layer_min = float(per_layer.min()) if per_layer.size else 0.0

    # group stats
    if groups:
        comp = np.array([g.compute_sum for g in groups], dtype=np.float64)
        mem = np.array([g.mem_min_gb for g in groups], dtype=np.float64)
        bw = np.array([g.bw_min_gbps for g in groups], dtype=np.float64)
        g_comp_mean = float(comp.mean())
        g_comp_max = float(comp.max())
        g_mem_min = float(mem.min())
        g_mem_mean = float(mem.mean())
        g_bw_min = float(bw.min())
        g_bw_mean = float(bw.mean())
    else:
        g_comp_mean = g_comp_max = g_mem_min = g_mem_mean = g_bw_min = g_bw_mean = 0.0

    base_vec = np.array(
        [
            world,
            num_nodes,
            total_params,
            num_layers,
            per_layer_mean,
            per_layer_max,
            per_layer_min,
            float(model.hidden_size or 0),
            float(model.seq_len or 0),
            float(model.num_heads or 0),
            g_comp_mean,
            g_comp_max,
            g_mem_min,
            g_mem_mean,
            g_bw_min,
            g_bw_mean,
            float(cluster.links.inter_node_gbps),
            float(cluster.links.intra_node_default_gbps),
        ],
        dtype=np.float32,
    )

    if use_gnn:
        try:
            from ..gnn.encoder import ClusterGNNEncoder
            enc = ClusterGNNEncoder(out_dim=int(gnn_dim))
            emb = enc.encode(cluster)
            emb = emb.astype(np.float32)
        except Exception:
            emb = np.zeros((int(gnn_dim),), dtype=np.float32)
        vec = np.concatenate([base_vec, emb], axis=0)
    else:
        vec = base_vec

    return Observation(vec=vec)
