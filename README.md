# HRL-3D Hybrid Parallel Planner (GPT-2 first)

This project provides:
1) **Model introspection** (GPT-2 or any HF causal LM) to extract:
   - total parameters
   - number of transformer blocks (layers)
   - per-layer parameter counts
2) **Cluster probing** (GPU/NPU + bandwidth, best-effort)
3) **Device group construction** under a **uniform nproc-per-node** launch constraint (torchrun-friendly)
4) **Pipeline stage partitioning** that accounts for group compute, bandwidth, and memory
5) **Micro-batch selection** as part of the HRL action space
6) **YAML plan export**

> This repo uses a *simulation cost model* to train policies. You can later replace the simulator with real runtime feedback.

## Quickstart (single node)

```bash
conda env create -f env/environment.yml
conda activate hrl3d

# 1) Probe local devices + intra-node bandwidth matrix (CUDA only)
hrl3d probe-cluster --out artifacts/cluster.json

# 2) Introspect GPT-2 without manually providing its details (requires internet or HF cache)
hrl3d introspect-model --model-id gpt2 --out artifacts/model_meta.json

# 3) Train HRL policies
hrl3d train-hrl --cluster artifacts/cluster.json --model-meta artifacts/model_meta.json --planner-cfg configs/planner_gpt2.yaml --out-dir artifacts/ckpt

# 4) Sample best plan and export YAML
hrl3d plan-hrl --cluster artifacts/cluster.json --model-meta artifacts/model_meta.json --planner-cfg configs/planner_gpt2.yaml --checkpoint-dir artifacts/ckpt --out artifacts/plan.yaml --out-info artifacts/plan_info.json
```

## Multi-node probe (optional)

Run `scripts/probe_cluster_distributed.py` via torchrun on all nodes and merge into one `cluster.json`.

```bash
# Example (2 nodes, 2 GPUs per node)
torchrun --nnodes=2 --node_rank=0 --master_addr=<MASTER_IP> --master_port=29500 --nproc_per_node=1 \
  scripts/probe_cluster_distributed.py --out artifacts/cluster.json

torchrun --nnodes=2 --node_rank=1 --master_addr=<MASTER_IP> --master_port=29500 --nproc_per_node=1 \
  scripts/probe_cluster_distributed.py --out artifacts/cluster.json
```

The script uses a file-system gather under `artifacts/.probe_tmp/`.

## Output

The exported `plan.yaml` contains:
- `pp` (pipeline stages)
- `microbatch_size`
- `device_groups` (stage groups, equal size)
- `stages` (layer spans and tp/dp)
- `meta.launch.per_node_visible_devices` (torchrun-friendly CUDA_VISIBLE_DEVICES suggestions)
