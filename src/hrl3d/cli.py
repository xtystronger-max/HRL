from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from .cluster.loader import cluster_from_dict
from .cluster.probe import probe_local_cluster, save_cluster_json
from .io import dump_json, dump_yaml, load_json, load_yaml
from .model.introspect import introspect_hf_causal_lm
from .planner_hrl.trainer import train_hrl
from .planner_hrl.infer import plan_with_hrl
from .schemas import ModelMeta


def _load_model_meta(path_or_model_id: str) -> ModelMeta:
    p = Path(path_or_model_id)
    if p.exists():
        d = load_json(p)
        return ModelMeta(**d)
    # treat as HF model id
    return introspect_hf_causal_lm(path_or_model_id)


def cmd_probe_cluster(args: argparse.Namespace) -> None:
    cluster = probe_local_cluster(
        node_name=args.node_name,
        inter_node_gbps=args.inter_node_gbps,
        intra_node_default_gbps=args.intra_node_default_gbps,
        measure_intra_matrix=(not args.no_matrix),
        size_mb=args.size_mb,
        iters=args.iters,
        warmup=args.warmup,
    )
    save_cluster_json(cluster, args.out)
    print(f"Wrote cluster spec to {args.out}")


def cmd_introspect_model(args: argparse.Namespace) -> None:
    meta = introspect_hf_causal_lm(
        args.model_id,
        seq_len=args.seq_len,
        local_files_only=args.local_files_only,
    )
    dump_json(meta.to_dict(), args.out)
    print(f"Wrote model meta to {args.out}")


def cmd_train_hrl(args: argparse.Namespace) -> None:
    cluster = cluster_from_dict(load_json(args.cluster))
    model = _load_model_meta(args.model_meta)
    cfg = load_yaml(args.planner_cfg)
    train_hrl(cluster=cluster, model=model, planner_cfg=cfg, out_dir=args.out_dir)
    print(f"Wrote checkpoints to {args.out_dir}")


def cmd_plan_hrl(args: argparse.Namespace) -> None:
    cluster = cluster_from_dict(load_json(args.cluster))
    model = _load_model_meta(args.model_meta)
    cfg = load_yaml(args.planner_cfg)
    plan, info = plan_with_hrl(
        cluster=cluster,
        model=model,
        planner_cfg=cfg,
        checkpoint_dir=args.checkpoint_dir,
        num_samples=args.num_samples,
    )
    dump_yaml(plan.to_dict(), args.out)
    print(f"Wrote plan YAML to {args.out}")
    if args.out_info:
        dump_json(info, args.out_info)
        print(f"Wrote planning info to {args.out_info}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="hrl3d")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe-cluster", help="Probe local node devices and write cluster JSON")
    p.add_argument("--out", required=True)
    p.add_argument("--node-name", default=None)
    p.add_argument("--inter-node-gbps", type=float, default=10.0)
    p.add_argument("--intra-node-default-gbps", type=float, default=100.0)
    p.add_argument("--no-matrix", action="store_true", help="Do not measure CUDA intra-node p2p bandwidth matrix")
    p.add_argument("--size-mb", type=int, default=256)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.set_defaults(func=cmd_probe_cluster)

    p = sub.add_parser("introspect-model", help="Introspect a HF causal LM and write model meta JSON")
    p.add_argument("--model-id", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.set_defaults(func=cmd_introspect_model)

    p = sub.add_parser("train-hrl", help="Train HRL policies with simulated cost model")
    p.add_argument("--cluster", required=True, help="cluster json from probe-cluster or distributed probe")
    p.add_argument("--model-meta", required=True, help="model meta json or HF model id (e.g., gpt2)")
    p.add_argument("--planner-cfg", required=True, help="planner yaml config")
    p.add_argument("--out-dir", required=True, help="directory to write checkpoints")
    p.set_defaults(func=cmd_train_hrl)

    p = sub.add_parser("plan-hrl", help="Sample plans using trained HRL policies and export best plan to YAML")
    p.add_argument("--cluster", required=True)
    p.add_argument("--model-meta", required=True)
    p.add_argument("--planner-cfg", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--out", required=True, help="output plan yaml")
    p.add_argument("--out-info", default=None, help="optional output info json")
    p.set_defaults(func=cmd_plan_hrl)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
