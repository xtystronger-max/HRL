from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..errors import OptionalDependencyError
from ..schemas import ModelMeta


_LAYER_RE = re.compile(r"\.h\.(\d+)\.")


def introspect_hf_causal_lm(
    model_id: str,
    *,
    seq_len: Optional[int] = None,
    local_files_only: bool = False,
) -> ModelMeta:
    """
    Introspect a Hugging Face causal LM *architecture* and return parameter statistics
    without requiring the user to manually provide GPT-2 details.

    Strategy:
      1) Download config via PreTrainedConfig.from_pretrained(model_id)
      2) Instantiate a model from config (no pretrained weights) to get exact parameter tensors
      3) Count total and per-layer parameters by grouping parameter names (GPT-2 blocks use transformer.h.{i}.*)

    Requires: transformers (+ torch as a dependency of transformers model instantiation).
    """
    try:
        from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore
    except Exception as e:
        raise OptionalDependencyError(
            "transformers is required for model introspection. Install with: pip install -e '.[hf]'"
        ) from e

    try:
        cfg = AutoConfig.from_pretrained(model_id, local_files_only=local_files_only)
    except Exception as e:
        raise OptionalDependencyError(
            f"Failed to load config for {model_id}. Ensure internet access or set local_files_only=True with cached files."
        ) from e

    # Extract canonical fields where possible
    hidden_size = getattr(cfg, "n_embd", None) or getattr(cfg, "hidden_size", None)
    num_layers = getattr(cfg, "n_layer", None) or getattr(cfg, "num_hidden_layers", None)
    num_heads = getattr(cfg, "n_head", None) or getattr(cfg, "num_attention_heads", None)
    vocab_size = getattr(cfg, "vocab_size", None)
    if seq_len is None:
        seq_len = getattr(cfg, "n_positions", None) or getattr(cfg, "max_position_embeddings", None)

    if num_layers is None:
        # Fallback: instantiate and infer from module list if possible
        num_layers = 0

    # Instantiate architecture only (no weight download)
    try:
        model = AutoModelForCausalLM.from_config(cfg)
    except Exception as e:
        raise OptionalDependencyError(
            f"Failed to instantiate model from config for {model_id}. Ensure torch is installed and compatible."
        ) from e

    total_params = 0
    per_layer: Dict[int, int] = {}
    embedding_params = 0
    lm_head_params = 0

    for name, p in model.named_parameters():
        n = int(p.numel())
        total_params += n

        m = _LAYER_RE.search(name)
        if m:
            idx = int(m.group(1))
            per_layer[idx] = per_layer.get(idx, 0) + n
            continue

        # rough buckets
        if "wte" in name or "word_embeddings" in name or "embed_tokens" in name:
            embedding_params += n
        if "lm_head" in name or name.endswith("lm_head.weight") or "output_projection" in name:
            lm_head_params += n

    # normalize per-layer list length
    if num_layers == 0:
        # infer by max key + 1
        num_layers = (max(per_layer.keys()) + 1) if per_layer else 0

    per_layer_params: List[int] = [0 for _ in range(int(num_layers))]
    for i, v in per_layer.items():
        if 0 <= i < len(per_layer_params):
            per_layer_params[i] = int(v)

    return ModelMeta(
        model_id=model_id,
        total_params=int(total_params),
        num_layers=int(num_layers),
        per_layer_params=per_layer_params,
        hidden_size=int(hidden_size) if hidden_size is not None else None,
        num_heads=int(num_heads) if num_heads is not None else None,
        seq_len=int(seq_len) if seq_len is not None else None,
        vocab_size=int(vocab_size) if vocab_size is not None else None,
        embedding_params=int(embedding_params) if embedding_params else None,
        lm_head_params=int(lm_head_params) if lm_head_params else None,
    )
