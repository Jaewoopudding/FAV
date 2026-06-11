"""
LoRA (Low-Rank Adaptation) for Flax / JAX models.

The approach works *without* modifying the model source code:

1. ``create_lora_params`` walks the frozen base-param tree and creates
   a *parallel* LoRA-param tree containing only ``lora_a`` / ``lora_b``
   matrices for the targeted Dense kernels.

2. ``merge_lora`` produces an effective param tree by adding the low-rank
   delta ``(A @ B) * (alpha / rank)`` to each targeted kernel.

3. The optimizer only sees the (small) LoRA-param tree; base params stay
   frozen and are never passed to ``optax``.

Usage sketch::

    lora_params = create_lora_params(base_params, rank=16, rng=rng)
    effective = merge_lora(base_params, lora_params, scaling=alpha/rank)
    output = model.apply({'params': effective}, ...)

    # optimiser step
    loss, grads = jax.value_and_grad(loss_fn)(lora_params)
    state = state.apply_gradients(grads=grads)   # state.params == lora_params
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from flax import serialization


# ═══════════════════════════════════════════════════════════════════════════
# Target-layer selectors
# ═══════════════════════════════════════════════════════════════════════════

def _path_str(path: tuple[str, ...]) -> str:
    return "/".join(path)


def attn_qkv_out_target(path: tuple[str, ...]) -> bool:
    """Target attention QKV and output projections (the two TorchLinear
    layers inside each ``Attention_0`` block)."""
    s = _path_str(path)
    return ("Attention_0/TorchLinear_0/Dense_0" in s or
            "Attention_0/TorchLinear_1/Dense_0" in s)


def attn_and_mlp_target(path: tuple[str, ...]) -> bool:
    """Target attention QKV/out *and* all MLP Dense layers."""
    s = _path_str(path)
    if "Attention_0/TorchLinear_0/Dense_0" in s:
        return True
    if "Attention_0/TorchLinear_1/Dense_0" in s:
        return True
    # SwiGLUFFN has 3 TorchLinear; StandardMLP has 2
    if "SwiGLUFFN_0/" in s and "Dense_0" in s:
        return True
    if "StandardMLP_0/" in s and "Dense_0" in s:
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Create LoRA parameters
# ═══════════════════════════════════════════════════════════════════════════

def create_lora_params(
    base_params: dict,
    rank: int = 16,
    rng: jax.Array = None,
    target_fn: Callable[[tuple[str, ...]], bool] = attn_qkv_out_target,
) -> dict:
    """Create a LoRA parameter tree that mirrors *base_params*.

    For every ``kernel`` leaf whose parent-dict path passes ``target_fn``,
    the output tree contains ``lora_a`` (Kaiming-style init) and ``lora_b``
    (zeros), so the initial LoRA delta is zero.

    Returns:
        A nested dict (pytree) suitable for ``optax`` and ``jax.tree_util``.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    def _walk(base: dict, path: tuple[str, ...], rng: jax.Array) -> dict:
        out: dict = {}
        for key, value in base.items():
            new_path = path + (key,)
            rng, sub_rng = jax.random.split(rng)
            if isinstance(value, dict):
                sub = _walk(value, new_path, sub_rng)
                if sub:                          # keep only non-empty branches
                    out[key] = sub
            elif key == "kernel" and target_fn(path):
                in_dim, out_dim = value.shape
                # Kaiming-uniform–style scale for A; B=0 so delta starts at 0
                out["lora_a"] = jax.random.normal(sub_rng, (in_dim, rank)) * (1.0 / rank)
                out["lora_b"] = jnp.zeros((rank, out_dim))
        return out

    return _walk(base_params, (), rng)


# ═══════════════════════════════════════════════════════════════════════════
# Merge LoRA into base parameters
# ═══════════════════════════════════════════════════════════════════════════

def merge_lora(
    base_params: dict,
    lora_params: dict,
    scaling: float,
) -> dict:
    """Return effective params: ``W_eff = W_base + (A @ B) * scaling``.

    Non-targeted leaves are passed through from *base_params* unchanged.
    """

    def _merge(base: Any, lora: Any) -> Any:
        if not isinstance(base, dict):
            return base

        result: dict = {}
        for key in base:
            if key not in lora:
                result[key] = base[key]
                continue

            lora_sub = lora[key]
            base_sub = base[key]

            # If lora_sub contains lora_a/lora_b → apply the delta to the
            # *parent* Dense dict which holds 'kernel' (and maybe 'bias').
            if isinstance(lora_sub, dict) and "lora_a" in lora_sub:
                if isinstance(base_sub, dict) and "kernel" in base_sub:
                    merged_sub = dict(base_sub)          # shallow copy
                    delta = lora_sub["lora_a"] @ lora_sub["lora_b"] * scaling
                    merged_sub["kernel"] = base_sub["kernel"] + delta
                    result[key] = merged_sub
                else:
                    # Shouldn't happen; fall through.
                    result[key] = base_sub
            else:
                # Recurse deeper.
                result[key] = _merge(base_sub, lora_sub)

        return result

    return _merge(base_params, lora_params)


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def count_params(tree: Any) -> int:
    """Total number of scalar parameters in a pytree."""
    leaves = jax.tree.leaves(tree)
    return sum(x.size for x in leaves)


def count_base_params(base_params: dict) -> int:
    return count_params(base_params)


def count_lora_params(lora_params: dict) -> int:
    return count_params(lora_params)


def lora_summary(base_params: dict, lora_params: dict) -> str:
    n_base = count_base_params(base_params)
    n_lora = count_lora_params(lora_params)
    pct = 100.0 * n_lora / n_base if n_base > 0 else 0.0
    return (f"LoRA: trainable={n_lora:,} / total={n_base:,} "
            f"({pct:.2f}%)")


# ── Checkpoint I/O ───────────────────────────────────────────────────────

def save_lora_checkpoint(
    lora_params: dict,
    path: str | Path,
    *,
    step: int = 0,
    lora_config: dict | None = None,
):
    """Save LoRA params + metadata to a directory."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    cpu_params = jax.device_get(lora_params)
    (out / "lora_params.msgpack").write_bytes(
        serialization.msgpack_serialize(cpu_params),
    )
    meta = {
        "step": step,
        "lora_config": dict(lora_config or {}),
    }
    (out / "lora_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )


def load_lora_checkpoint(path: str | Path):
    """Load LoRA params + metadata from a directory.

    Returns ``(lora_params, meta_dict)``.
    """
    d = Path(path)
    lora_params = serialization.msgpack_restore(
        (d / "lora_params.msgpack").read_bytes(),
    )
    meta = json.loads((d / "lora_meta.json").read_text(encoding="utf-8"))
    return lora_params, meta
