"""LoRA injection utilities for StyleGAN-XL generators.

Targets the FullyConnectedLayer family only (mapping FC + synthesis style
affine) — matching the paper's StyleGAN-XL fine-tuning (originally selected
via ``--lora-target fc``). The synthesis conv kernels are left frozen.

Because @persistence.persistent_class reconstructs classes from pickled source
code, isinstance() checks against the original class may fail after unpickling.
We therefore match by class name.

LoRA formulation:
    FC:    effective_weight = W * weight_gain + scale * lora_B @ lora_A
           out_pre_act      = x @ effective_weight.T + bias

where lora_A ∈ ℝ^{rank × in}, lora_B ∈ ℝ^{out × rank}, scale = alpha / rank.
lora_B is initialised to zero so the delta is zero at the start of training.
"""

import types
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_fc_layer(module):
    """Match by class name to survive @persistence.persistent_class unpickling."""
    return type(module).__name__ == 'FullyConnectedLayer'


def _patch_fc_layer(module, rank: int, scale: float, dropout_p: float):
    """
    Inject LoRA into a single FullyConnectedLayer.

    Adds:
        module.lora_A  – nn.Parameter, shape (rank, in_features)
        module.lora_B  – nn.Parameter, shape (out_features, rank)
        module._lora_scale      – float
        module._lora_dropout_p  – float

    Replaces module.forward with a LoRA-aware version.
    """
    in_f = module.in_features
    out_f = module.out_features

    # lora_B=0 so delta is zero at initialisation (standard LoRA init)
    module.register_parameter('lora_A', nn.Parameter(torch.randn(rank, in_f) * 0.01))
    module.register_parameter('lora_B', nn.Parameter(torch.zeros(out_f, rank)))
    module._lora_scale = scale
    module._lora_dropout_p = dropout_p

    # Lazily import bias_act inside the closure to avoid circular imports at
    # module load time.
    def _lora_forward(self, x):
        from torch_utils.ops import bias_act as _bias_act  # noqa: PLC0415

        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        # LoRA delta computed on the (optionally dropped) input
        x_drop = (
            F.dropout(x, p=self._lora_dropout_p, training=self.training)
            if self._lora_dropout_p > 0.0
            else x
        )
        lora_delta = (
            x_drop
            .matmul(self.lora_A.to(x.dtype).t())   # (N, rank)
            .matmul(self.lora_B.to(x.dtype).t())   # (N, out_f)
            * self._lora_scale
        )

        # Mirror the original FullyConnectedLayer forward exactly, but add the
        # LoRA delta to the pre-activation linear output.
        if self.activation == 'linear' and b is not None:
            return torch.addmm(b.unsqueeze(0), x, w.t()) + lora_delta
        else:
            out = x.matmul(w.t()) + lora_delta
            return _bias_act.bias_act(out, b, act=self.activation)

    # Bind as a proper method so `self` is set correctly.
    module.forward = types.MethodType(_lora_forward, module)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inject_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> tuple:
    """Inject LoRA into a StyleGAN-XL generator (FullyConnectedLayer only).

    StyleGAN-XL fine-tuning targets the FullyConnectedLayer family (mapping FC
    + synthesis style affine) — matching the paper. The synthesis conv kernels
    are intentionally left frozen.

    Args:
        model:   The generator (G_ema) loaded from a StyleGAN-XL checkpoint.
        rank:    LoRA rank (inner dimension of the low-rank decomposition).
        alpha:   LoRA scaling factor; the effective scale is alpha / rank.
        dropout: Dropout probability applied to the input before the LoRA path.

    Returns:
        model        – same model object, base weights frozen, LoRA params trainable.
        n_trainable  – number of trainable parameters after injection.
        n_total      – total number of parameters (frozen + trainable).
    """
    # 1. Freeze everything first.
    for p in model.parameters():
        p.requires_grad_(False)

    scale = alpha / rank

    # 2. Walk every sub-module and patch the FullyConnectedLayer instances.
    n_fc = 0
    for module in model.modules():
        if _is_fc_layer(module):
            _patch_fc_layer(module, rank=rank, scale=scale, dropout_p=dropout)
            n_fc += 1

    if n_fc == 0:
        raise RuntimeError(
            "inject_lora: no FullyConnectedLayer found. "
            "Check that the checkpoint uses a supported StyleGAN architecture."
        )

    # 3. Count parameters.
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    return model, n_trainable, n_total


def get_lora_state_dict(model: nn.Module) -> dict:
    """Return a state-dict containing only LoRA parameters (lora_A, lora_B).

    Uses named_parameters() so only trainable Parameter tensors are collected
    (not buffers like magnitude_ema or w_avg).  Values are detached copies,
    identical to what torch.save / torch.load expect.

    Compatible with accelerator.unwrap_model(model) — the unwrapped model has
    the original key names (no 'module.' DDP prefix).

    Returns:
        dict mapping e.g. 'mapping.fc0.lora_A' → tensor, suitable for
        torch.save and load_lora_state_dict.

    Raises:
        RuntimeError if inject_lora has not been called yet.
    """
    sd = {
        k: v.detach().cpu()
        for k, v in model.named_parameters()
        if 'lora_A' in k or 'lora_B' in k
    }
    if not sd:
        raise RuntimeError(
            "get_lora_state_dict: no lora_A/lora_B parameters found. "
            "Did you call inject_lora() before saving?"
        )
    return sd


def load_lora_state_dict(model: nn.Module, lora_state: dict, strict: bool = True):
    """Load LoRA parameters back into a model that has already had inject_lora applied.

    Args:
        model:       Model with LoRA already injected.
        lora_state:  Dict returned by get_lora_state_dict().
        strict:      If True, raise if any lora key is missing from the model.
    """
    model_keys = {k for k in model.state_dict() if 'lora_A' in k or 'lora_B' in k}
    if strict:
        missing = model_keys - set(lora_state.keys())
        unexpected = set(lora_state.keys()) - model_keys
        if missing or unexpected:
            raise RuntimeError(
                f"load_lora_state_dict: missing keys: {missing}, unexpected keys: {unexpected}"
            )
    # Load only matching keys
    current_sd = model.state_dict()
    current_sd.update({k: v for k, v in lora_state.items() if k in current_sd})
    model.load_state_dict(current_sd, strict=False)
