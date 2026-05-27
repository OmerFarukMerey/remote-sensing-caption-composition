"""Lightweight LoRA adapters for fine-tuning frozen openai/CLIP encoders.

Phase 3 unfreezes CLIP via low-rank adapters instead of full fine-tuning.
We wrap the reachable ``nn.Linear`` layers of each residual block --- the MLP
projections (``mlp.c_fc``, ``mlp.c_proj``) and the attention output projection
(``attn.out_proj``). The packed QKV projection inside ``nn.MultiheadAttention``
(``in_proj_weight``) is not a standalone ``nn.Linear`` and is intentionally left
untouched; adapting MLP + out_proj across the top blocks is enough to shift the
representation while keeping the change contained and reproducible.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Linear submodule names targeted inside each CLIP residual block.
# Only the MLP projections are adapted: openai/CLIP's nn.MultiheadAttention calls
# F.multi_head_attention_forward with out_proj.weight directly (it never invokes
# out_proj.forward), so wrapping the attention projections with a LoRA module
# would be silently bypassed. The block MLP Linears ARE called via their forward,
# so LoRA there is effective.
DEFAULT_TARGETS = ("mlp.c_fc", "mlp.c_proj")


class LoRALinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` with a trainable low-rank update.

    Computes ``base(x) + (dropout(x) @ A^T) @ B^T * scaling`` where only ``A``
    and ``B`` carry gradients. ``A`` is Kaiming-initialised and ``B`` is zero, so
    the adapter is an identity at initialisation.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_features = base.in_features
        out_features = base.out_features
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)
        self.scaling = alpha / r

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        # Match the adapter matmul to the input dtype (CLIP may run in fp16 on
        # some backends while the LoRA params are fp32), then cast back.
        a = self.lora_A.t().to(x.dtype)
        b = self.lora_B.t().to(x.dtype)
        update = (self.lora_dropout(x) @ a) @ b
        return out + update.to(out.dtype) * self.scaling


def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    *parents, attr = dotted_name.split(".")
    obj = root
    for p in parents:
        obj = getattr(obj, p)
    setattr(obj, attr, new_module)


def _resblocks(encoder_root: nn.Module):
    """Return the residual blocks of a CLIP transformer encoder, or []."""
    transformer = getattr(encoder_root, "transformer", None)
    if transformer is None:
        return []
    return list(getattr(transformer, "resblocks", []))


def inject_lora(
    clip_model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    last_k_blocks: int = 6,
    vision: bool = True,
    text: bool = True,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
) -> int:
    """Replace target Linear layers in the last ``k`` blocks with ``LoRALinear``.

    Returns the number of layers adapted. Base CLIP weights are frozen first;
    only the injected ``lora_A``/``lora_B`` parameters remain trainable.
    """
    for p in clip_model.parameters():
        p.requires_grad_(False)

    encoders = []
    if vision:
        encoders.append(clip_model.visual)
    if text:
        encoders.append(clip_model)  # text transformer lives on the root module

    n_injected = 0
    for enc in encoders:
        blocks = _resblocks(enc)
        if not blocks:
            continue
        chosen = blocks[-last_k_blocks:] if last_k_blocks > 0 else blocks
        for block in chosen:
            for name in targets:
                try:
                    sub = block.get_submodule(name)
                except AttributeError:
                    continue
                if isinstance(sub, nn.Linear):
                    _set_submodule(
                        block, name, LoRALinear(sub, r=r, alpha=alpha, dropout=dropout)
                    )
                    n_injected += 1
    return n_injected


def lora_parameters(module: nn.Module):
    """Yield only the trainable LoRA parameters of a module tree."""
    for name, p in module.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            yield p
