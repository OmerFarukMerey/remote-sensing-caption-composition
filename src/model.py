"""CLIP-based regressors for the Information Richness ablation.

Both encoders are loaded once externally (via clip.load) and shared across
conditions to keep memory use low. Only fusion + head parameters are trained.

Each model exposes two forward paths:

* `forward(images, tokens=None)` — runs CLIP forward inside the module (online).
* `forward_emb(patches, text_emb=None)` — skips CLIP and operates directly on
  precomputed (B, 49, 512) / (B, 77, 512) tensors. Use this with
  `PrecomputedARASDataset` to amortise CLIP cost across all 15 runs.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

NUM_CLASSES = 7
EMBED_DIM = 512        # ViT-B/32 shared embedding dimension
NUM_PATCHES = 49       # 7x7 patch grid from ViT-B/32 at 224px (CLS dropped)
MAX_TOKENS = 77        # CLIP text context length
NUM_HEADS = 8          # cross-attention heads
HIDDEN_DIM = 256       # regression-head hidden width


def _encoder_grad_ctx(freeze: bool):
    """No-grad when the encoder is frozen; pass-through when LoRA-trainable."""
    return torch.no_grad() if freeze else contextlib.nullcontext()


def _encode_image_patches(clip_model, images: torch.Tensor) -> torch.Tensor:
    """Run CLIP ViT and return per-patch embeddings (B, 49, 512). Skips CLS."""
    vit = clip_model.visual
    dtype = clip_model.dtype
    x = vit.conv1(images.type(dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

    cls = vit.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls, x], dim=1)
    x = x + vit.positional_embedding.to(x.dtype)
    x = vit.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = vit.transformer(x)
    x = x.permute(1, 0, 2)
    x = vit.ln_post(x)
    x = x @ vit.proj
    return x[:, 1:, :].float()  # drop CLS, cast to fp32 for downstream training


def _encode_text_tokens(clip_model, tokens: torch.Tensor) -> torch.Tensor:
    """Run CLIP text transformer and return per-token embeddings (B, 77, 512)."""
    dtype = clip_model.dtype
    x = clip_model.token_embedding(tokens).type(dtype)
    x = x + clip_model.positional_embedding.type(dtype)
    x = x.permute(1, 0, 2)
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)
    x = clip_model.ln_final(x)
    return x.float()


class _RegressionHead(nn.Sequential):
    def __init__(self, in_dim: int = EMBED_DIM, hidden: int = HIDDEN_DIM, out_dim: int = NUM_CLASSES):
        super().__init__(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.Softmax(dim=-1),
        )


class MultimodalRegressor(nn.Module):
    """Cross-attention fusion model used for R0a, R1, R2a, R2b."""

    def __init__(self, clip_model=None, num_heads: int = NUM_HEADS, freeze_encoder: bool = True):
        super().__init__()
        self.clip_model = clip_model
        self.freeze_encoder = freeze_encoder
        if clip_model is not None and freeze_encoder:
            for p in self.clip_model.parameters():
                p.requires_grad_(False)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=EMBED_DIM, num_heads=num_heads, batch_first=True
        )
        self.head = _RegressionHead()

    def trainable_parameters(self):
        params = list(self.cross_attn.parameters()) + list(self.head.parameters())
        if self.clip_model is not None and not self.freeze_encoder:
            # LoRA adapters injected into the encoder are the only grad-enabled
            # CLIP params (base weights stay frozen by inject_lora).
            params += [p for p in self.clip_model.parameters() if p.requires_grad]
        return params

    def forward_emb(
        self,
        patches: torch.Tensor,
        text_emb: torch.Tensor,
        return_attention: bool = False,
    ):
        fused, attn = self.cross_attn(query=patches, key=text_emb, value=text_emb)
        pooled = fused.mean(dim=1)
        pred = self.head(pooled)
        if return_attention:
            return pred, attn
        return pred

    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        return_attention: bool = False,
    ):
        if self.clip_model is None:
            raise RuntimeError(
                "MultimodalRegressor was built without a clip_model; "
                "call forward_emb(patches, text_emb) instead."
            )
        with _encoder_grad_ctx(self.freeze_encoder):
            patches = _encode_image_patches(self.clip_model, images)
            text = _encode_text_tokens(self.clip_model, tokens)
        return self.forward_emb(patches, text, return_attention=return_attention)


class VisionOnlyRegressor(nn.Module):
    """Pure-vision baseline used for R0b. No text path, no cross-attention."""

    def __init__(self, clip_model=None, freeze_encoder: bool = True):
        super().__init__()
        self.clip_model = clip_model
        self.freeze_encoder = freeze_encoder
        if clip_model is not None and freeze_encoder:
            for p in self.clip_model.parameters():
                p.requires_grad_(False)
        self.head = _RegressionHead()

    def trainable_parameters(self):
        params = list(self.head.parameters())
        if self.clip_model is not None and not self.freeze_encoder:
            params += [p for p in self.clip_model.parameters() if p.requires_grad]
        return params

    def forward_emb(self, patches: torch.Tensor):
        pooled = patches.mean(dim=1)
        return self.head(pooled)

    def forward(self, images: torch.Tensor):
        if self.clip_model is None:
            raise RuntimeError(
                "VisionOnlyRegressor was built without a clip_model; "
                "call forward_emb(patches) instead."
            )
        with _encoder_grad_ctx(self.freeze_encoder):
            patches = _encode_image_patches(self.clip_model, images)
        return self.forward_emb(patches)


def build_model(condition: str, clip_model=None, freeze_encoder: bool = True) -> nn.Module:
    """Build the right regressor for `condition`.

    Pass `clip_model=None` when training from precomputed embeddings — the
    module skips the CLIP forward and only the trainable head/fusion live on
    the device. Pass `freeze_encoder=False` for LoRA fine-tuning (online mode),
    so the encoder forward runs with gradients enabled.
    """
    if condition == "R0b":
        return VisionOnlyRegressor(clip_model, freeze_encoder=freeze_encoder)
    return MultimodalRegressor(clip_model, freeze_encoder=freeze_encoder)
