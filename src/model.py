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

import torch
import torch.nn as nn

NUM_CLASSES = 7
EMBED_DIM = 512  # ViT-B/32 shared embedding


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
    def __init__(self, in_dim: int = EMBED_DIM, hidden: int = 256, out_dim: int = NUM_CLASSES):
        super().__init__(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.Softmax(dim=-1),
        )


class MultimodalRegressor(nn.Module):
    """Cross-attention fusion model used for R0a, R1, R2a, R2b."""

    def __init__(self, clip_model=None, num_heads: int = 8):
        super().__init__()
        self.clip_model = clip_model
        if clip_model is not None:
            for p in self.clip_model.parameters():
                p.requires_grad_(False)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=EMBED_DIM, num_heads=num_heads, batch_first=True
        )
        self.head = _RegressionHead()

    def trainable_parameters(self):
        return list(self.cross_attn.parameters()) + list(self.head.parameters())

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
        with torch.no_grad():
            patches = _encode_image_patches(self.clip_model, images)
            text = _encode_text_tokens(self.clip_model, tokens)
        return self.forward_emb(patches, text, return_attention=return_attention)


class VisionOnlyRegressor(nn.Module):
    """Pure-vision baseline used for R0b. No text path, no cross-attention."""

    def __init__(self, clip_model=None):
        super().__init__()
        self.clip_model = clip_model
        if clip_model is not None:
            for p in self.clip_model.parameters():
                p.requires_grad_(False)
        self.head = _RegressionHead()

    def trainable_parameters(self):
        return list(self.head.parameters())

    def forward_emb(self, patches: torch.Tensor):
        pooled = patches.mean(dim=1)
        return self.head(pooled)

    def forward(self, images: torch.Tensor):
        if self.clip_model is None:
            raise RuntimeError(
                "VisionOnlyRegressor was built without a clip_model; "
                "call forward_emb(patches) instead."
            )
        with torch.no_grad():
            patches = _encode_image_patches(self.clip_model, images)
        return self.forward_emb(patches)


def build_model(condition: str, clip_model=None) -> nn.Module:
    """Build the right regressor for `condition`.

    Pass `clip_model=None` when training from precomputed embeddings — the
    module will skip the CLIP forward and only the trainable head/fusion live
    on the device.
    """
    if condition == "R0b":
        return VisionOnlyRegressor(clip_model)
    return MultimodalRegressor(clip_model)
