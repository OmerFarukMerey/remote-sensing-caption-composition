"""Evaluation metrics and attention visualisation for Phase 2."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import ARASDataset, PrecomputedARASDataset, COMPOSITION_CLASSES
from .model import build_model
from .train import _collate_online, _collate_precomputed, _forward

EPS = 1e-8


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    return np.sum(p * np.log(p / q), axis=-1)


def _r2_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / np.where(ss_tot < EPS, EPS, ss_tot)


@torch.no_grad()
def predict_split(
    condition: str,
    seed: int,
    csv_path: str | Path,
    images_dir: str | Path,
    clip_model,
    image_transform,
    tokenizer,
    device: str,
    ckpt_path: str | Path,
    split: str = "test",
    batch_size: int = 32,
    precomputed_dir: Optional[str | Path] = None,
):
    precomputed = precomputed_dir is not None
    if precomputed:
        ds = PrecomputedARASDataset(csv_path, precomputed_dir, split, condition)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_precomputed)
        model = build_model(condition, clip_model=None).to(device)
    else:
        ds = ARASDataset(csv_path, images_dir, split, condition, image_transform, tokenizer)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_online)
        model = build_model(condition, clip_model).to(device)

    state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    # CLIP weights are intentionally missing (we share a frozen CLIP outside).
    model.eval()

    preds, gts, fns = [], [], []
    for batch in loader:
        pred, gt = _forward(model, batch, device, condition, precomputed)
        preds.append(pred.cpu().numpy())
        gts.append(gt.cpu().numpy())
        fns.extend(batch["filename"])
    return {
        "pred": np.concatenate(preds, axis=0),
        "gt": np.concatenate(gts, axis=0),
        "filenames": fns,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    diff = pred - gt
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    per_class_mae = np.mean(np.abs(diff), axis=0)
    r2 = _r2_per_class(gt, pred)
    kl = float(np.mean(_kl_divergence(gt, pred)))
    sum_violation = float(np.mean(np.abs(pred.sum(axis=-1) - 1.0)))
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "kl": kl,
        "sum_violation": sum_violation,
        "per_class_mae": dict(zip(COMPOSITION_CLASSES, per_class_mae.tolist())),
        "per_class_r2": dict(zip(COMPOSITION_CLASSES, r2.tolist())),
    }


def aggregate_runs(metrics_per_run: list[dict]) -> dict:
    """Mean ± std over seeds for a single condition."""
    keys = ["mse", "mae", "rmse", "kl", "sum_violation"]
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in metrics_per_run])
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out


def plot_attention(image_pil, attn_weights: np.ndarray, ax=None, alpha: float = 0.5):
    """Overlay attention (mean over text tokens) on the image as a 7×7 heatmap."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    if attn_weights.ndim == 3:  # (B, 49, 77)
        attn_weights = attn_weights[0]
    spatial = attn_weights.mean(axis=-1).reshape(7, 7)
    ax.imshow(image_pil)
    ax.imshow(
        spatial,
        cmap="hot",
        alpha=alpha,
        extent=(0, image_pil.size[0], image_pil.size[1], 0),
        interpolation="bilinear",
    )
    ax.axis("off")
    return ax
