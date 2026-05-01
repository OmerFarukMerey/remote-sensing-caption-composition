"""Training loop for the Information Richness ablation.

One condition × one seed per call. The notebook drives the 5 × 3 grid.

Two modes, transparently selected by `TrainConfig.precomputed_dir`:

* **Online**: builds an `ARASDataset`, runs CLIP + fusion + head every batch.
* **Precomputed**: builds a `PrecomputedARASDataset`, runs only fusion + head.
  Recommended on M-series Macs — same final weights, ~5–10× faster wall time.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .dataset import ARASDataset, PrecomputedARASDataset
from .model import build_model


@dataclass
class TrainConfig:
    condition: str
    seed: int = 42
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 5
    num_workers: int = 2
    use_wandb: bool = True
    wandb_project: str = "rs-caption-composition"
    wandb_tags: list[str] = field(default_factory=list)
    checkpoint_dir: str = "checkpoints"
    precomputed_dir: Optional[str] = None  # if set, skip CLIP forward


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate_online(batch):
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "gt": torch.stack([b["gt"] for b in batch]),
        "filename": [b["filename"] for b in batch],
    }
    if "tokens" in batch[0]:
        out["tokens"] = torch.stack([b["tokens"] for b in batch])
        out["text"] = [b["text"] for b in batch]
    return out


def _collate_precomputed(batch):
    out = {
        "patches": torch.stack([b["patches"] for b in batch]),
        "gt": torch.stack([b["gt"] for b in batch]),
        "filename": [b["filename"] for b in batch],
    }
    if "text_emb" in batch[0]:
        out["text_emb"] = torch.stack([b["text_emb"] for b in batch])
    return out


def _forward(model, batch, device, condition, precomputed: bool):
    gt = batch["gt"].to(device, non_blocking=True)
    if precomputed:
        patches = batch["patches"].to(device, non_blocking=True)
        if condition == "R0b":
            pred = model.forward_emb(patches)
        else:
            text_emb = batch["text_emb"].to(device, non_blocking=True)
            pred = model.forward_emb(patches, text_emb)
    else:
        images = batch["image"].to(device, non_blocking=True)
        if condition == "R0b":
            pred = model(images)
        else:
            tokens = batch["tokens"].to(device, non_blocking=True)
            pred = model(images, tokens)
    return pred, gt


def _run_epoch(model, loader, criterion, optimizer, device, condition, precomputed, train: bool):
    model.train(train)
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            pred, gt = _forward(model, batch, device, condition, precomputed)
            loss = criterion(pred, gt)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            bs = gt.size(0)
            total += loss.item() * bs
            n += bs
    return total / max(n, 1)


def _build_loaders(cfg: TrainConfig, csv_path, images_dir, image_transform, tokenizer, device):
    pin = device == "cuda"
    if cfg.precomputed_dir is not None:
        train_ds = PrecomputedARASDataset(csv_path, cfg.precomputed_dir, "train", cfg.condition)
        val_ds = PrecomputedARASDataset(csv_path, cfg.precomputed_dir, "val", cfg.condition)
        collate = _collate_precomputed
    else:
        train_ds = ARASDataset(
            csv_path, images_dir, "train", cfg.condition, image_transform, tokenizer
        )
        val_ds = ARASDataset(
            csv_path, images_dir, "val", cfg.condition, image_transform, tokenizer
        )
        collate = _collate_online

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        collate_fn=collate,
    )
    return train_loader, val_loader


def train_one_condition(
    cfg: TrainConfig,
    csv_path: str | Path,
    images_dir: str | Path,
    clip_model,
    image_transform,
    tokenizer,
    device: str,
):
    """Train a single (condition, seed) and return per-epoch history + best ckpt path."""
    _set_seed(cfg.seed)
    precomputed = cfg.precomputed_dir is not None
    train_loader, val_loader = _build_loaders(
        cfg, csv_path, images_dir, image_transform, tokenizer, device
    )

    # In precomputed mode CLIP isn't needed inside the model; build_model(None) leaves
    # only the trainable layers on device.
    model = build_model(cfg.condition, clip_model=None if precomputed else clip_model).to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    run = None
    if cfg.use_wandb:
        try:
            import wandb

            run = wandb.init(
                project=cfg.wandb_project,
                name=f"{cfg.condition}_seed{cfg.seed}",
                tags=[cfg.condition, f"seed{cfg.seed}"] + cfg.wandb_tags,
                config={**cfg.__dict__, "device": device, "precomputed": precomputed},
                reinit=True,
            )
        except Exception as exc:
            print(f"[wandb disabled] {exc}")
            run = None

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cfg.condition}_seed{cfg.seed}_best.pt"

    best_val = math.inf
    best_epoch = -1
    history: list[dict] = []

    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_loss = _run_epoch(
            model, train_loader, criterion, optimizer, device, cfg.condition, precomputed, train=True
        )
        val_loss = _run_epoch(
            model, val_loader, criterion, optimizer, device, cfg.condition, precomputed, train=False
        )
        scheduler.step()

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "condition": cfg.condition,
                    "seed": cfg.seed,
                    "state_dict": {
                        k: v for k, v in model.state_dict().items()
                        if not k.startswith("clip_model.")
                    },
                    "val_loss": val_loss,
                },
                ckpt_path,
            )

        elapsed = time.time() - t0
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_s": elapsed,
        }
        history.append(record)
        print(
            f"[{cfg.condition} s{cfg.seed}] ep{epoch:02d} "
            f"train={train_loss:.5f} val={val_loss:.5f} "
            f"best={best_val:.5f}@{best_epoch} ({elapsed:.1f}s)"
        )
        if run is not None:
            run.log(record)

        if epoch - best_epoch >= cfg.patience:
            print(f"  early stop (no val improvement for {cfg.patience} epochs)")
            break

    if run is not None:
        run.summary["best_val_loss"] = best_val
        run.summary["best_epoch"] = best_epoch
        run.finish()

    return {"history": history, "best_val": best_val, "best_epoch": best_epoch, "ckpt": ckpt_path}
