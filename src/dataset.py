"""ARAS400k PyTorch dataset with leakage-free 5-condition text routing.

Two modes:

* **Online** — load PIL images and CLIP-tokenize captions on the fly. Use this
  when the encoder is also trained or when you don't want to precompute.
* **Precomputed** — read frozen CLIP patch / text embeddings produced by
  `src.precompute`. Eliminates the CLIP forward from the training loop and is
  the recommended path on M-series Macs (5–10× faster).

All 10 000 rows in `captions.csv` carry split=`synth`. We re-split deterministically
into 70/15/15 train/val/test using a fixed seed, without modifying the CSV.

Conditions
----------
R0a   : empty caption (architecture-controlled vision-only)
R0b   : no caption     (pure-vision architecture)
R1    : keyword string from NER over both vision_* captions
R2a   : masked vision_gemma3-4b
R2b   : masked vision_qwen3-vl-8b
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .sanitize import extract_keywords, mask_numbers

CONDITIONS = ("R0a", "R0b", "R1", "R2a", "R2b", "Rrand")
Condition = Literal["R0a", "R0b", "R1", "R2a", "R2b", "Rrand"]
Split = Literal["train", "val", "test"]

COMPOSITION_CLASSES = ("Tree", "Shrub", "Grass", "Crop", "Built-up", "Barren", "Water")
VISION_GEMMA = "vision_gemma3-4b"
VISION_QWEN = "vision_qwen3-vl-8b"


def make_splits(
    n: int,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[Split, np.ndarray]:
    """Deterministic 70/15/15 index split."""
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    return {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }


class ARASDataset(Dataset):
    """Online dataset: opens images and tokenizes text per __getitem__."""

    def __init__(
        self,
        csv_path: str | Path,
        images_dir: str | Path,
        split: Split,
        condition: Condition,
        image_transform,
        tokenizer,
        split_seed: int = 42,
        precompute_text: bool = True,
    ):
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}")

        self.images_dir = Path(images_dir)
        self.condition = condition
        self.image_transform = image_transform
        self.tokenizer = tokenizer

        df = pd.read_csv(csv_path)
        idx = make_splits(len(df), seed=split_seed)[split]
        self.df = df.iloc[idx].reset_index(drop=True)

        self._texts: list[str] | None = None
        if precompute_text and condition != "R0b":
            self._texts = [self._build_text(row) for _, row in self.df.iterrows()]

    def _build_text(self, row: pd.Series) -> str:
        c = self.condition
        if c == "R0a":
            return ""
        if c == "R1":
            return extract_keywords([row.get(VISION_GEMMA, ""), row.get(VISION_QWEN, "")])
        if c == "R2a":
            return mask_numbers(row.get(VISION_GEMMA, ""))
        if c == "R2b":
            return mask_numbers(row.get(VISION_QWEN, ""))
        if c == "Rrand":
            raise NotImplementedError(
                "Rrand is precomputed-only; build random tokens via "
                "src.precompute.precompute_texts(conditions=('Rrand',)) and "
                "use PrecomputedARASDataset."
            )
        return ""

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img_path = self.images_dir / row["filename"]
        image = Image.open(img_path).convert("RGB")
        image = self.image_transform(image)

        gt = torch.tensor(
            [float(row[c]) for c in COMPOSITION_CLASSES],
            dtype=torch.float32,
        ) / 100.0

        if self.condition == "R0b":
            return {"image": image, "gt": gt, "filename": row["filename"]}

        text = (
            self._texts[i] if self._texts is not None else self._build_text(row)
        )
        # CLIP tokenizer returns (1, 77); squeeze for batching by DataLoader.
        tokens = self.tokenizer([text], truncate=True).squeeze(0)
        return {
            "image": image,
            "tokens": tokens,
            "text": text,
            "gt": gt,
            "filename": row["filename"],
        }


class PrecomputedARASDataset(Dataset):
    """Precomputed dataset: reads CLIP patch+text tensors from disk.

    Expects `precomputed_dir` to contain `images.pt` (N, 49, 512), `filenames.json`
    (length N), and per text-condition `text_<cond>.pt` (N, 77, 512). Produced by
    `src.precompute.precompute_all`.
    """

    def __init__(
        self,
        csv_path: str | Path,
        precomputed_dir: str | Path,
        split: Split,
        condition: Condition,
        split_seed: int = 42,
    ):
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}")
        self.condition = condition

        precomputed_dir = Path(precomputed_dir)
        df = pd.read_csv(csv_path)
        filenames = json.loads((precomputed_dir / "filenames.json").read_text())
        if filenames != df["filename"].tolist():
            raise ValueError(
                "filenames.json does not match captions.csv ordering. "
                "Re-run precompute_all after dataset changes."
            )

        idx = make_splits(len(df), seed=split_seed)[split]
        self.idx = torch.as_tensor(idx, dtype=torch.long)
        self.df = df.iloc[idx].reset_index(drop=True)

        self.images = torch.load(precomputed_dir / "images.pt", map_location="cpu")
        self.texts = None
        if condition != "R0b":
            self.texts = torch.load(
                precomputed_dir / f"text_{condition}.pt", map_location="cpu"
            )

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        global_idx = int(self.idx[i].item())
        row = self.df.iloc[i]

        gt = torch.tensor(
            [float(row[c]) for c in COMPOSITION_CLASSES],
            dtype=torch.float32,
        ) / 100.0

        out = {
            "patches": self.images[global_idx].float(),
            "gt": gt,
            "filename": row["filename"],
        }
        if self.texts is not None:
            out["text_emb"] = self.texts[global_idx].float()
        return out
