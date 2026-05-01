"""Precompute frozen CLIP embeddings for all images and per-condition texts.

Runs CLIP forward exactly once over the dataset and stores fp16 tensors on disk.
Training then reads tensors instead of re-running the encoder every epoch — on
M4/MPS this turns a ~10 h grid into ~1.5-2 h.

Layout produced under `precomputed_dir/`:

    images.pt              # (N, 49, 512) fp16 — frozen CLIP patch embeddings
    filenames.json         # ordered list aligned with images.pt rows
    text_R0a.pt            # (N, 77, 512) fp16 — empty-caption text embeddings
    text_R1.pt             # (N, 77, 512) fp16
    text_R2a.pt            # (N, 77, 512) fp16
    text_R2b.pt            # (N, 77, 512) fp16

`R0b` does not need text and reuses `images.pt` only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from .dataset import VISION_GEMMA, VISION_QWEN
from .model import _encode_image_patches, _encode_text_tokens
from .sanitize import extract_keywords, mask_numbers

TEXT_CONDITIONS = ("R0a", "R1", "R2a", "R2b")


def _build_text(condition: str, row: pd.Series) -> str:
    if condition == "R0a":
        return ""
    if condition == "R1":
        return extract_keywords([row.get(VISION_GEMMA, ""), row.get(VISION_QWEN, "")])
    if condition == "R2a":
        return mask_numbers(row.get(VISION_GEMMA, ""))
    if condition == "R2b":
        return mask_numbers(row.get(VISION_QWEN, ""))
    raise ValueError(condition)


@torch.no_grad()
def precompute_images(
    csv_path: str | Path,
    images_dir: str | Path,
    out_dir: str | Path,
    clip_model,
    image_transform,
    device: str,
    batch_size: int = 64,
    overwrite: bool = False,
) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "images.pt"
    fn_path = out_dir / "filenames.json"
    if out_path.exists() and fn_path.exists() and not overwrite:
        print(f"[skip] {out_path} exists")
        return out_path

    df = pd.read_csv(csv_path)
    images_dir = Path(images_dir)
    n = len(df)
    embeddings = torch.empty(n, 49, 512, dtype=torch.float16)

    for start in tqdm(range(0, n, batch_size), desc="image embed"):
        rows = df.iloc[start : start + batch_size]
        batch = torch.stack(
            [image_transform(Image.open(images_dir / fn).convert("RGB")) for fn in rows["filename"]]
        ).to(device)
        emb = _encode_image_patches(clip_model, batch).to("cpu", torch.float16)
        embeddings[start : start + len(rows)] = emb

    torch.save(embeddings, out_path)
    fn_path.write_text(json.dumps(df["filename"].tolist()))
    print(f"[done] images: {tuple(embeddings.shape)} -> {out_path}")
    return out_path


@torch.no_grad()
def precompute_texts(
    csv_path: str | Path,
    out_dir: str | Path,
    clip_model,
    tokenizer,
    device: str,
    conditions: tuple[str, ...] = TEXT_CONDITIONS,
    batch_size: int = 256,
    overwrite: bool = False,
) -> dict[str, Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    n = len(df)
    paths: dict[str, Path] = {}

    for cond in conditions:
        out_path = out_dir / f"text_{cond}.pt"
        paths[cond] = out_path
        if out_path.exists() and not overwrite:
            print(f"[skip] {out_path} exists")
            continue

        texts = [_build_text(cond, row) for _, row in df.iterrows()]
        embeddings = torch.empty(n, 77, 512, dtype=torch.float16)
        for start in tqdm(range(0, n, batch_size), desc=f"text {cond}"):
            chunk = texts[start : start + batch_size]
            tokens = tokenizer(chunk, truncate=True).to(device)
            emb = _encode_text_tokens(clip_model, tokens).to("cpu", torch.float16)
            embeddings[start : start + len(chunk)] = emb
        torch.save(embeddings, out_path)
        print(f"[done] text {cond}: {tuple(embeddings.shape)} -> {out_path}")

    return paths


def precompute_all(
    csv_path: str | Path,
    images_dir: str | Path,
    out_dir: str | Path,
    clip_model,
    image_transform,
    tokenizer,
    device: str,
    overwrite: bool = False,
):
    precompute_images(
        csv_path, images_dir, out_dir, clip_model, image_transform, device, overwrite=overwrite
    )
    precompute_texts(csv_path, out_dir, clip_model, tokenizer, device, overwrite=overwrite)
