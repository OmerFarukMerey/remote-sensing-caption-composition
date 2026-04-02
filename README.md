# Remote Sensing Caption Composition

**Multimodal land cover composition prediction from satellite imagery using CLIP and cross-attention.**

This project investigates whether natural language captions can improve the estimation of land cover composition percentages from satellite imagery — without relying on pixel-level segmentation masks at inference time. A Vision Transformer (ViT) encodes the image into patch tokens, a text encoder processes the caption, and a cross-attention layer fuses the two modalities before a regression head predicts per-class composition percentages.

---

## Research Question

> *Does the informativeness of text captions — ranging from vision-only descriptions to composition-aware hybrid captions — improve land cover composition prediction from satellite imagery beyond a vision-only baseline, and if so, by how much?*

---

## Method Overview

Three experimental conditions are evaluated progressively:

| Condition | Input | Description |
|-----------|-------|-------------|
| **Baseline** | Image only | ViT → MLP → composition % |
| **Method 2** | Image + vision caption | Caption generated from image alone (no % knowledge) |
| **Method 3** | Image + hybrid caption | Caption generated with composition % knowledge |

The cross-attention mechanism uses image patch tokens as queries and text tokens as keys/values, allowing each image region to attend to relevant parts of the caption.

```
[Satellite Image] ──► ViT Encoder ──► Patch Tokens ──┐
                                                      ├──► Cross-Attention ──► Pool ──► MLP ──► [7 composition %]
[Caption Text] ──────► Text Encoder ──► Word Tokens ──┘
```

Loss: Mean Squared Error (MSE) between predicted and ground truth composition percentages.

---

## Dataset

This project uses the **ARAS400k** remote sensing dataset.

- 📦 Download: [Zenodo — ARAS400k](https://zenodo.org/records/18890661)
- 📁 Dataset repository: [github.com/caglarmert/ARAS400k](https://github.com/caglarmert/ARAS400k)

The dataset provides:
- True-color satellite images (`images/`)
- Pixel-level segmentation masks (`masks/`) with 7 land cover classes
- Text captions from 5 model variants (`captions.csv`)
- Land cover composition percentages derived from masks (`Tree`, `Shrub`, `Grass`, `Crop`, `Built-up`, `Barren`, `Water`)

### Land Cover Classes

| Class | RGB |
|-------|-----|
| Tree | (0, 100, 0) |
| Shrub | (255, 182, 193) |
| Grass | (154, 205, 50) |
| Crop | (255, 215, 0) |
| Built-up | (139, 69, 19) |
| Barren | (211, 211, 211) |
| Water | (0, 0, 255) |

### Caption Sources

| Column | Type | Composition-aware? |
|--------|------|--------------------|
| `hybrid_gemma3-4b` | Hybrid | ✅ Yes |
| `hybrid_qwen3-vl-8b` | Hybrid | ✅ Yes |
| `text_qwen3-4b` | Text-only | ✅ Yes |
| `vision_gemma3-4b` | Vision-only | ❌ No |
| `vision_qwen3-vl-8b` | Vision-only | ❌ No |

---

## Project Structure

```
rs-caption-composition/
├── data/                   # Symlink or path to dataset (not included)
├── notebooks/
│   ├── phase1_poc.ipynb    # Proof of concept — pipeline validation
│   ├── phase2_results.ipynb
│   └── phase3_ablation.ipynb
├── src/
│   ├── dataset.py          # PyTorch Dataset class
│   ├── model.py            # Architecture definition
│   └── train.py            # Training loop
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/<your-username>/rs-caption-composition.git
cd rs-caption-composition
pip install -r requirements.txt
```

Download the dataset from [Zenodo](https://zenodo.org/records/18890661) and place it under `data/` using relative paths as expected by the notebooks.

### Requirements

```
torch
torchvision
ftfy
regex
tqdm
pandas
pillow
matplotlib
git+https://github.com/openai/CLIP.git
```

---

## Results

*To be updated as experiments are completed.*

| Model | MSE ↓ |
|-------|-------|
| Vision only (baseline) | — |
| Vision + vision caption | — |
| Vision + hybrid caption | — |

---

## Acknowledgements

Dataset provided by [ARAS400k](https://github.com/caglarmert/ARAS400k). Image-text encoders from [OpenAI CLIP](https://github.com/openai/CLIP).
