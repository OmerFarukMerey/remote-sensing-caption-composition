# Remote Sensing Caption Composition

**Multimodal land cover composition prediction from satellite imagery using CLIP and cross-attention — DI 725 term project.**

This project investigates how textual captions affect land-cover composition regression from satellite imagery, under strict no-leakage conditions. A frozen CLIP ViT-B/32 encodes the image into 49 patch tokens and the caption into 77 token embeddings; an 8-head cross-attention block fuses them; a small MLP head regresses the seven composition percentages.

---

## Research Question

> *How does the semantic richness of auxiliary text captions — ranging from a vision-only baseline, to simple keywords, to detailed qualitative descriptions — impact the accuracy of land cover composition regression from satellite images while strictly avoiding data leakage?*

The original Phase 1 question (vision-only vs.\ hybrid captions) was reformulated in Phase 2 because the dataset's `hybrid_*` and `text_qwen3-4b` captions embed the GT percentages verbatim, which would let the model solve the regression lexically rather than multimodally. Phase 2 uses only the leakage-free `vision_*` captions and applies regex-based numeric sanitization as a safety filter.

---

## Method Overview — Phase 2 (6 conditions, 3 seeds = 18 runs)

| Code | Architecture | Text input |
|---|---|---|
| **R0a** | CLIP + cross-attn + MLP | `""` (empty token; architecture-controlled vision-only) |
| **R0b** | CLIP + GAP + MLP | — (no text path; pure-vision floor) |
| **R1** | CLIP + cross-attn + MLP | noun-lemma NER over both `vision_*` captions |
| **R2a** | CLIP + cross-attn + MLP | sanitized `vision_gemma3-4b` |
| **R2b** | CLIP + cross-attn + MLP | sanitized `vision_qwen3-vl-8b` |
| **R-rand** | CLIP + cross-attn + MLP | 77 random CLIP vocab tokens (fixed seed; sanity control) |

```
[Satellite Image] ──► CLIP ViT ──► 49 patch tokens ──┐
                                                      ├──► Cross-Attention ──► Pool ──► MLP ──► [7 composition %]
[Caption Text] ──────► CLIP Text ─► 77 token embeds ─┘
```

Loss: MSE between predicted softmax distribution and ground-truth percentages (normalised to sum to 1). CLIP encoders are frozen; only fusion + MLP head ($\sim$1.18M params) train.

A caption-fidelity stratified evaluation tags each test tile `agree`/`disagree` by whether the conditioning caption mentions the dominant GT class.

---

## Dataset

This project uses the **ARAS400k** remote sensing dataset.

- 📦 Download: [Zenodo — ARAS400k](https://zenodo.org/records/18890661)
- 📁 Dataset repository: [github.com/caglarmert/ARAS400k](https://github.com/caglarmert/ARAS400k)

The dataset provides 10 000 RGB satellite tiles (256×256), pixel-level segmentation masks with 7 land cover classes, text captions from 5 model variants, and per-class composition percentages derived from the masks.

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

### Caption Sources (Phase 2 leakage policy)

| Column | Type | Used in Phase 2? | Reason |
|--------|------|---|---|
| `hybrid_gemma3-4b` | Hybrid | ❌ | embeds GT percentages |
| `hybrid_qwen3-vl-8b` | Hybrid | ❌ | embeds GT percentages |
| `text_qwen3-4b` | Text-only | ❌ | embeds GT percentages |
| `vision_gemma3-4b` | Vision-only | ✅ R2a | image-only generation, leakage-free |
| `vision_qwen3-vl-8b` | Vision-only | ✅ R2b | image-only generation, leakage-free |

---

## Project Structure

```
remote-sensing-caption-composition/
├── dataset/                   # not tracked; download from Zenodo
│   ├── images/                # 10 000 RGB tiles
│   ├── masks/                 # 10 000 segmentation masks
│   └── captions.csv
├── src/
│   ├── dataset.py             # ARASDataset (online + precomputed)
│   ├── sanitize.py            # mask_numbers, extract_keywords, fidelity helper
│   ├── model.py               # MultimodalRegressor + VisionOnlyRegressor
│   ├── train.py               # train_one_condition with W&B logging
│   ├── eval.py                # compute_metrics, attention viz
│   └── precompute.py          # one-shot CLIP embedding cache
├── notebooks/
│   ├── phase-1/phase1_poc.ipynb        # Phase 1 PoC
│   └── phase-2/phase2_main.ipynb       # Phase 2 end-to-end
├── reports/
│   ├── phase-1/phase1_report_ieee.pdf
│   └── phase-2/phase2_report.pdf       # Phase 2 IEEE report (this submission)
├── precomputed/               # not tracked; produced by src/precompute.py
├── checkpoints/               # not tracked; produced by training
├── requirements.txt
├── phase1_sub_2784254.zip     # Phase 1 submission archive
├── phase2_sub_2784254.zip     # Phase 2 submission archive
└── README.md
```

---

## Setup

```bash
git clone https://github.com/OmerFarukMerey/remote-sensing-caption-composition.git
cd remote-sensing-caption-composition
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # required by the R1 keyword extractor
```

Download the dataset from [Zenodo](https://zenodo.org/records/18890661) and place its contents under `dataset/` (relative paths are used throughout the notebooks).

Then open `notebooks/phase-2/phase2_main.ipynb` and run all cells. On first execution, `src/precompute.py` materialises CLIP outputs once (~5–10 min on M4); subsequent runs reuse the cached tensors. The full 18-run ablation grid finishes in under 50 minutes on an Apple M4 MacBook Pro.

---

## Results — Phase 2

Test set MSE, mean ± std over 3 seeds (lower is better):

| Condition | MSE ↓ | MAE ↓ | RMSE ↓ |
|-----------|-------|-------|-------|
| **R0a** (empty caption) | **0.00707 ± 0.00006** | **0.0406** | 0.0841 |
| R0b (pure vision)   | 0.01101 ± 0.00032 | 0.0554 | 0.1049 |
| R1 (NER keywords)   | 0.00760 ± 0.00016 | 0.0425 | 0.0872 |
| R2a (vision_gemma)  | 0.00751 ± 0.00026 | 0.0422 | 0.0867 |
| R2b (vision_qwen)   | 0.00739 ± 0.00020 | 0.0421 | 0.0859 |
| R-rand (random text) | 0.00787 ± 0.00008 | 0.0425 | 0.0887 |

**Headline findings.** (1) Removing cross-attention raises MSE by ~36 % relative — most of the apparent multimodal gain is architectural. (2) The empty-caption baseline beats random-token text with $p\!=\!8.6\!\times\!10^{-9}$ — content is not irrelevant. (3) Sanitised vision captions help on tiles where the caption mentions the dominant GT class and hurt on tiles where it does not, with the two effects approximately cancelling on the unstratified test set.

📄 Full Phase 2 report: [`reports/phase-2/phase2_report.pdf`](reports/phase-2/phase2_report.pdf)

### Experiment Tracking

All Phase 2 training runs (6 conditions × 3 seeds = 18 runs) are tracked on Weights & Biases:

📊 **W&B project:** [`mereyomerfaruk/di725-project`](https://wandb.ai/mereyomerfaruk/di725-project)

Run names follow `{condition}_seed{seed}` (e.g. `R0a_seed42`, `R2b_seed1337`, `Rrand_seed2024`).

---

## Acknowledgements

Dataset provided by [ARAS400k](https://github.com/caglarmert/ARAS400k). Image–text encoders from [OpenAI CLIP](https://github.com/openai/CLIP).
