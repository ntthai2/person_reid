# Person Feature Extractor for Cross-Modal Person Retrieval

A research/PoC project for learning unified person representations that support both **text→image** and **image→image** retrieval in RGB surveillance/camera scenarios.

---

## Goal

Build a single feature extractor that maps person images and text descriptions into a shared embedding space, enabling:
- **Text → Image**: Retrieve person images from a natural-language query (e.g., "woman in blue floral dress carrying a backpack")
- **Image → Image**: Retrieve person images given a probe image (standard re-identification)

The model should generalize across cameras, viewpoints, and lighting conditions.

---

## Datasets

### Text + Image (Cross-Modal)

| Dataset | Train Pairs | Notes |
|---|---|---|
| CUHK-PEDES | 238,768 | Primary training corpus; **HuggingFace Parquet** format (`{image, text}`); no identity labels → InfoNCE only |
| ICFG-PEDES | ~54K | Rich attribute-level descriptions; raw captions in `captions_cleaned.csv` (Windows paths → strip to relative `imgs/…/file.jpg`); identity labels available |
| IIITD-20K | ~20K | 2 descriptions per image in `Filtered.json` as `{Image_ID, Description_1, Description_2}`; no splits → train-only; InfoNCE only |
| RSTPReid | ~4,101 | 5 captions per image; `[{id, img_path, captions, split}]` in `data_captions.json`; identity labels available |
| ORBench | ~large | RGB `_vis.jpg` surveillance images + rich text in `train_annos.json`; `{id, file_path, caption, split}`; identity labels available |

### Image Only (Re-Identification)

| Dataset | Identities | Images | Notes |
|---|---|---|---|
| DukeMTMC-reID | 1,812 | ~36K | Multi-camera; `bounding_box_train/` with `{pid}_{camid}_*.jpg` naming |
| Market-1203 | 1,203 | ~12K | Orientation-labeled; 128×64 px crops; images under `Market1203/` subfolder |
| LaST | 4,000+ | ~228K | Long-term, clothing change; identity subfolders under `train/`, `val/` |
| CAVIARa | ~72 | ~13K | Indoor surveillance benchmark; flat folder |
| GRID (underground_reid) | 250 | ~1,275 | CCTV/underground; splits in `features_and_partitions.mat` |
| ENTIRe-ID | — | test only | Held-out benchmark |

> **Excluded**: V-47_Images (synthetic, unlabeled), ORBench IR images — out of RGB scope. ORBench `vis/` RGB images **are included** in text+image training.
> **Note**: WARD dataset has no text annotations and is excluded.

---

## Hardware Target

| Resource | Spec |
|---|---|
| GPU | NVIDIA RTX 5090 |
| VRAM (BF16, batch 64) | **7.7 GB** actual (FP16 unsupported on Blackwell) |
| Backbone | CLIP ViT-B/16 (~150M params) |
| Upgrade path | CLIP ViT-L/14 if accuracy plateaus |

---

## Project Structure

```
person_reid/
├── data/
│   ├── image/           # Image-only datasets (DukeMTMC, Market, LaST, CAVIARa, GRID, ORBench)
│   └── text/            # Text+image datasets (CUHK-PEDES, ICFG-PEDES, RSTPReid, IIITD-20K)
├── src/
│   ├── datasets/
│   │   ├── text_image.py    # Loaders: CUHK-PEDES (HF Arrow), ICFG-PEDES (CSV), RSTPReid, IIITD-20K, ORBench
│   │   └── image_only.py    # Loaders: DukeMTMC, Market, LaST, CAVIARa, GRID
│   ├── models/
│   │   └── dual_encoder.py  # CLIP ViT-B/16 + MLP projection heads; BF16-safe
│   ├── losses/
│   │   └── contrastive.py   # InfoNCE (symmetric), IDLoss (masked), TripletLoss (batch-hard)
│   └── engine/
│       ├── trainer.py       # Training loop: BF16 AMP, cosine+warmup LR, gradient accumulation
│       └── evaluator.py     # FAISS FlatIP retrieval; Rank-K + mAP; text→image + image→image
├── configs/
│   └── baseline.yaml        # All hyperparameters, dataset paths, model spec
├── scripts/
│   ├── train.py             # python scripts/train.py --config configs/baseline.yaml [--smoke-test] [--resume PATH]
│   └── eval.py              # python scripts/eval.py --config configs/baseline.yaml --checkpoint PATH
├── outputs/                 # Checkpoints (ckpt_epoch{N:02d}.pt)
├── runs/                    # TensorBoard logs
├── README.md
└── EXPERIMENTS.md           # Progress tracker — source of truth for current state
```

---

## Evaluation Protocol

**Text → Image** (standard CUHK-PEDES / RSTPReid / ICFG-PEDES protocol):
- Rank-1, Rank-5, Rank-10 retrieval accuracy
- mAP

**Image → Image** (standard ReID protocol):
- Rank-1, Rank-5, Rank-10
- mAP on DukeMTMC-reID and ENTIRe-ID

**Performance baselines to beat**:
- CLIP ViT-B/16 zero-shot: ~55% R1 text→image on CUHK-PEDES
- Published IRRA (CVPR 2023): ~73.4% R1 on CUHK-PEDES (ViT-B/16)

---

## Retrieval Inference

Gallery embeddings are pre-computed offline (no runtime cost per query). Query encoding (text or image) targets < 50 ms on RTX 5090.
