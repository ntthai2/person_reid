# Experiments & Implementation Plan

This document tracks proposed approaches, current status, and implementation details for the person feature extractor project.

---

## Current State

- **Phase**: Phase 1 implementation complete — starting training runs
- **Code**: Full Phase 1 codebase written and smoke-tested
- **Completed**: Dataset loaders, dual encoder, losses, trainer, evaluator, scripts
- **Environment**: PyTorch 2.7.1+cu128, transformers 5.3.0, HuggingFace datasets, FAISS-GPU
- **Hardware**: RTX 5090 (32 GB); BF16 AMP → 7.7 GB VRAM @ batch 64
- **Next action**: Run full Phase 1 training (20 epochs) and record baseline metrics

---

## Approaches

### Approach A — CLIP Dual Encoder Baseline *(Phase 1)*

**Status**: ✅ Implementation complete — training run pending

Fine-tune a pretrained CLIP ViT-B/16 model on text-image person pairs. Both towers are adapted with a lightweight projection MLP while the backbone is fine-tuned at a lower learning rate.

**Architecture**:
- Image tower: CLIP ViT-B/16 → MLP projection → 512-dim L2-normalized embedding
- Text tower: CLIP text transformer → MLP projection → 512-dim L2-normalized embedding
- Projection heads: Linear(512, 512) + ReLU + Linear(512, 512)

**Losses**:
- **InfoNCE** (symmetric): bidirectional contrastive loss over image-text pairs in batch
- **ID classification**: softmax over training identity labels (one head per tower, logit scale = 1/temperature)

**Training details**:
- Optimizer: AdamW
- LR: backbone 1e-5, projection heads 1e-4 (differential LR)
- Scheduler: cosine decay with linear warmup (2 epochs)
- Batch size: 64 (text-image pairs)
- **Mixed precision: BF16** (FP16 fails on Blackwell — CUBLAS unsupported; BF16 = 7.7 GB VRAM)
- Max epochs: 20

**Training data**: CUHK-PEDES, ICFG-PEDES, RSTPReid, IIITD-20K, ORBench (351,927 samples total)
- Labeled identities (remapped to contiguous): **31,892** (ICFG=27,591 + RSTPReid=3,701 + ORBench=8 sub-ids)
- CUHK-PEDES and IIITD-20K: `pid=-1` → InfoNCE only, skipped by ID loss

**Target Rank-1 on RSTPReid test (text→image)**: ≥ 68% (primary eval during training)
**Target Rank-1 on ICFG-PEDES (text→image)**: ≥ 55%

**Actual results**: *pending first full training run*

---

### Approach B — Local Attribute Alignment *(Phase 2)*

**Status**: Not started — blocked on Phase 1 training metrics

Adds a cross-modal interaction module between word tokens and image patch tokens to enable fine-grained grounding (e.g., "red jacket" ↔ torso region patches).

**Additional components**:
- **Cross-attention module**: 2–3 transformer layers; text tokens attend to image patches and vice versa
- **MLM auxiliary loss**: randomly mask 15% of text tokens, predict them using cross-attended image features — forces attribute-region grounding
- **Fusion**: learnable weighted sum of global (Approach A) and local (cross-attention) embeddings

**Losses added**: MLM (cross-entropy over masked tokens), retained InfoNCE + ID loss

**VRAM estimate**: ~7–8 GB

**Expected Rank-1 improvement**: +2–4% over Phase 1 baseline

**Reference architectures**: IRRA (CVPR 2023), PLIP, HAP

---

### Approach C — Multi-Task Image-Only Training *(Phase 3)*

**Status**: Not started — infrastructure ready (`image_only.py` written, `lambda_img` flag in config)

Extends the image encoder by jointly training on image-only ReID datasets alongside text-image pairs. The shared image encoder receives gradient signal from both tasks, improving visual representation quality and boosting image→image retrieval.

**Training setup**:
- Alternate between text-image batches (Approach A/B losses) and image-only batches
- Image-only losses: triplet loss (hard mining, margin 0.3) + ID classification loss
- Image-only datasets: DukeMTMC-reID, Market-1203, LaST, CAVIARa, GRID

**Implementation detail**:
- DataLoader yields mixed batches; loss is summed with a weighting factor `lambda_img` (tunable, default 0.5)
- Text encoder is frozen during image-only steps (no gradient needed)

**VRAM estimate**: ~7–8 GB at mixed batch size 64

**Expected Rank-1 improvement (image→image on DukeMTMC-reID)**: >85%

---

## Implementation Roadmap

### Phase 1 — Baseline (Priority)

- [x] Project scaffold (`src/`, `configs/`, `scripts/` + all `__init__.py`)
- [x] Dataset loaders (`src/datasets/text_image.py`)
  - [x] **CUHK-PEDES**: `datasets.Dataset.from_parquet()` over 9 shards; Arrow memory-map for O(1) row access; `pid=-1`
  - [x] **ICFG-PEDES**: `captions_cleaned.csv`; regex strips Windows path prefix `r".*[/\\]imgs[/\\]"` + backslash→slash; 27,591 IDs remapped 0-based
  - [x] **RSTPReid**: `data_captions.json` `[{id, img_path, captions:[5], split}]`; random caption per step; 3,701 IDs
  - [x] **IIITD-20K**: `Filtered.json` `{"idx": {Image_ID, Description_1, Description_2}}`; image ext fallback `.jpeg/.jpg/.png`; `pid=-1`
  - [x] **ORBench**: `train_annos.json` `[{id, file_path, caption, split}]`; RGB `_vis.jpg` images
  - [x] Pid remapping: per-dataset contiguous offset; total 31,892 labeled classes
  - [x] Augmentation: Resize(224) → HFlip → ColorJitter → ToTensor → Normalize(CLIP) → RandomErasing(p=0.3)
- [x] Model (`src/models/dual_encoder.py`)
  - [x] `DualEncoder` wrapping `CLIPModel` (transformers 5.x API: `vision_model → pooler_output → visual_projection`)
  - [x] MLP projection heads: Linear(512,512) + ReLU + Linear(512,512) per modality
  - [x] Learnable log-temperature (clamped to ±log(100))
- [x] Losses (`src/losses/contrastive.py`)
  - [x] `InfoNCELoss`: symmetric cross-entropy with diagonal targets
  - [x] `IDLoss`: masked (pid≥0) cross-entropy; separate image+text classifiers
  - [x] `TripletLoss`: batch-hard; supports soft margin (Phase 3 ready)
- [x] Trainer (`src/engine/trainer.py`): gradient accumulation, cosine+warmup LR, BF16 AMP
- [x] Evaluator (`src/engine/evaluator.py`): FAISS FlatIP; text→image + image→image; Rank-K + mAP
- [x] Scripts: `scripts/train.py` (--smoke-test, --resume), `scripts/eval.py`
- [x] Config: `configs/baseline.yaml`
- [x] **Smoke test**: 2-batch run in 1.6s, InfoNCE ≈ 4.16 ≈ log(64), BF16 7.7 GB VRAM ✅

**Next — Phase 1 Training Run:**
- [ ] Full 20-epoch training run on all 5 text+image datasets
- [ ] Record RSTPReid test R1/R5/R10 + mAP after each epoch
- [ ] Record ICFG-PEDES test metrics at epoch 20
- [ ] Save best checkpoint by RSTPReid R1

**Targets:**
- RSTPReid test R1 ≥ 60%, ICFG-PEDES R1 ≥ 50% (fine-tuned CLIP baseline)
- VRAM ≤ 8 GB ✅ (actual: 7.7 GB BF16)

---

### Phase 2 — Local Alignment

- [ ] Cross-attention interaction module (pluggable, disabled by config flag)
- [ ] MLM loss module
- [ ] Global/local embedding fusion
- [ ] Config: `configs/local_align.yaml`
- [ ] Ablation: global-only vs global+local on CUHK-PEDES / ICFG-PEDES

**Target**: R1 ≥ 73% on CUHK-PEDES (text→image); VRAM ≤ 8 GB

---

### Phase 3 — Multi-Task Training

- [x] `image_only.py`: loaders for DukeMTMC-reID, Market-1203, LaST, CAVIARa, GRID
  - [x] `FolderReIDDataset` (generic), cumulative pid offsets, contiguous class mapping
  - [x] GRID probe/gallery dirs; DukeMTMC `{pid}_{cam}_*.jpg`; LaST identity subfolders
- [x] `lambda_img` flag in config (set > 0 to enable image-only branch in trainer)
- [x] `TripletLoss` (batch-hard) in `contrastive.py`
- [ ] Run Phase 3 training: set `lambda_img: 0.3`, create `configs/multitask.yaml`
- [ ] Evaluate image→image on DukeMTMC-reID and ENTIRe-ID

**Target**: DukeMTMC-reID image→image R1 ≥ 85%; no regression on text→image metrics

---

## Results Log

| Run | Config | Epochs | RSTPReid R1 | ICFG R1 | Notes |
|-----|--------|--------|-------------|---------|-------|
| smoke | baseline.yaml | 1 (2 batches) | — | — | Loss≈4.16, 7.7GB BF16 ✅ |
| run1 | baseline.yaml | 0/20 **running** | — | — | NCE↓4.16→**1.36** (step 1170/5499 E1, 21%), ID flat (warmup); ~60 min/epoch; no-eval mode |

**Live training log**: `runs/train_baseline_run1.log`  
**Checkpoints**: `outputs/baseline_run1/ckpt_epoch{N:02d}.pt`

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Backbone | CLIP ViT-B/16 | ≤ 8 GB VRAM, strong visual-semantic prior, upgrade to ViT-L/14 if needed |
| Embedding dim | 512 | Native CLIP dim; no additional projection needed initially |
| Similarity | Cosine (L2-norm + dot product) | Numerically stable, standard for CLIP-style models |
| Gallery indexing | FAISS FlatIP | Exact search for PoC; migrate to IVF for large galleries |
| Image size | 224×224 | CLIP native resolution |
| Loss temperature | Learnable (initialized to CLIP default 0.07) | Adapts to fine-tuned embedding scale |
| Hard negative mining | Batch-hard for triplet (Phase 3) | Efficient, no external mining needed |

---

## Ablation Experiments (Planned)

| Experiment | Variables | Metric |
|---|---|---|
| Backbone scale | ViT-B/16 vs ViT-L/14 | R1 CUHK-PEDES, VRAM |
| Loss ablation | InfoNCE only vs +ID loss vs +triplet | R1, mAP |
| Training data scale | CUHK-PEDES only vs +ICFG+RSTPReid+IIITD | R1, mAP |
| Local alignment | Global-only vs +cross-attention | R1 delta |
| Multi-task weight | lambda_img ∈ {0.1, 0.3, 0.5, 1.0} | Both modality metrics |

---

## Notes & References

- **IRRA** (Jiang et al., CVPR 2023): 73.38% R1 CUHK-PEDES with ViT-B/16 — primary target to match
- **CLIP** (Radford et al., 2021): pretrained weights via `openai/clip-vit-base-patch16` on HuggingFace
- **LaST challenge**: long-term ReID with clothing changes — may require dedicated augmentation (color jitter, random erasing)
- **GRID dataset**: 10-fold cross-validation protocol; MATLAB `.mat` file contains pre-partitioned splits
- **CUHK-PEDES format**: train/val/test JSON with keys `id`, `file_path`, `captions`
