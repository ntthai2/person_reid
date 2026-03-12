# Experiments & Implementation Plan

This document tracks proposed approaches, current status, and implementation details for the person feature extractor project.

---

## Current State

- **Phase**: Phase 2 training epoch 6/10 running; Phase 3 config ready
- **Code**: Full Phase 1 + Phase 2 + Phase 3 codebase implemented
- **Environment**: PyTorch 2.7.1+cu128, transformers 5.3.0, HuggingFace datasets, FAISS-GPU
- **Hardware**: RTX 5090 (32 GB); BF16 → Phase 1: 7.7 GB, Phase 2: ~15.8 GB VRAM @ batch 64
- **Best Phase 1 checkpoint**: `outputs/baseline_run1/ckpt_epoch16.pt` (RSTPReid R@1=45.30)
- **Phase 2 training**: PID 2876663, warm-started from E16, epoch 4/10 step ~20430; mlm↓10.93→1.49; temp collapsed to 0.019 (clamped at min_temp=0.04 for future runs) ✅
- **Phase 3 config**: `configs/multitask.yaml` created; smoke test pending (blocked by Phase 2 VRAM ~15.8 GB)

### ⚠️ ICFG-PEDES Data Quality Issues (discovered post-training)

1. **ID column bug in CSV**: `captions_cleaned.csv` `id` column is a per-caption sequential index (0–50201, 27,591 unique values), NOT person IDs. ICFG-PEDES.json has correct person IDs (4,102 unique identities).  
   - **Impact on Phase 1**: ID loss classified 27,591 fake "identities" → random chance throughout training. NCE was the only effective loss.
   - **Fix applied**: `ICFGPEDESDataset` now loads from JSON with proper person IDs when `json_path` is provided.
   
2. **Test-set leakage**: CSV contains all available images (train+test = 27,591 rows). Test split is not filtered.  
   - **Fix applied**: Training now uses `ICFG-PEDES.json` with `split="train"` filter.

3. **Missing files**: 78.6% of ICFG test images are absent from disk (15,599/19,848).  
   - **Decision**: ICFG evaluation skipped — RSTPReid is the primary eval metric.
   - **Fix applied**: JSON loader skips missing files with warning.

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
- [x] Full 20-epoch training run on all 5 text+image datasets — complete
- [x] Record RSTPReid test R1/R5/R10 + mAP per epoch — see Results Log
- [ ] Record ICFG-PEDES test R1/R5/R10 + mAP (eval in progress for E16)
- [x] Save best checkpoint by RSTPReid R1 → **ckpt_epoch16.pt** (R@1=45.30)

**Targets:**
- RSTPReid test R1 ≥ 60%, ICFG-PEDES R1 ≥ 50% (fine-tuned CLIP baseline)
- VRAM ≤ 8 GB ✅ (actual: 7.7 GB BF16)

---

### Phase 2 — Local Alignment *(started)*

- [x] `LocalAlignModule` — cross-attention (text→image) + MLM head (`src/models/local_align.py`)
- [x] `DualEncoder.forward_local()` — exposes patch tokens + `LocalAlignModule` fusion
- [x] `MLMLoss` in `contrastive.py`
- [x] Trainer updated: `_make_mlm_mask()` + MLM loss branch in `_train_epoch()`
- [x] Config: `configs/local_align.yaml` (10 epochs, lr_bb=5e-6, warm-start from E16)
- [x] Phase 2 training launched:  PID 2876663, `runs/train_local_align_run1.log`
- [ ] Eval sweep on `outputs/local_align_run1/`
- [ ] Record RSTPReid R@1 improvement over Phase 1 baseline

---

### Phase 3 — Multi-Task Training

- [x] `image_only.py`: loaders for DukeMTMC-reID, Market-1203, LaST, CAVIARa, GRID
  - [x] `FolderReIDDataset` (generic), cumulative pid offsets, contiguous class mapping
  - [x] GRID probe/gallery dirs; DukeMTMC `{pid}_{cam}_*.jpg`; LaST identity subfolders
- [x] `lambda_img` flag in config (set > 0 to enable image-only branch in trainer)
- [x] `TripletLoss` (batch-hard) in `contrastive.py`
- [x] **Bug fix**: `build_image_only_dataset` read `image_size` from wrong config key (`model` → `data`)
- [x] **Bug fix**: Trainer image-only branch used text-image ID classifier on image-only pids (OOB); now triplet-only for image-only batch
- [x] **Bug fix**: Image-only DataLoader used `shuffle=True` → near-zero positive pairs per batch; replaced with `IdentityBalancedSampler` (P=16 identities × K=4 images, verified 16 unique PIDs/batch ✅)
- [x] **Bug fix**: CAVIARa PID parser looked for `person(\d+)` prefix but actual filenames use `{pid:04d}{frame:03d}.jpg` format; parser updated to handle both formats; 72 unique PIDs confirmed ✅
- [x] **Bug fix**: Duke/ENTIRe-ID parser `int(parts[1][1:])` failed on `c021s0` format; fixed with regex `c(\d+)` extraction
- [x] **Fix**: GRID disabled for training (singleton probe images = no positive pairs for triplet); eval-only dataset
- [x] **New dataset**: WARD added (`WARDDataset`: 70 identities × 3 cams × ~23 frames = 4,786 images, 68.4 avg/id)
- [x] Phase 3 final image-only training data: Duke (702 ids, 23.5/id), Market (1203, 7.1), LaST (5000, 14.2), CAVIARa (72, 16.9), WARD (70, 68.4) → 7,047 total IDs, 440 PK-balanced batches/epoch ✅
- [x] Full DataLoader smoke-test: batch=(64, 3, 224, 224), 16 unique PIDs per batch ✅
- [x] **Fix**: ENTIRe-ID evaluation added (10,415 gallery, 2,741 PIDs); `exclude_self=True` in `_compute_metrics` avoids trivial self-match
- [x] **Fix**: `eval_image_reid` updated to use `image_reid_datasets` section in multitask.yaml config (Duke + ENTIRe-ID)
- [x] **Fix**: Temperature reset on `--init-from` warm-start (prevents inheriting collapsed temp ≈ 0.027 from Phase 1)
- [x] **Fix**: `min_temperature: 0.04` in all configs + `DualEncoder._max_logit_scale` clamps logit_scale at `log(25)≈3.22` (prevents NCE gradient death)
- [x] `configs/multitask.yaml` created (`lambda_img: 0.3`, 10 epochs, warm-start from E16)
- [x] **Fix**: `ICFGPEDESDataset.samples` property added (returns `_records`) so evaluator fast-path works without re-loading images for text/pid extraction
- [x] **Fix**: `CUHKPEDESDataset` had duplicate `__len__` + `__getitem__` methods — removed duplicates
- [x] **Fix**: Trainer now logs `loss_tri` to TensorBoard + print string when `use_image_only=True` (Phase 3)
- [x] **Fix**: `eval_sweep.py` summary table now also prints per-epoch image→image ReID metrics (Duke, ENTIRe-ID) when present
- [x] **New**: ORBench text→image evaluation added (`_eval_orbench_text2image`): 18,050 test images + 18,050 text descriptions, all present on disk; added as `secondary_text_image` in `multitask.yaml` and `local_align.yaml`
- [x] **Fix**: Evaluator `_eval_text2image` fast-path via `gallery_ds.samples` metadata avoids re-loading images for caption/pid extraction (both RSTPReid and ICFG supported)
- [x] **Fix**: Evaluator now uses ALL captions per gallery image as separate text queries (RSTPReid: 2 captions × 1000 images = 2000 queries, matching standard multi-caption protocol)
- [x] **Bug fix**: `_embed_images` assumed `(images, text, pids)` format but `_eval_image2image` passes `collate_image_only` tuples `(images, pids, camids)` — causing camids to be used as pids, making all img2img metrics completely wrong. Fixed via isinstance check on batch[1].
- [x] **Bug fix**: `CUHKPEDESDataset.__getitem__` referenced `np.ndarray` but `numpy` was never imported in `text_image.py` — fixed by adding `import numpy as np`.
- [x] **Bug fix**: `--resume` path in `train.py` restored model + optimizer but not `id_loss` classifier weights or `global_step`, and always restarted from epoch 1 instead of continuing. Fixed: now restores `id_loss`, `global_step`, and `start_epoch = ckpt["epoch"] + 1`.
- [x] **Bug fix**: `eval.py` loaded checkpoint with `torch.load(path, map_location=device)` (no `weights_only=False`), which fails in PyTorch 2.x where default changed to `True`. Fixed.
- [x] **Code quality**: Duplicate comment section header in `image_only.py` for CAVIARa — removed.
- [x] **Code quality**: Misleading `local_align.py` comment "init α=0 → start purely global" — both wrong (sigmoid(0)=0.5, not 0; and α=0 is purely local, not global). Fixed.
- [x] **eval_sweep.py**: Added elapsed-time reporting (seconds) per checkpoint and `[i/N]` progress counter.
- [x] **auto_eval.sh**: Added configurable timeout (6th arg, default 1440min=24h) to prevent infinite loop if training crashes.
- [x] **README.md**: Updated dataset stats (correct DukeMTMC/Market/LaST sizes), added WARD to image-only table, added `local_align.py` to project structure, corrected GRID note (eval-only), removed incorrect WARD exclusion note.
- [x] **Critical bug fix**: `_GalleryDataset` + `_collate` were defined inside `_eval_orbench_text2image()` method — closure-defined classes cannot be pickled by Python, so `DataLoader(num_workers=4)` would crash with `AttributeError: Can't get local object ...`. Fixed by moving to module-level `_ORBenchGalleryDataset` class + `_collate_orbench` function. Validated: `pickle.dumps()` succeeds. ✅
- [x] **Phase 2 training**: COMPLETE (10/10 epochs, ~5400 steps/epoch, ~570s/epoch)
- [x] **Phase 2 eval sweep**: COMPLETE — RSTPReid E01 best R@1=12.35‡; ORBench E03 best R@1=35.14‡. Results saved to `runs/sweep_local_align_run1.json`.
- [x] **Auto-eval bug diagnosed**: Phase 2 auto_eval.sh process (PID 3006325) crashed with `syntax error: unexpected ')'` because bash reads scripts progressively — edits made to the file mid-run produced a stale view. Sweep re-run manually successfully.
- [x] **Phase 3 smoke test**: PASSED — 2 batches in 4s; temperature=0.0700 confirmed (reset from Phase 1's 0.0288); VRAM 17 GB; all dataloaders (text+image + image-only triplet) loading correctly.
- [ ] Launch Phase 3a training warm-started from Phase 1 E16
- [ ] Evaluate image→image on DukeMTMC-reID and ENTIRe-ID

**Phase 3b (deprioritized)**: Originally planned to warm-start from Phase 2 E10 vs Phase 1 E16. Given Phase 2's catastrophic RSTPReid collapse (3-12% vs Phase 1's 45.3%), warm-starting from Phase 2 checkpoints is unlikely to benefit Phase 3. Skip unless Phase 3a underperforms.
```bash
# If needed later:
python scripts/train.py --config configs/multitask.yaml \
    --init-from outputs/local_align_run1/ckpt_epoch10.pt \
    --checkpoint-dir outputs/multitask_run2 \
    --no-eval > runs/train_multitask_run2.log 2>&1 &
```
Note: temperature reset from Phase 2's collapsed 0.017 → back to 0.07 via `--init-from` code path. ✓

**Target**: DukeMTMC-reID image→image R1 ≥ 85%; RSTPReid R@1 ≥ 46% (beat Phase 1, multi-caption protocol)

---

## Results Log

| Run | Config | Epoch | RSTPReid R1 | RSTPReid R5 | RSTPReid R10 | RSTPReid mAP | ICFG R1 | Notes |
|-----|--------|-------|-------------|-------------|--------------|--------------|---------|-------|
| smoke | baseline.yaml | — | — | — | — | — | — | Loss≈4.16, 7.7GB BF16 ✅ |
| run1 E01 | baseline.yaml | 1/20 | 33.70 | 60.00 | 72.60 | 40.25 | — | |
| run1 E02 | baseline.yaml | 2/20 | 39.60 | 66.40 | 78.00 | 45.30 | — | |
| run1 E03 | baseline.yaml | 3/20 | 37.50 | 65.70 | 76.60 | 43.49 | — | |
| run1 E04 | baseline.yaml | 4/20 | 40.50 | 64.60 | 75.70 | 45.64 | — | |
| run1 E05 | baseline.yaml | 5/20 | 41.50 | 67.70 | 77.40 | 46.40 | — | |
| run1 E06 | baseline.yaml | 6/20 | 38.70 | 64.50 | 74.80 | 44.97 | — | |
| run1 E07 | baseline.yaml | 7/20 | 41.30 | 68.30 | 76.30 | 46.65 | — | |
| run1 E08 | baseline.yaml | 8/20 | 43.30 | 67.40 | 78.20 | 48.10 | — | |
| run1 E09 | baseline.yaml | 9/20 | 42.40 | 67.10 | 76.70 | 46.95 | — | |
| run1 E10 | baseline.yaml | 10/20 | 39.80 | 66.30 | 76.00 | 45.70 | — | |
| run1 E11 | baseline.yaml | 11/20 | 43.70 | 67.40 | 77.00 | 48.24 | — | |
| run1 E12 | baseline.yaml | 12/20 | 42.90 | 68.80 | 78.70 | 47.94 | — | |
| run1 E13 | baseline.yaml | 13/20 | 43.50 | 66.00 | 75.30 | 47.58 | — | |
| run1 E14 | baseline.yaml | 14/20 | 44.40 | 66.70 | 74.90 | 48.49 | — | |
| run1 E15 | baseline.yaml | 15/20 | 45.20 | 67.10 | 76.20 | 49.58 | — | |
| **run1 E16** | baseline.yaml | **16/20** | **45.30** | **66.60** | **75.80** | **49.10** | N/A† | **BEST RSTPReid R@1** |
| la_run1 E01 | local_align.yaml | 1/10 | 12.35‡ | 38.95‡ | 51.55‡ | 21.80‡ | N/A† | Phase 2 E01; ORBench R@1=35.07% |
| la_run1 E02 | local_align.yaml | 2/10 | 6.70‡ | 23.00‡ | 36.55‡ | 12.79‡ | N/A† | RSTPReid collapsing; ORBench=34.39% |
| la_run1 E03 | local_align.yaml | 3/10 | 3.15‡ | 12.25‡ | 21.20‡ | 6.88‡ | N/A† | ORBench=35.14% (best ORBench) |
| la_run1 E04 | local_align.yaml | 4/10 | 4.05‡ | 12.10‡ | 21.95‡ | 7.37‡ | N/A† | |
| la_run1 E05 | local_align.yaml | 5/10 | 3.15‡ | 12.80‡ | 22.40‡ | 7.14‡ | N/A† | |
| la_run1 E06 | local_align.yaml | 6/10 | 3.15‡ | 14.00‡ | 24.65‡ | 7.46‡ | N/A† | |
| la_run1 E07 | local_align.yaml | 7/10 | 3.40‡ | 12.90‡ | 21.30‡ | 7.21‡ | N/A† | |
| la_run1 E08 | local_align.yaml | 8/10 | 3.35‡ | 13.05‡ | 20.65‡ | 7.14‡ | N/A† | |
| la_run1 E09 | local_align.yaml | 9/10 | 3.30‡ | 12.10‡ | 20.00‡ | 6.70‡ | N/A† | |
| **la_run1 E10** | local_align.yaml | **10/10** | **3.15‡** | **12.30‡** | **20.10‡** | **6.71‡** | N/A† | Phase 2 complete; RSTPReid best=E01; ORBench best=E03 |

†ICFG evaluation skipped: 78.6% of test images missing from disk.
‡Phase 2 results use **multi-caption protocol** (2 captions × 1000 images = 2000 queries). Phase 1 used single-caption (1000 queries). Protocol change makes numbers NOT directly comparable to Phase 1.

> **⚠️ Eval protocol change (Phase 2 onwards)**: Evaluator updated to use ALL captions per gallery image as separate queries (RSTPReid: 2 captions × 1000 images = 2000 queries). Phase 1 results in the table above used single-caption (1000 queries). Phase 2+ eval sweep results will use the new multi-caption protocol. Results may differ by 1–3% R@1. For a fair comparison, re-run Phase 1 eval_sweep with the current evaluator code.

### Phase 1 Analysis

- RSTPReid R@1 plateaued at **45.3%** (vs target 60%). Likely causes:
  - Temperature collapsed 0.07→0.0285 (aggressive), reducing intra-batch diversity signal
  - NCE near-zero by epoch 5 (batch-size 64 too easy for fine-tuned CLIP); ID loss dominated late training
  - No token-level / patch-level alignment (global embeddings only)
- Result is solid improvement over CLIP zero-shot (~25–30% RSTPReid R@1)
- Phase 2 local alignment expected to close gap toward 60%+ target

### Phase 2 Analysis (COMPLETE — disappointing results)

- Temperature collapse carry-over: Phase 2 warm-started with temp≈0.0275 (inherited from Phase 1), declining to ~0.016 by epoch 3
  - Root cause: `--init-from` warm-start did NOT reset temperature (fix was applied after Phase 2 launched; applies to Phase 3)
  - The `min_temperature: 0.04` clamp (added post-launch) did NOT affect the running process
  - Practical effect: **NCE ≈ 0 throughout all 10 Phase 2 epochs**; only ID loss + MLM loss were active
- MLM loss converged 10.93 → 1.41 by epoch 1, stabilized — local alignment IS learning cross-modal token alignment
- **Critical finding**: RSTPReid R@1 COLLAPSED: 12.35% E01 → 3.15% E03 and stayed flat. Multi-caption protocol with 2000 queries confirmed this is genuine regression, not a metric artifact
- **ORBench stayed stable**: 35.07% E01 → 35.14% E03 (best) → 30% E05 onwards. The divergence between RSTPReid and ORBench collapse rates is informative: ORBench uses longer, richer description text which may benefit from MLM grounding; RSTPReid uses shorter queries that rely more on global cross-modal CLIP alignment
- **Root cause**: MLM-driven gradient updates on the shared CLIP backbone break the global cross-modal alignment. Without NCE as a regularizer, ID loss and MLM pull text embeddings toward classification/token-prediction subspaces not aligned with image embeddings
- **Auto-eval.sh note**: Phase 2 auto-eval process (PID 3006325) failed with `syntax error: unexpected token ')'`. Root cause: bash reads scripts "progressively" (not all at once); the running process saw a partial/stale version of the file after the WAITED/MAX_MINUTES edits were applied mid-run. Eval sweep was re-run manually and succeeded.
- **Conclusion**: Phase 2 is NOT usable for initializing Phase 3. We use Phase 1 E16 as Phase 3 init point.

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
