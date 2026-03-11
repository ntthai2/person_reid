"""
Evaluation engine: pre-compute gallery embeddings, FAISS search, Rank-K + mAP.

Supports:
  - Text → Image retrieval (primary: CUHK-PEDES / RSTPReid / ICFG-PEDES protocol)
  - Image → Image retrieval (DukeMTMC-reID / ENTIRe-ID protocol)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

from src.datasets.text_image import (
    RSTPReidDataset,
    ICFGPEDESDataset,
    build_val_transform,
    collate_text_image,
    Sample,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _embed_images(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_embs, all_pids = [], []
    for batch in loader:
        images, _, pids = batch
        images = images.to(device)
        emb = model.encode_image(images)
        all_embs.append(emb.cpu().float().numpy())
        all_pids.append(pids.numpy())
    return np.concatenate(all_embs), np.concatenate(all_pids)


@torch.no_grad()
def _embed_texts(model, processor, texts: list[str],
                 device, batch_size: int = 128) -> np.ndarray:
    model.eval()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        enc = processor(
            text=batch_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=77,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        emb = model.encode_text(input_ids, attention_mask)
        all_embs.append(emb.cpu().float().numpy())
    return np.concatenate(all_embs)


def _build_faiss_index(gallery_embs: np.ndarray,
                       use_gpu: bool = True) -> faiss.Index:
    d = gallery_embs.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product on L2-normalised → cosine
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(gallery_embs.astype(np.float32))
    return index


def _compute_metrics(query_pids: np.ndarray, gallery_pids: np.ndarray,
                     indices: np.ndarray, k_values: list[int]) -> dict:
    """Compute Rank-K accuracy and mAP from retrieved indices."""
    max_k = indices.shape[1]
    ranks_hit = {k: 0 for k in k_values}
    ap_sum = 0.0
    n = len(query_pids)

    for i, qpid in enumerate(query_pids):
        retrieved = gallery_pids[indices[i]]  # (K,)
        correct = (retrieved == qpid)

        # Rank-K
        for k in k_values:
            if any(correct[:k]):
                ranks_hit[k] += 1

        # AP
        positions = np.where(correct)[0] + 1  # 1-indexed
        if len(positions) == 0:
            continue
        precision_at_k = [(j + 1) / pos for j, pos in enumerate(positions)]
        ap_sum += float(np.mean(precision_at_k))

    metrics = {f"R{k}": ranks_hit[k] / n * 100 for k in k_values}
    metrics["mAP"] = ap_sum / n * 100
    return metrics


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Evaluate text→image and image→image retrieval.

    Args:
        cfg: Full config dict.
        device: Torch device.
    """

    def __init__(self, cfg: dict, device: Optional[torch.device] = None):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k_values = cfg["evaluation"]["k_values"]
        self.faiss_gpu = cfg["evaluation"]["faiss_use_gpu"]

    # ------------------------------------------------------------------
    # Text → Image
    # ------------------------------------------------------------------

    def _eval_text2image(self, model, dataset_name: str) -> dict:
        """Build gallery from image embeddings, query with text embeddings."""
        cfg = self.cfg
        transform = build_val_transform(cfg["data"]["image_size"])
        processor: CLIPProcessor = model.processor

        if dataset_name == "rstp_reid":
            gallery_ds = RSTPReidDataset(
                json_path=cfg["data"]["text_image"]["rstp_reid"]["json"],
                img_root=cfg["data"]["text_image"]["rstp_reid"]["img_root"],
                split="test", transform=transform, deterministic=True,
            )
        elif dataset_name == "icfg_pedes":
            gallery_ds = ICFGPEDESDataset(
                csv_path=cfg["data"]["text_image"]["icfg_pedes"]["csv"],
                img_root=cfg["data"]["text_image"]["icfg_pedes"]["img_root"],
                transform=transform,
            )
        else:
            raise ValueError(f"Unknown eval dataset: {dataset_name}")

        loader = DataLoader(
            gallery_ds, batch_size=128, shuffle=False,
            num_workers=4, collate_fn=collate_text_image,
        )

        # Embed gallery images
        gallery_embs, gallery_pids = _embed_images(model, loader, self.device)

        # Embed query texts (one per gallery sample, same order)
        query_texts = [s.text for s in gallery_ds]   # type: ignore[attr-defined]
        query_pids = gallery_pids.copy()

        # If the dataset provides a list interface iterate directly
        try:
            query_texts = [gallery_ds[i].text for i in range(len(gallery_ds))]
            query_pids = np.array([gallery_ds[i].pid for i in range(len(gallery_ds))])
        except Exception:
            pass

        query_embs = _embed_texts(model, processor, query_texts, self.device)

        # FAISS search
        index = _build_faiss_index(gallery_embs, self.faiss_gpu)
        max_k = max(self.k_values)
        _, indices = index.search(query_embs.astype(np.float32), max_k)

        return _compute_metrics(query_pids, gallery_pids, indices, self.k_values)

    # ------------------------------------------------------------------
    # Image → Image
    # ------------------------------------------------------------------

    def _eval_image2image(self, model, img_dir: str,
                          query_dir: Optional[str] = None) -> dict:
        from src.datasets.image_only import FolderReIDDataset, collate_image_only

        transform = build_val_transform(self.cfg["data"]["image_size"])
        gallery_ds = FolderReIDDataset(img_dir, transform=transform)
        gallery_loader = DataLoader(
            gallery_ds, batch_size=128, shuffle=False,
            num_workers=4, collate_fn=collate_image_only,
        )
        gallery_embs, gallery_pids = _embed_images(model, gallery_loader, self.device)

        if query_dir:
            query_ds = FolderReIDDataset(query_dir, transform=transform)
            query_loader = DataLoader(
                query_ds, batch_size=128, shuffle=False,
                num_workers=4, collate_fn=collate_image_only,
            )
            query_embs, query_pids = _embed_images(model, query_loader, self.device)
        else:
            query_embs, query_pids = gallery_embs, gallery_pids

        index = _build_faiss_index(gallery_embs, self.faiss_gpu)
        max_k = max(self.k_values)
        _, indices = index.search(query_embs.astype(np.float32), max_k)

        return _compute_metrics(query_pids, gallery_pids, indices, self.k_values)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, model, split: str = "test") -> dict:
        metrics = {}
        primary = self.cfg["evaluation"]["primary"]
        ti_metrics = self._eval_text2image(model, primary)
        metrics.update({f"text2img/{k}": v for k, v in ti_metrics.items()})
        print(f"Text→Image ({primary}): {ti_metrics}")

        if self.cfg["evaluation"]["eval_image_reid"]:
            duke_cfg = self.cfg["data"]["image_only"]["duke"]
            if duke_cfg["enabled"]:
                ii_metrics = self._eval_image2image(
                    model,
                    img_dir=duke_cfg["gallery_dir"],
                    query_dir=duke_cfg["query_dir"],
                )
                metrics.update({f"img2img/duke/{k}": v for k, v in ii_metrics.items()})
                print(f"Image→Image (DukeMTMC): {ii_metrics}")

        return metrics
