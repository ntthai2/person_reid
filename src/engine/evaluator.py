"""
Evaluation engine: pre-compute gallery embeddings, FAISS search, Rank-K + mAP.

Supports:
  - Text → Image retrieval (primary: CUHK-PEDES / RSTPReid / ICFG-PEDES protocol)
  - Image → Image retrieval (DukeMTMC-reID / ENTIRe-ID protocol)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor

from src.datasets.text_image import (
    RSTPReidDataset,
    ICFGPEDESDataset,
    build_val_transform,
    collate_text_image,
    Sample,
)


# ---------------------------------------------------------------------------
# ORBench gallery dataset (module-level so DataLoader multi-processing can pickle it)
# ---------------------------------------------------------------------------

class _ORBenchGalleryDataset(Dataset):
    """Minimal dataset for ORBench RGB gallery images.

    Defined at module level so it can be pickled by DataLoader worker processes.
    """

    def __init__(self, items, img_root, transform):
        self.items = items           # list of [pid, rel_path]
        self.img_root = img_root     # pathlib.Path to ORBench data root
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pid, path = self.items[idx]
        image = Image.open(self.img_root / path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, pid


def _collate_orbench(batch):
    imgs = torch.stack([b[0] for b in batch])
    pids = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return imgs, pids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _embed_images(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Embed images from a DataLoader, handling two collate formats:

    - text-image: (images_tensor, texts_list, pids_tensor) → batch[1] is list
    - image-only: (images_tensor, pids_tensor, camids_tensor) → batch[1] is tensor
    """
    model.eval()
    all_embs, all_pids = [], []
    for batch in loader:
        images = batch[0]
        # Distinguish collate_text_image (texts as list) from collate_image_only (tensor)
        if isinstance(batch[1], torch.Tensor):
            pids = batch[1]   # image-only: (images, pids, camids)
        else:
            pids = batch[2]   # text-image: (images, texts, pids)
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
                     indices: np.ndarray, k_values: list[int],
                     exclude_self: bool = False) -> dict:
    """Compute Rank-K accuracy and mAP from retrieved indices.

    Args:
        exclude_self: When query == gallery set, skip the rank-1 match that
                      corresponds to each query's own index (query_idx == gallery_idx).
    """
    max_k = indices.shape[1]
    ranks_hit = {k: 0 for k in k_values}
    ap_sum = 0.0
    n = len(query_pids)

    for i, qpid in enumerate(query_pids):
        retrieved_indices = indices[i]
        if exclude_self:
            # Remove self-match (the query image itself is in the gallery)
            retrieved_indices = retrieved_indices[retrieved_indices != i]
        retrieved = gallery_pids[retrieved_indices]  # (K,)
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
            icfg_cfg = cfg["data"]["text_image"]["icfg_pedes"]
            gallery_ds = ICFGPEDESDataset(
                json_path=icfg_cfg.get("json"),  # proper split filtering
                csv_path=icfg_cfg.get("csv") if not icfg_cfg.get("json") else None,
                img_root=icfg_cfg["img_root"],
                split="test",
                transform=transform,
            )
        elif dataset_name == "orbench":
            return self._eval_orbench_text2image(model, processor, transform)
        else:
            raise ValueError(f"Unknown eval dataset: {dataset_name}")

        loader = DataLoader(
            gallery_ds, batch_size=128, shuffle=False,
            num_workers=4, collate_fn=collate_text_image,
        )

        # Embed gallery images
        gallery_embs, gallery_pids = _embed_images(model, loader, self.device)

        # Extract query texts + pids from stored metadata (avoids re-loading images).
        # When multiple captions exist per image (like RSTPReid's 2 per image),
        # expand each caption as a separate query — the standard protocol.
        if hasattr(gallery_ds, "samples") and gallery_ds.samples:
            # RSTPReidDataset / ICFGPEDESDataset: samples = [(img_path, caption_or_list, pid)]
            query_texts, query_pid_list = [], []
            for gidx, s in enumerate(gallery_ds.samples):
                caps = s[1] if isinstance(s[1], list) else [s[1]]
                for cap in caps:
                    query_texts.append(cap)
                    query_pid_list.append(gallery_pids[gidx])
            query_pids = np.array(query_pid_list, dtype=np.int64)
        else:
            # Fallback: attribute access (slow path, kept for compatibility)
            query_texts = [gallery_ds[i].text for i in range(len(gallery_ds))]
            query_pids = gallery_pids.copy()

        query_embs = _embed_texts(model, processor, query_texts, self.device)

        # FAISS search
        index = _build_faiss_index(gallery_embs, self.faiss_gpu)
        max_k = max(self.k_values)
        _, indices = index.search(query_embs.astype(np.float32), max_k)

        return _compute_metrics(query_pids, gallery_pids, indices, self.k_values)

    def _eval_orbench_text2image(self, model, processor, transform) -> dict:
        """Evaluate on ORBench RGB text→image benchmark.

        The test JSON has separate RGB_GALLERY (image paths+pids) and TEXT (descriptions+pids)
        lists, unlike RSTPReid where both come from the same sample list.
        """
        cfg = self.cfg
        orbench_cfg = cfg["data"]["text_image"]["orbench"]
        test_json = orbench_cfg.get("test_json")
        if test_json is None:
            # Derive test_json path from training json if not explicitly set
            train_json = Path(orbench_cfg["json"])
            test_json = str(train_json.parent / "test_gallery_and_queries.json")

        img_root = Path(orbench_cfg["img_root"])

        with open(test_json) as f:
            test_data = json.load(f)

        gallery_items = test_data["RGB_GALLERY"]   # [[pid, file_path], ...]
        text_items = test_data["TEXT"]              # [[pid, description], ...]

        gallery_ds = _ORBenchGalleryDataset(gallery_items, img_root, transform)
        gallery_loader = DataLoader(
            gallery_ds, batch_size=128, shuffle=False, num_workers=4,
            collate_fn=_collate_orbench,
        )

        # Embed gallery images (reuse _embed_images helper which expects (img, _, pid) tuples)
        # Use a custom embedding loop since our Dataset returns (img, pid) not (img, text, pid)
        model.eval()
        all_embs, all_pids = [], []
        with torch.no_grad():
            for imgs, pids in gallery_loader:
                imgs = imgs.to(self.device)
                emb = model.encode_image(imgs)
                all_embs.append(emb.cpu().float().numpy())
                all_pids.append(pids.numpy())
        gallery_embs = np.concatenate(all_embs)
        gallery_pids = np.concatenate(all_pids)

        # Extract text queries
        query_texts = [desc for _, desc in text_items]
        query_pids = np.array([pid for pid, _ in text_items], dtype=np.int64)

        query_embs = _embed_texts(model, processor, query_texts, self.device)

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
            exclude_self = False
        else:
            query_embs, query_pids = gallery_embs, gallery_pids
            exclude_self = True  # avoid trivial self-match

        index = _build_faiss_index(gallery_embs, self.faiss_gpu)
        max_k = max(self.k_values) + (1 if exclude_self else 0)  # +1 to account for self
        _, indices = index.search(query_embs.astype(np.float32), max_k)

        return _compute_metrics(query_pids, gallery_pids, indices, self.k_values,
                                exclude_self=exclude_self)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, model, split: str = "test") -> dict:
        metrics = {}
        primary = self.cfg["evaluation"]["primary"]
        ti_metrics = self._eval_text2image(model, primary)
        metrics.update({f"text2img/{k}": v for k, v in ti_metrics.items()})
        print(f"Text→Image ({primary}): {ti_metrics}")

        # Optional secondary text→image datasets (e.g. orbench)
        for secondary in self.cfg["evaluation"].get("secondary_text_image", []):
            sec_metrics = self._eval_text2image(model, secondary)
            metrics.update({f"text2img/{secondary}/{k}": v for k, v in sec_metrics.items()})
            print(f"Text→Image ({secondary}): {sec_metrics}")

        if self.cfg["evaluation"]["eval_image_reid"]:
            # Prefer new image_reid_datasets section; fall back to old duke config
            ii_datasets = self.cfg["evaluation"].get("image_reid_datasets")
            if ii_datasets is None:
                duke_cfg = self.cfg["data"]["image_only"]["duke"]
                ii_datasets = {"duke": {
                    "gallery_dir": duke_cfg["gallery_dir"],
                    "query_dir": duke_cfg.get("query_dir"),
                }} if duke_cfg.get("enabled") else {}

            for name, ds_cfg in ii_datasets.items():
                ii_metrics = self._eval_image2image(
                    model,
                    img_dir=ds_cfg["gallery_dir"],
                    query_dir=ds_cfg.get("query_dir"),
                )
                metrics.update({f"img2img/{name}/{k}": v for k, v in ii_metrics.items()})
                print(f"Image→Image ({name}): {ii_metrics}")

        return metrics
