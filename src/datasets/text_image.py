"""
Unified text+image dataset for cross-modal person retrieval.

Supports:
  - CUHK-PEDES  (HuggingFace Parquet; no identity labels)
  - ICFG-PEDES  (cleaned CSV; identity labels present)
  - RSTPReid    (JSON; identity labels; 5 captions/image)
  - IIITD-20K   (Filtered.json; no identity labels; 2 descriptions/image)
  - ORBench     (train_annos.json; RGB vis images; identity labels)
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Shared transforms
# ---------------------------------------------------------------------------

def build_train_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])


def build_val_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class Sample:
    """Lightweight named tuple for a single text-image sample."""
    __slots__ = ("image", "text", "pid")

    def __init__(self, image: torch.Tensor, text: str, pid: int):
        self.image = image
        self.text = text
        self.pid = pid  # -1 when identity label is unavailable


# ---------------------------------------------------------------------------
# CUHK-PEDES — HuggingFace Parquet
# ---------------------------------------------------------------------------

class CUHKPEDESDataset(Dataset):
    """Loads CUHK-PEDES from HuggingFace-format Parquet shards.

    On first run, converts shards to an Arrow cache at `{parquet_dir}/.arrow_cache/`
    (via `datasets.Dataset.save_to_disk`). Subsequent runs load the cache with
    `load_from_disk`, which is memory-mapped and fork-safe for DataLoader workers.
    No identity labels; pid is always -1.
    """

    def __init__(self, parquet_dir: str, split: str = "train",
                 transform=None):
        import glob
        from pathlib import Path as _Path
        from datasets import Dataset as HFDataset, load_from_disk

        cache_dir = str(_Path(parquet_dir) / ".arrow_cache")
        if _Path(cache_dir).exists():
            self._ds = load_from_disk(cache_dir)
        else:
            parquet_files = sorted(glob.glob(f"{parquet_dir}/*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No .parquet files found in {parquet_dir}")
            print(f"Building Arrow cache at {cache_dir} ...")
            ds = HFDataset.from_parquet(parquet_files)
            ds.save_to_disk(cache_dir)
            self._ds = load_from_disk(cache_dir)

        self._transform = transform
        self._img_col = "image" if "image" in self._ds.column_names else "image_bytes"
        self._txt_col = next(
            (c for c in ("text", "caption", "captions") if c in self._ds.column_names), None
        )
        if self._txt_col is None:
            raise ValueError(f"No caption column found. Columns: {self._ds.column_names}")

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx: int):
        import io
        row = self._ds[idx]
        img_data = row[self._img_col]
        text = str(row[self._txt_col]).strip()

        if isinstance(img_data, Image.Image):
            image = img_data.convert("RGB")
        elif isinstance(img_data, dict):
            raw = img_data.get("bytes") or img_data.get("path")
            image = Image.open(io.BytesIO(raw)).convert("RGB") if isinstance(raw, bytes) \
                else Image.open(raw).convert("RGB")
        elif isinstance(img_data, bytes):
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
        elif isinstance(img_data, np.ndarray):
            image = Image.fromarray(img_data).convert("RGB")
        else:
            image = img_data.convert("RGB")

        if self._transform:
            image = self._transform(image)
        return Sample(image, text, pid=-1)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx: int):
        import io
        row = self._ds[idx]
        img_data = row[self._img_col]
        text = str(row[self._txt_col]).strip()

        if isinstance(img_data, Image.Image):
            image = img_data.convert("RGB")
        elif isinstance(img_data, dict):
            raw = img_data.get("bytes") or img_data.get("path")
            image = Image.open(io.BytesIO(raw)).convert("RGB") if isinstance(raw, bytes) \
                else Image.open(raw).convert("RGB")
        elif isinstance(img_data, bytes):
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
        elif isinstance(img_data, np.ndarray):
            image = Image.fromarray(img_data).convert("RGB")
        else:
            image = img_data.convert("RGB")

        if self._transform:
            image = self._transform(image)
        return Sample(image, text, pid=-1)


# ---------------------------------------------------------------------------
# ICFG-PEDES — cleaned CSV
# ---------------------------------------------------------------------------

class ICFGPEDESDataset(Dataset):
    """Loads ICFG-PEDES from captions_cleaned.csv.

    CSV columns: image, caption, id
    The 'image' column contains Windows absolute paths; we strip everything
    before 'imgs/' and resolve it relative to img_root.
    """

    _WIN_PREFIX_RE = re.compile(r".*[/\\]imgs[/\\]", re.IGNORECASE)

    def __init__(self, csv_path: str, img_root: str,
                 split: Optional[str] = "train", transform=None):
        df = pd.read_csv(csv_path, dtype=str)
        df["_rel"] = df["image"].apply(self._strip_prefix)
        df["_pid"] = df["id"].astype(int)
        self.df = df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.transform = transform

    @classmethod
    def _strip_prefix(cls, path: str) -> str:
        m = cls._WIN_PREFIX_RE.search(path)
        if m:
            rel = path[m.end():]
        else:
            rel = Path(path).name
        # Convert any remaining Windows backslashes to forward slashes
        return rel.replace("\\", "/")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(self.img_root / row["_rel"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return Sample(image, row["caption"], row["_pid"])


# ---------------------------------------------------------------------------
# RSTPReid
# ---------------------------------------------------------------------------

class RSTPReidDataset(Dataset):
    """Loads RSTPReid from data_captions.json.

    JSON schema: list of {id, img_path, captions: [str]*5, split}
    Each image generates 5 separate training samples. During training, one
    caption is sampled randomly at iteration time for variety.
    """

    def __init__(self, json_path: str, img_root: str,
                 split: str = "train", transform=None, deterministic: bool = False):
        with open(json_path) as f:
            data = json.load(f)
        self.samples = [
            (item["img_path"], item["captions"], item["id"])
            for item in data
            if item.get("split", "train") == split
        ]
        self.img_root = Path(img_root)
        self.transform = transform
        self.deterministic = deterministic

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, captions, pid = self.samples[idx]
        image = Image.open(self.img_root / img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.deterministic:
            text = captions[0]
        else:
            text = random.choice(captions)
        return Sample(image, text, pid)


# ---------------------------------------------------------------------------
# IIITD-20K
# ---------------------------------------------------------------------------

class IIITD20KDataset(Dataset):
    """Loads IIITD-20K from Filtered.json.

    JSON schema: {"idx_str": {Image_ID: str, Description_1: str, Description_2: str}, ...}
    No split or identity labels. Each image generates 2 samples.
    Images are under img_root/{Image_ID}.jpeg (or .jpg if jpeg not found).
    """

    def __init__(self, json_path: str, img_root: str, transform=None):
        with open(json_path) as f:
            raw = json.load(f)
        self.samples: list[tuple[str, str]] = []  # (image_id, description)
        for entry in raw.values():
            img_id = entry["Image ID"]
            for key in ("Description 1", "Description 2"):
                desc = entry.get(key, "").strip()
                if desc:
                    self.samples.append((img_id, desc))
        self.img_root = Path(img_root)
        self.transform = transform

    def _resolve_path(self, img_id: str) -> Path:
        for ext in (".jpeg", ".jpg", ".png"):
            p = self.img_root / f"{img_id}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"Image not found for ID {img_id!r} in {self.img_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_id, text = self.samples[idx]
        image = Image.open(self._resolve_path(img_id)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return Sample(image, text, pid=-1)


# ---------------------------------------------------------------------------
# ORBench
# ---------------------------------------------------------------------------

class ORBenchDataset(Dataset):
    """Loads ORBench RGB ('_vis.jpg') images from train_annos.json.

    JSON schema: list of {id, file_path, caption, split}
    file_path is relative to img_root (e.g. "vis/0001/0001_llcm_…_vis.jpg").
    """

    def __init__(self, json_path: str, img_root: str,
                 split: str = "train", transform=None):
        with open(json_path) as f:
            data = json.load(f)
        self.samples = [
            (item["file_path"], item["caption"], item["id"])
            for item in data
            if item.get("split", "train") == split
        ]
        self.img_root = Path(img_root)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_path, caption, pid = self.samples[idx]
        image = Image.open(self.img_root / rel_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return Sample(image, caption, pid)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_text_image(batch: list):
    images = torch.stack([s.image for s in batch])
    texts = [s.text for s in batch]
    pids = torch.tensor([s.pid for s in batch], dtype=torch.long)
    return images, texts, pids


# ---------------------------------------------------------------------------
# Pid remapping helpers
# ---------------------------------------------------------------------------

def _remap_dataset_pids(dataset, offset: int = 0) -> int:
    """Remap a dataset's labeled pids to contiguous [offset, offset+N) indices.

    Returns N (number of unique labeled identities in this dataset).
    Mutates the dataset in place.  Datasets with all pid == -1 return 0.
    """
    # Collect all unique valid pids
    if isinstance(dataset, ICFGPEDESDataset):
        raw_pids = dataset.df["_pid"].values
    elif isinstance(dataset, (RSTPReidDataset, ORBenchDataset)):
        raw_pids = [s[2] for s in dataset.samples]
    else:
        return 0  # CUHK-PEDES, IIITD-20K — all pid == -1

    unique = sorted({p for p in raw_pids if p >= 0})
    if not unique:
        return 0
    pid_map = {raw: offset + i for i, raw in enumerate(unique)}

    # Apply remapping in place
    if isinstance(dataset, ICFGPEDESDataset):
        dataset.df["_pid"] = dataset.df["_pid"].map(
            lambda p: pid_map.get(p, -1)
        )
    else:
        dataset.samples = [
            (path, cap, pid_map.get(pid, -1))
            for path, cap, pid in dataset.samples
        ]

    return len(unique)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_text_image_dataset(cfg, split: str = "train"):
    """Build a ConcatDataset from all enabled text+image sources.

    Returns:
        (ConcatDataset, total_num_classes)
        total_num_classes: total number of labeled identities across all
            datasets (after remapping to contiguous [0, N) indices).
    """
    is_train = split == "train"
    transform = build_train_transform(cfg["data"]["image_size"]) if is_train \
        else build_val_transform(cfg["data"]["image_size"])

    datasets: list[Dataset] = []
    tcfg = cfg["data"]["text_image"]

    if tcfg["cuhk_pedes"]["enabled"]:
        datasets.append(CUHKPEDESDataset(
            parquet_dir=tcfg["cuhk_pedes"]["parquet_dir"],
            split="train",   # parquet only has train
            transform=transform,
        ))

    if tcfg["icfg_pedes"]["enabled"]:
        datasets.append(ICFGPEDESDataset(
            csv_path=tcfg["icfg_pedes"]["csv"],
            img_root=tcfg["icfg_pedes"]["img_root"],
            transform=transform,
        ))

    if tcfg["rstp_reid"]["enabled"]:
        datasets.append(RSTPReidDataset(
            json_path=tcfg["rstp_reid"]["json"],
            img_root=tcfg["rstp_reid"]["img_root"],
            split=split,
            transform=transform,
            deterministic=not is_train,
        ))

    if tcfg["iiitd_20k"]["enabled"]:
        datasets.append(IIITD20KDataset(
            json_path=tcfg["iiitd_20k"]["json"],
            img_root=tcfg["iiitd_20k"]["img_root"],
            transform=transform,
        ))

    if tcfg["orbench"]["enabled"]:
        datasets.append(ORBenchDataset(
            json_path=tcfg["orbench"]["json"],
            img_root=tcfg["orbench"]["img_root"],
            split=split,
            transform=transform,
        ))

    if not datasets:
        raise ValueError("No text+image datasets enabled in config.")

    # Remap each dataset's pids to a non-overlapping contiguous range
    total_classes = 0
    for d in datasets:
        n = _remap_dataset_pids(d, offset=total_classes)
        total_classes += n

    combined = ConcatDataset(datasets)
    combined.num_labeled_classes = total_classes  # expose for Trainer
    return combined, total_classes

