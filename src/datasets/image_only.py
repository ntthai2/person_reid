"""
Image-only ReID dataset loaders.

Supports:
  - DukeMTMC-reID  (folder-based; {pid}_{camid}_*.jpg naming)
  - Market-1203    (Market1203 subfolder; same naming convention)
  - LaST           (identity subfolders under train/ and val/)
  - CAVIARa        (flat folder; {pid}_{frame}.* naming)
  - GRID           (underground_reid; .mat split file)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import scipy.io
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.datasets.text_image import build_train_transform, build_val_transform


# ---------------------------------------------------------------------------
# Shared image sample
# ---------------------------------------------------------------------------

class ImageSample:
    __slots__ = ("image", "pid", "camid")

    def __init__(self, image: torch.Tensor, pid: int, camid: int = -1):
        self.image = image
        self.pid = pid
        self.camid = camid


def collate_image_only(batch: list):
    images = torch.stack([s.image for s in batch])
    pids = torch.tensor([s.pid for s in batch], dtype=torch.long)
    camids = torch.tensor([s.camid for s in batch], dtype=torch.long)
    return images, pids, camids


# ---------------------------------------------------------------------------
# DukeMTMC-reID / Market-1203  (shared naming convention)
# ---------------------------------------------------------------------------
# File naming: {pid:04d}_{camid:02d}_{seq:06d}_{frame:06d}.jpg
#   Duke:    0001_c1_f0044158.jpg  (camid after 'c', no leading zeros)
#   Market:  0001_c1s1_000001_00.jpg
# We parse pid from the first numeric segment before '_'.

_PID_RE = re.compile(r"^(\d+)_")


def _parse_pid_camid_duke(fname: str):
    """Parse pid and camid from DukeMTMC-reID filename."""
    # e.g. 0001_c1_f0044158.jpg
    parts = Path(fname).stem.split("_")
    pid = int(parts[0])
    camid = int(parts[1][1:]) if len(parts) > 1 else -1
    return pid, camid


def _parse_pid_camid_market(fname: str):
    """Parse pid and camid from Market-1501/1203 filename."""
    # e.g. 0001_c1s1_000001_00.jpg
    parts = Path(fname).stem.split("_")
    pid = int(parts[0])
    camid = int(parts[1][1]) if len(parts) > 1 else -1  # 'c1s1' → 1
    return pid, camid


class FolderReIDDataset(Dataset):
    """Generic ReID loader for flat folder with pid-encoded filenames.

    Args:
        img_dir: Directory containing images.
        parser: Callable(filename str) -> (pid, camid).
        pid_offset: Add this to all pids (for mixing multiple datasets).
        transform: Image transform.
    """

    def __init__(self, img_dir: str, parser=_parse_pid_camid_duke,
                 pid_offset: int = 0, transform=None):
        self.paths: list[Path] = sorted(
            p for p in Path(img_dir).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        self.parser = parser
        self.pid_offset = pid_offset
        self.transform = transform

        # Build contiguous pid mapping
        raw_pids = sorted({self.parser(p.name)[0] for p in self.paths})
        self.pid_map = {raw: i + pid_offset for i, raw in enumerate(raw_pids)}
        self.num_pids = len(raw_pids)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        raw_pid, camid = self.parser(p.name)
        pid = self.pid_map[raw_pid]
        image = Image.open(p).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return ImageSample(image, pid, camid)


class DukeMTMCDataset(FolderReIDDataset):
    def __init__(self, train_dir: str, pid_offset: int = 0, transform=None):
        super().__init__(train_dir, _parse_pid_camid_duke, pid_offset, transform)


class MarketDataset(FolderReIDDataset):
    def __init__(self, train_dir: str, pid_offset: int = 0, transform=None):
        # Market1203 folder may have direct images or subdirectories
        img_dir = Path(train_dir)
        # Flatten: if it has subdirectories, gather from those
        has_subdirs = any(d.is_dir() for d in img_dir.iterdir())
        if has_subdirs:
            self.paths = sorted(
                p for subdir in img_dir.iterdir() if subdir.is_dir()
                for p in subdir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
        else:
            self.paths = sorted(
                p for p in img_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
        self.parser = _parse_pid_camid_market
        self.pid_offset = pid_offset
        self.transform = transform
        raw_pids = sorted({self.parser(p.name)[0] for p in self.paths})
        self.pid_map = {raw: i + pid_offset for i, raw in enumerate(raw_pids)}
        self.num_pids = len(raw_pids)


# ---------------------------------------------------------------------------
# LaST  (identity subfolders)
# ---------------------------------------------------------------------------

class LaSTDataset(Dataset):
    """LaST dataset: identity subfolders under train/ or val/.

    Structure: {split}/{pid:06d}/{image}.jpg
    """

    def __init__(self, split_dir: str, pid_offset: int = 0, transform=None):
        self.samples: list[tuple[Path, int]] = []
        root = Path(split_dir)
        pid_dirs = sorted(d for d in root.iterdir() if d.is_dir())
        pid_map = {d.name: i + pid_offset for i, d in enumerate(pid_dirs)}
        self.num_pids = len(pid_dirs)
        for pid_dir in pid_dirs:
            pid = pid_map[pid_dir.name]
            for img in sorted(pid_dir.iterdir()):
                if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img, pid))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, pid = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return ImageSample(image, pid)


# ---------------------------------------------------------------------------
# CAVIARa  (flat folder, e.g. personXXX_frameYYY.png)
# ---------------------------------------------------------------------------

_CAVIAR_PID_RE = re.compile(r"person(\d+)", re.IGNORECASE)


def _parse_caviar_pid(fname: str) -> int:
    m = _CAVIAR_PID_RE.search(fname)
    return int(m.group(1)) if m else -1


class CAVIARaDataset(Dataset):
    """CAVIARa dataset: flat folder, filename encodes person id."""

    def __init__(self, img_dir: str, pid_offset: int = 0, transform=None):
        paths = sorted(
            p for p in Path(img_dir).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        raw_pids = sorted({_parse_caviar_pid(p.name) for p in paths})
        pid_map = {raw: i + pid_offset for i, raw in enumerate(raw_pids)}
        self.samples = [(p, pid_map[_parse_caviar_pid(p.name)]) for p in paths]
        self.num_pids = len(raw_pids)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, pid = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return ImageSample(image, pid)


# ---------------------------------------------------------------------------
# GRID / underground_reid  (MATLAB .mat partitions)
# ---------------------------------------------------------------------------

class GRIDDataset(Dataset):
    """GRID underground ReID dataset.

    The .mat file contains precomputed features and partition indices.
    We use the raw image files from probe/ and gallery/ directories
    instead of the precomputed features, loading split 0 by default.
    Filenames: {pid:04d}_45.bmp (probe), {pid:04d}_{cam:02d}.bmp (gallery).
    """

    def __init__(self, probe_dir: str, gallery_dir: str,
                 mat_file: Optional[str] = None,
                 split_idx: int = 0, mode: str = "probe",
                 pid_offset: int = 0, transform=None):
        assert mode in {"probe", "gallery"}
        src_dir = probe_dir if mode == "probe" else gallery_dir
        paths = sorted(
            p for p in Path(src_dir).iterdir()
            if p.suffix.lower() in {".bmp", ".jpg", ".jpeg", ".png"}
        )
        # Parse pid from first numeric portion of filename
        def _pid_from_name(fname):
            m = re.match(r"(\d+)", fname)
            return int(m.group(1)) if m else -1

        raw_pids = sorted({_pid_from_name(p.name) for p in paths})
        pid_map = {raw: i + pid_offset for i, raw in enumerate(raw_pids)}
        self.samples = [(p, pid_map[_pid_from_name(p.name)]) for p in paths]
        self.num_pids = len(raw_pids)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, pid = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return ImageSample(image, pid)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_image_only_dataset(cfg, split: str = "train"):
    """Build a list of enabled image-only datasets with cumulative pid offsets."""
    is_train = split == "train"
    icfg = cfg["data"]["image_only"]
    image_size = cfg["model"]["image_size"]
    transform = build_train_transform(image_size) if is_train \
        else build_val_transform(image_size)

    datasets: list[Dataset] = []
    offset = 0

    if icfg["duke"]["enabled"]:
        ds = DukeMTMCDataset(icfg["duke"]["train_dir"], pid_offset=offset,
                             transform=transform)
        datasets.append(ds)
        offset += ds.num_pids

    if icfg["market"]["enabled"]:
        ds = MarketDataset(icfg["market"]["train_dir"], pid_offset=offset,
                           transform=transform)
        datasets.append(ds)
        offset += ds.num_pids

    if icfg["last"]["enabled"]:
        ds = LaSTDataset(icfg["last"]["train_dir"], pid_offset=offset,
                         transform=transform)
        datasets.append(ds)
        offset += ds.num_pids

    if icfg["caviar"]["enabled"]:
        ds = CAVIARaDataset(icfg["caviar"]["img_dir"], pid_offset=offset,
                             transform=transform)
        datasets.append(ds)
        offset += ds.num_pids

    if icfg["grid"]["enabled"]:
        ds = GRIDDataset(icfg["grid"]["probe_dir"], icfg["grid"]["gallery_dir"],
                         mat_file=icfg["grid"].get("mat_file"),
                         mode="probe", pid_offset=offset, transform=transform)
        datasets.append(ds)
        offset += ds.num_pids

    return datasets, offset  # datasets list + total pid count
