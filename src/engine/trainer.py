"""
Training engine for the dual-encoder baseline.

Handles:
  - Mixed text+image + (optionally) image-only batches
  - Differential learning rates for backbone vs heads
  - Cosine LR schedule with linear warmup
  - Gradient accumulation
  - TensorBoard logging
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPProcessor

from src.datasets.text_image import build_text_image_dataset, collate_text_image
from src.datasets.image_only import build_image_only_dataset, collate_image_only
from src.models.dual_encoder import build_model
from src.losses.contrastive import InfoNCELoss, IDLoss, TripletLoss


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def _cosine_lr(optimizer, step: int, total_steps: int, warmup_steps: int,
               base_lrs: list[float]):
    """Apply cosine LR with linear warmup, in-place."""
    if step < warmup_steps:
        factor = step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * factor


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: dict, device: Optional[torch.device] = None):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tcfg = cfg["training"]

        # ------------------------------------------------------------------ #
        # Datasets & dataloaders
        # ------------------------------------------------------------------ #
        self.train_ti_ds, ti_num_classes = build_text_image_dataset(cfg, split="train")
        print(f"Text+image dataset: {len(self.train_ti_ds)} samples, "
              f"{ti_num_classes} labeled identities")
        self.train_ti_loader = DataLoader(
            self.train_ti_ds,
            batch_size=tcfg["batch_size"],
            shuffle=True,
            num_workers=tcfg["num_workers"],
            pin_memory=tcfg["pin_memory"],
            drop_last=True,
            collate_fn=collate_text_image,
            persistent_workers=tcfg["num_workers"] > 0,
        )

        # Image-only branch (Phase 3; disabled when lambda_img == 0)
        self.use_image_only = tcfg["lambda_img"] > 0
        if self.use_image_only:
            io_datasets, self.io_num_pids = build_image_only_dataset(cfg, split="train")
            self.train_io_loader = DataLoader(
                ConcatDataset(io_datasets),
                batch_size=tcfg["batch_size"],
                shuffle=True,
                num_workers=tcfg["num_workers"],
                pin_memory=tcfg["pin_memory"],
                drop_last=True,
                collate_fn=collate_image_only,
                persistent_workers=tcfg["num_workers"] > 0,
            )

        # ------------------------------------------------------------------ #
        # Model
        # ------------------------------------------------------------------ #
        self.model = build_model(cfg).to(self.device)
        self.processor = self.model.processor

        # ------------------------------------------------------------------ #
        # Losses
        # ------------------------------------------------------------------ #
        self.infonce = InfoNCELoss()

        self.id_loss = IDLoss(
            embed_dim=cfg["model"]["proj_out_dim"],
            num_classes=max(ti_num_classes, 1),  # use actual labeled identity count
        ).to(self.device)

        if self.use_image_only:
            self.triplet = TripletLoss(margin=tcfg["triplet_margin"])

        # ------------------------------------------------------------------ #
        # Optimizer
        # ------------------------------------------------------------------ #
        param_groups = self.model.param_groups(
            lr_backbone=tcfg["lr_backbone"],
            lr_heads=tcfg["lr_heads"],
        )
        # Add ID loss classifier to heads group
        param_groups[1]["params"] += list(self.id_loss.parameters())
        self.optimizer = torch.optim.AdamW(
            param_groups, weight_decay=tcfg["weight_decay"]
        )
        self._base_lrs = [g["lr"] for g in self.optimizer.param_groups]

        # ------------------------------------------------------------------ #
        # Schedule
        # ------------------------------------------------------------------ #
        steps_per_epoch = len(self.train_ti_loader) // tcfg["gradient_accumulation_steps"]
        self.total_steps = steps_per_epoch * tcfg["max_epochs"]
        self.warmup_steps = steps_per_epoch * tcfg["warmup_epochs"]

        # ------------------------------------------------------------------ #
        # Logging
        # ------------------------------------------------------------------ #
        log_dir = tcfg["log_dir"]
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.log_interval = tcfg["log_interval"]
        self.eval_interval = tcfg.get("eval_interval", 1)
        self.global_step = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tokenize(self, texts: list[str]):
        enc = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int, scaler, amp_dtype):
        self.model.train()
        self.id_loss.train()
        acfg = self.cfg["training"]
        accum = acfg["gradient_accumulation_steps"]
        w_nce = acfg["loss_weights"]["infonce"]
        w_id = acfg["loss_weights"]["id"]
        w_img = acfg["lambda_img"]

        io_iter = iter(self.train_io_loader) if self.use_image_only else None
        self.optimizer.zero_grad()
        smoke_test = self.cfg["training"].get("smoke_test", False)
        max_steps = 2 if smoke_test else None  # limit to 2 batches for sanity check

        for step_in_epoch, (images, texts, pids) in enumerate(self.train_ti_loader):
            if max_steps is not None and step_in_epoch >= max_steps:
                break
            images = images.to(self.device)
            pids = pids.to(self.device)
            input_ids, attention_mask = self._tokenize(texts)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype is not None)):
                img_emb, txt_emb, logit_scale = self.model(
                    images, input_ids, attention_mask
                )

                loss_nce = self.infonce(img_emb, txt_emb, logit_scale)
                loss_id = self.id_loss(img_emb, txt_emb, pids)
                loss = w_nce * loss_nce + w_id * loss_id

                # Image-only branch
                if self.use_image_only and io_iter is not None:
                    try:
                        io_images, io_pids, _ = next(io_iter)
                    except StopIteration:
                        io_iter = iter(self.train_io_loader)
                        io_images, io_pids, _ = next(io_iter)
                    io_images = io_images.to(self.device)
                    io_pids = io_pids.to(self.device)
                    io_emb = self.model.encode_image(io_images)
                    loss_tri = self.triplet(io_emb, io_pids)
                    loss_io_id = self.id_loss.classifier(io_emb)
                    loss_io_id = torch.nn.functional.cross_entropy(loss_io_id, io_pids)
                    loss = loss + w_img * (loss_tri + loss_io_id)

            loss_scaled = loss / accum
            if scaler:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if (step_in_epoch + 1) % accum == 0:
                self.global_step += 1
                _cosine_lr(self.optimizer, self.global_step, self.total_steps,
                           self.warmup_steps, self._base_lrs)
                if scaler:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()

                if self.global_step % self.log_interval == 0:
                    lr_bb = self.optimizer.param_groups[0]["lr"]
                    lr_hd = self.optimizer.param_groups[1]["lr"]
                    temp = (1.0 / self.model.logit_scale.exp()).item()
                    self.writer.add_scalar("train/loss_total", loss.item(), self.global_step)
                    self.writer.add_scalar("train/loss_nce", loss_nce.item(), self.global_step)
                    self.writer.add_scalar("train/loss_id", loss_id.item(), self.global_step)
                    self.writer.add_scalar("train/lr_backbone", lr_bb, self.global_step)
                    self.writer.add_scalar("train/lr_heads", lr_hd, self.global_step)
                    self.writer.add_scalar("train/temperature", temp, self.global_step)
                    print(
                        f"[E{epoch:02d} S{self.global_step:06d}] "
                        f"loss={loss.item():.4f} nce={loss_nce.item():.4f} "
                        f"id={loss_id.item():.4f} temp={temp:.4f} "
                        f"lr_bb={lr_bb:.2e}"
                    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, evaluator=None, checkpoint_dir: str = "checkpoints"):
        tcfg = self.cfg["training"]
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        amp_dtype_str = tcfg.get("amp_dtype", None)  # None | "fp16" | "bf16"
        if amp_dtype_str == "bf16" and self.device.type == "cuda":
            amp_dtype = torch.bfloat16
            scaler = None  # bf16 doesn't need GradScaler
        elif amp_dtype_str == "fp16" and self.device.type == "cuda":
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda")
        else:
            amp_dtype = None  # FP32 (full precision)
            scaler = None

        for epoch in range(1, tcfg["max_epochs"] + 1):
            t0 = time.time()
            self._train_epoch(epoch, scaler, amp_dtype)
            elapsed = time.time() - t0
            print(f"Epoch {epoch} done in {elapsed:.1f}s")

            if evaluator and epoch % self.eval_interval == 0:
                metrics = evaluator.evaluate(self.model, split="test")
                for k, v in metrics.items():
                    self.writer.add_scalar(f"eval/{k}", v, epoch)
                print(f"Eval @ epoch {epoch}: {metrics}")

            # Save checkpoint every epoch
            ckpt = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "id_loss": self.id_loss.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
            }
            torch.save(ckpt, Path(checkpoint_dir) / f"ckpt_epoch{epoch:02d}.pt")

        self.writer.close()
