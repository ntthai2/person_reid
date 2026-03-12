#!/usr/bin/env python3
"""Train script.

Usage:
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/baseline.yaml --smoke-test
    python scripts/train.py --config configs/baseline.yaml --resume outputs/ckpt_epoch01.pt
    python scripts/train.py --config configs/local_align.yaml \\
        --init-from outputs/baseline_run1/ckpt_epoch16.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Make repo root importable when running from any cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CLIP-based person re-ID")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run only 2 batches per epoch for sanity-checking")
    p.add_argument("--resume", default=None,
                   help="Resume from checkpoint path (strict, optimizer state included)")
    p.add_argument("--init-from", default=None,
                   help="Warm-start model weights from a previous checkpoint (non-strict, "
                        "for cross-phase initialisation e.g. Phase1→Phase2)")
    p.add_argument("--checkpoint-dir", default="outputs",
                   help="Directory to save checkpoints (default: outputs/)")
    p.add_argument("--no-eval", action="store_true",
                   help="Skip evaluation after each epoch")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.smoke_test:
        # Minimal epochs to verify pipeline is wired correctly
        cfg["training"]["max_epochs"] = 1
        cfg["training"]["smoke_test"] = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainer = Trainer(cfg, device)

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        trainer.model.load_state_dict(ckpt["model"])
        if "id_loss" in ckpt:
            trainer.id_loss.load_state_dict(ckpt["id_loss"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        if "global_step" in ckpt:
            trainer.global_step = ckpt["global_step"]
        start_epoch = ckpt.get("epoch", 0) + 1  # resume from next epoch
        print(f"Resumed from {args.resume} (epoch {start_epoch - 1} done, continuing from {start_epoch})")
    elif args.init_from:
        # Non-strict warm-start: load matching weights, skip new params
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt)
        missing, unexpected = trainer.model.load_state_dict(state, strict=False)
        print(f"Warm-started from {args.init_from}")
        if missing:
            print(f"  New params (randomly initialised): {missing[:10]}{'...' if len(missing)>10 else ''}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected[:5]}")
        # Reset temperature to config value so cross-phase warm-start doesn't
        # inherit a collapsed temperature from the previous phase.
        init_temp = cfg["model"].get("init_temperature", 0.07)
        with torch.no_grad():
            trainer.model.logit_scale.fill_(float(__import__("math").log(1.0 / init_temp)))

    evaluator = None if args.no_eval else Evaluator(cfg, device)

    trainer.train(
        evaluator=evaluator,
        checkpoint_dir=args.checkpoint_dir,
        start_epoch=start_epoch,
    )


if __name__ == "__main__":
    main()
