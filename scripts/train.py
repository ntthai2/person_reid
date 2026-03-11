#!/usr/bin/env python3
"""Train script.

Usage:
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/baseline.yaml --smoke-test
    python scripts/train.py --config configs/baseline.yaml --resume outputs/ckpt_epoch01.pt
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
                   help="Resume from checkpoint path")
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

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        trainer.model.load_state_dict(ckpt["model"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Resumed from {args.resume}")

    evaluator = None if args.no_eval else Evaluator(cfg, device)

    trainer.train(
        evaluator=evaluator,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
