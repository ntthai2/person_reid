#!/usr/bin/env python3
"""Evaluation script.

Usage:
    python scripts/eval.py --config configs/baseline.yaml --checkpoint outputs/ckpt_epoch20.pt
    python scripts/eval.py --config configs/baseline.yaml --checkpoint outputs/ckpt_epoch20.pt --dataset icfg_pedes
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.models.dual_encoder import build_model
from src.engine.evaluator import Evaluator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CLIP-based person re-ID")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    p.add_argument("--dataset", default=None,
                   help="Override primary eval dataset (e.g. rstp_reid, icfg_pedes)")
    p.add_argument("--output", default=None,
                   help="Optional JSON file to write metrics to")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.dataset:
        cfg["evaluation"]["primary"] = args.dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model", ckpt)  # handle both wrapped and bare state dicts
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}")

    evaluator = Evaluator(cfg, device)
    metrics = evaluator.evaluate(model, split="test")

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()
