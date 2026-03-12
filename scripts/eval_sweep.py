#!/usr/bin/env python3
"""Sweep evaluation across all checkpoints in a directory.

Usage:
    python scripts/eval_sweep.py --config configs/baseline.yaml \
        --checkpoint-dir outputs/baseline_run1 \
        --output runs/sweep_results.json
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--dataset", default=None, help="Override primary eval dataset")
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset:
        cfg["evaluation"]["primary"] = args.dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_dir = Path(args.checkpoint_dir)
    checkpoints = sorted(ckpt_dir.glob("ckpt_epoch*.pt"))
    print(f"Found {len(checkpoints)} checkpoints")

    evaluator = Evaluator(cfg, device)
    results = {}

    for ckpt_path in checkpoints:
        epoch = int(ckpt_path.stem.split("epoch")[1])
        print(f"\n--- Epoch {epoch:02d} ({ckpt_path.name}) ---")
        model = build_model(cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)

        metrics = evaluator.evaluate(model, split="test")
        results[f"epoch_{epoch:02d}"] = metrics

        r1 = metrics.get("text2img/R1", 0.0)
        r5 = metrics.get("text2img/R5", 0.0)
        r10 = metrics.get("text2img/R10", 0.0)
        mAP = metrics.get("text2img/mAP", 0.0)
        print(f"  R@1={r1:.2f}  R@5={r5:.2f}  R@10={r10:.2f}  mAP={mAP:.2f}")

        del model
        torch.cuda.empty_cache()

    # Summary
    primary = cfg["evaluation"]["primary"]
    print(f"\n=== Sweep Summary ({primary}) ===")
    print(f"{'Epoch':>6}  {'R@1':>6}  {'R@5':>6}  {'R@10':>7}  {'mAP':>6}")
    best_epoch, best_r1 = None, -1.0
    for key, m in results.items():
        ep = int(key.split("_")[1])
        r1 = m.get("text2img/R1", 0.0)
        r5 = m.get("text2img/R5", 0.0)
        r10 = m.get("text2img/R10", 0.0)
        mAP = m.get("text2img/mAP", 0.0)
        marker = " <-- best" if r1 > best_r1 else ""
        if r1 > best_r1:
            best_r1, best_epoch = r1, ep
        print(f"  E{ep:02d}   {r1:6.2f}  {r5:6.2f}  {r10:7.2f}  {mAP:6.2f}{marker}")

    print(f"\nBest: Epoch {best_epoch:02d}  R@1={best_r1:.2f}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"results": results, "best_epoch": best_epoch, "best_r1": best_r1}, f, indent=2)
        print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
