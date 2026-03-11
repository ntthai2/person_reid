"""
Dual-encoder model: CLIP ViT-B/16 with MLP projection heads.

Both image and text towers produce L2-normalised 512-dim embeddings.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class MLP(nn.Module):
    """Linear → ReLU → Linear projection head."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualEncoder(nn.Module):
    """CLIP-based dual encoder for cross-modal person retrieval.

    Args:
        backbone: HuggingFace model identifier (default: openai/clip-vit-base-patch16).
        embed_dim: CLIP native embedding dimension (512 for ViT-B/16).
        proj_hidden_dim: Hidden dim of projection MLP.
        proj_out_dim: Output dim of projection MLP.
        init_temperature: Initial value for 1/temperature (logit scale).
    """

    def __init__(
        self,
        backbone: str = "openai/clip-vit-base-patch16",
        embed_dim: int = 512,
        proj_hidden_dim: int = 512,
        proj_out_dim: int = 512,
        init_temperature: float = 0.07,
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(backbone)
        self.processor = CLIPProcessor.from_pretrained(backbone)

        # Projection heads (one per modality)
        self.image_proj = MLP(embed_dim, proj_hidden_dim, proj_out_dim)
        self.text_proj = MLP(embed_dim, proj_hidden_dim, proj_out_dim)

        # Learnable log-temperature; match CLIP convention (logit_scale = log(1/t))
        init_logit_scale = math.log(1.0 / init_temperature)
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensor(out) -> torch.Tensor:
        """Unwrap a possibly-wrapped model output to a plain tensor."""
        if isinstance(out, torch.Tensor):
            return out
        # transformers BaseModelOutputWithPooling / CLIPOutput, etc.
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0]  # CLS token
        raise TypeError(f"Cannot extract tensor from {type(out)}")

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised image embeddings."""
        raw = self.clip.vision_model(pixel_values=pixel_values)
        # pooler_output is post-layernorm CLS: shape (B, hidden_dim)
        pool = self._to_tensor(raw)
        # Apply CLIP's visual projection to get (B, embed_dim=512)
        feats = self.clip.visual_projection(pool)
        feats = self.image_proj(feats)
        return nn.functional.normalize(feats, dim=-1)

    def encode_text(self, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised text embeddings."""
        raw = self.clip.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pool = self._to_tensor(raw)
        # Apply CLIP's text projection to get (B, embed_dim=512)
        feats = self.clip.text_projection(pool)
        feats = self.text_proj(feats)
        return nn.functional.normalize(feats, dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """
        Returns:
            image_embeds: (B, D) normalised
            text_embeds:  (B, D) normalised
            logit_scale:  scalar (clamped to [log(1/100), log(100)])
        """
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)
        # Clamp temperature from wandering to degenerate values
        logit_scale = self.logit_scale.clamp(-math.log(100), math.log(100))
        return image_embeds, text_embeds, logit_scale

    # ------------------------------------------------------------------
    # Parameter groups for differential LR
    # ------------------------------------------------------------------

    def param_groups(self, lr_backbone: float, lr_heads: float):
        backbone_params = list(self.clip.parameters())
        head_params = (
            list(self.image_proj.parameters())
            + list(self.text_proj.parameters())
            + [self.logit_scale]
        )
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_heads},
        ]


def build_model(cfg) -> DualEncoder:
    mcfg = cfg["model"]
    return DualEncoder(
        backbone=mcfg["backbone"],
        embed_dim=mcfg["embed_dim"],
        proj_hidden_dim=mcfg["proj_hidden_dim"],
        proj_out_dim=mcfg["proj_out_dim"],
        init_temperature=mcfg["init_temperature"],
    )
