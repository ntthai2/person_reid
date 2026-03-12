"""
Dual-encoder model: CLIP ViT-B/16 with MLP projection heads.

Both image and text towers produce L2-normalised 512-dim embeddings.

Phase 2 extension: when use_local_align=True, the model also exposes
patch-level image tokens and word-level text tokens for the LocalAlignModule.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

from src.models.local_align import LocalAlignModule


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
        min_temperature: float = 0.04,  # floor to prevent temperature collapse
        use_local_align: bool = False,
        local_align_layers: int = 2,
        local_align_heads: int = 8,
        local_align_dropout: float = 0.1,
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
        # max logit_scale = log(1/min_temp); beyond this, NCE gradient dies
        self._max_logit_scale = math.log(1.0 / max(min_temperature, 1e-6))

        # Phase 2: local alignment module (optional)
        self.use_local_align = use_local_align
        if use_local_align:
            img_hidden = self.clip.config.vision_config.hidden_size    # 768 ViT-B
            txt_hidden = self.clip.config.text_config.hidden_size       # 512
            self.local_align = LocalAlignModule(
                img_hidden=img_hidden,
                txt_hidden=txt_hidden,
                d_model=proj_out_dim,
                n_heads=local_align_heads,
                n_layers=local_align_layers,
                vocab_size=self.clip.config.text_config.vocab_size,
                dropout=local_align_dropout,
            )
        else:
            self.local_align = None

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
        """Return L2-normalised image embeddings (global only)."""
        raw = self.clip.vision_model(pixel_values=pixel_values)
        # pooler_output is post-layernorm CLS: shape (B, hidden_dim)
        pool = self._to_tensor(raw)
        # Apply CLIP's visual projection to get (B, embed_dim=512)
        feats = self.clip.visual_projection(pool)
        feats = self.image_proj(feats)
        return nn.functional.normalize(feats, dim=-1)

    def encode_image_with_patches(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (global_emb, patch_tokens) for local alignment.

        Returns:
            global_emb:    (B, proj_out_dim) L2-normalised
            patch_tokens:  (B, N_patches+1, img_hidden) — includes CLS at [0]
        """
        raw = self.clip.vision_model(pixel_values=pixel_values)
        patch_tokens = raw.last_hidden_state            # (B, 197, 768)
        pool = self._to_tensor(raw)
        feats = self.clip.visual_projection(pool)
        feats = self.image_proj(feats)
        return nn.functional.normalize(feats, dim=-1), patch_tokens

    def encode_text(self, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised text embeddings (global only).

        Note: At evaluation time we ALWAYS use global embeddings, even in
        Phase 2.  Cross-attention local alignment is computed during training
        (forward_local) with a paired image batch, but at retrieval time text
        and image galleries are encoded independently.  The MLM auxiliary task
        improves the shared backbone, so global embeddings benefit indirectly.
        """
        raw = self.clip.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pool = self._to_tensor(raw)
        # Apply CLIP's text projection to get (B, embed_dim=512)
        feats = self.clip.text_projection(pool)
        feats = self.text_proj(feats)
        return nn.functional.normalize(feats, dim=-1)

    def encode_text_with_tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (global_emb, token_hiddens) for local alignment.

        Returns:
            global_emb:     (B, proj_out_dim) L2-normalised
            token_hiddens:  (B, N_txt, txt_hidden) per-token hidden states
        """
        raw = self.clip.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        token_hiddens = raw.last_hidden_state           # (B, 77, 512)
        pool = self._to_tensor(raw)
        feats = self.clip.text_projection(pool)
        feats = self.text_proj(feats)
        return nn.functional.normalize(feats, dim=-1), token_hiddens

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """Global-only forward (Phase 1 compatible).

        Returns:
            image_embeds: (B, D) normalised
            text_embeds:  (B, D) normalised
            logit_scale:  scalar (clamped)
        """
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)
        logit_scale = self.logit_scale.clamp(-math.log(100), self._max_logit_scale)
        return image_embeds, text_embeds, logit_scale

    def forward_local(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mlm_mask: Optional[torch.Tensor] = None,  # (B, N_txt) bool
    ):
        """Phase 2 forward: global + local alignment + optional MLM.

        Returns:
            image_embeds:    (B, D) fused global+local (or global if no local_align)
            text_embeds:     (B, D) fused global+local
            logit_scale:     scalar
            mlm_logits:      (B, N_txt, vocab) or None
        """
        assert self.local_align is not None, "use_local_align must be True"

        # Encode with full token/patch sequences
        global_img, patch_tokens = self.encode_image_with_patches(pixel_values)
        global_txt, token_hiddens = self.encode_text_with_tokens(input_ids, attention_mask)

        # Cross-modal local alignment (text→image attention)
        local_txt_emb, fused_hiddens = self.local_align(
            img_patch_tokens=patch_tokens[:, 1:],  # exclude CLS patch
            txt_token_hiddens=token_hiddens,
            txt_attention_mask=attention_mask,
        )

        # Fuse global + local embeddings
        image_embeds = global_img   # image uses global only (text→image only)
        text_embeds = self.local_align.fuse(global_txt, local_txt_emb)

        logit_scale = self.logit_scale.clamp(-math.log(100), self._max_logit_scale)
        mlm_logits = None
        if mlm_mask is not None:
            mlm_logits = self.local_align.mlm_predictions(fused_hiddens)

        return image_embeds, text_embeds, logit_scale, mlm_logits

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
        if self.local_align is not None:
            head_params += list(self.local_align.parameters())
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_heads},
        ]


def build_model(cfg) -> DualEncoder:
    mcfg = cfg["model"]
    la_cfg = mcfg.get("local_align", {})
    return DualEncoder(
        backbone=mcfg["backbone"],
        embed_dim=mcfg["embed_dim"],
        proj_hidden_dim=mcfg["proj_hidden_dim"],
        proj_out_dim=mcfg["proj_out_dim"],
        init_temperature=mcfg["init_temperature"],
        min_temperature=mcfg.get("min_temperature", 0.04),
        use_local_align=la_cfg.get("enabled", False),
        local_align_layers=la_cfg.get("n_layers", 2),
        local_align_heads=la_cfg.get("n_heads", 8),
        local_align_dropout=la_cfg.get("dropout", 0.1),
    )
