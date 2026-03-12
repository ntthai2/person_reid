"""Local alignment module for Phase 2 — cross-modal attribute-region grounding.

Architecture:
  - CLIP image patch tokens (B, 197, 768) are projected to d_model
  - CLIP text token hidden states (B, 77, 512) are projected to d_model
  - N layers of cross-attention: text tokens attend to image patches
  - Pooled local text embedding fused with global embedding (learnable α)
  - Optional MLM head: Linear(d_model, vocab_size) predicts masked tokens

Reference: IRRA (Jiang et al., CVPR 2023) — implicit relation reasoning.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CrossAttnLayer(nn.Module):
    """Single cross-attention + feed-forward layer.

    Query comes from *query_seq*; keys and values come from *kv_seq*.
    Residual connection + LayerNorm around both sub-layers.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,          # (B, Nq, d_model)
        kv: torch.Tensor,             # (B, Nkv, d_model)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, Nkv) bool; True = ignore
    ) -> torch.Tensor:
        attn_out, _ = self.attn(query, kv, kv, key_padding_mask=key_padding_mask)
        x = self.norm1(query + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ---------------------------------------------------------------------------
# Local alignment module
# ---------------------------------------------------------------------------

class LocalAlignModule(nn.Module):
    """Cross-modal local alignment + optional MLM head.

    Args:
        img_hidden:  Hidden dim of image patch tokens (768 for CLIP ViT-B/16).
        txt_hidden:  Hidden dim of text token embeddings (512 for CLIP text).
        d_model:     Internal cross-attention dim (default 512).
        n_heads:     Number of attention heads.
        n_layers:    Number of cross-attention layers.
        vocab_size:  CLIP BPE vocabulary size (49408).
        dropout:     Dropout rate.
    """

    CLIP_IMG_HIDDEN = 768   # ViT-B/16 vision encoder hidden dim
    CLIP_TXT_HIDDEN = 512   # CLIP text encoder hidden dim
    CLIP_VOCAB_SIZE = 49408 # CLIP BPE vocabulary

    def __init__(
        self,
        img_hidden: int = 768,
        txt_hidden: int = 512,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        vocab_size: int = 49408,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project image patches (768) and text tokens (512) to common d_model
        self.img_proj = nn.Linear(img_hidden, d_model, bias=False)
        self.txt_proj = (
            nn.Linear(txt_hidden, d_model, bias=False)
            if txt_hidden != d_model
            else nn.Identity()
        )

        # Cross-attention stack: text tokens attend to image patches
        self.txt2img_layers = nn.ModuleList(
            [CrossAttnLayer(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # MLM head: project fused token states to token logits
        self.mlm_head = nn.Linear(d_model, vocab_size)

        # Learnable global/local fusion weight; sigmoid(0) = 0.5 → equal mix at init.
        # alpha=1 → purely global; alpha=0 → purely local. Learned during training.
        self.fusion_logit = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(
        self,
        img_patch_tokens: torch.Tensor,   # (B, N_img, img_hidden)
        txt_token_hiddens: torch.Tensor,  # (B, N_txt, txt_hidden)
        txt_attention_mask: Optional[torch.Tensor] = None,  # (B, N_txt) 1=valid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-attend text tokens with image patches.

        Returns:
            local_txt_emb:  (B, d_model) mean-pooled, L2-normalised
            fused_hiddens:  (B, N_txt, d_model) per-token; used for MLM
        """
        img_p = self.img_proj(img_patch_tokens)  # (B, N_img, d_model)
        txt_p = self.txt_proj(txt_token_hiddens)  # (B, N_txt, d_model)

        # Cross-attention: text tokens as query, image patches as key/value
        x = txt_p
        for layer in self.txt2img_layers:
            x = layer(x, img_p)  # (B, N_txt, d_model)

        # Masked mean pooling over valid text positions
        if txt_attention_mask is not None:
            m = txt_attention_mask.unsqueeze(-1).float()   # (B, N_txt, 1)
            local_txt_emb = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        else:
            local_txt_emb = x.mean(1)

        return F.normalize(local_txt_emb, dim=-1), x

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def mlm_predictions(self, fused_hiddens: torch.Tensor) -> torch.Tensor:
        """Compute MLM logits at all token positions.

        Args:
            fused_hiddens: (B, N_txt, d_model)
        Returns:
            logits: (B, N_txt, vocab_size)
        """
        return self.mlm_head(fused_hiddens)

    def fuse(
        self,
        global_emb: torch.Tensor,  # (B, D) L2-normalised
        local_emb: torch.Tensor,   # (B, D) L2-normalised
    ) -> torch.Tensor:
        """Weighted sum of global and local embeddings, re-normalised.

        α = sigmoid(fusion_logit); starts near 0.5 but is learnable.
        """
        alpha = torch.sigmoid(self.fusion_logit)
        fused = alpha * global_emb + (1.0 - alpha) * local_emb
        return F.normalize(fused, dim=-1)
