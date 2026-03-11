"""
Loss functions for cross-modal person retrieval.

  InfoNCELoss  — symmetric bidirectional contrastive loss.
  IDLoss       — softmax cross-entropy over identity classes.
  TripletLoss  — batch-hard triplet loss (Phase 3 image-only branch).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# InfoNCE (symmetric)
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE over image-text pairs.

    For a batch of B (image_i, text_i) pairs the logit matrix is
        S = exp(logit_scale) * (image_embeds @ text_embeds.T)
    and the loss is the mean of the image→text and text→image cross-entropies
    with diagonal targets.
    """

    def forward(
        self,
        image_embeds: torch.Tensor,    # (B, D) L2-normalised
        text_embeds: torch.Tensor,     # (B, D) L2-normalised
        logit_scale: torch.Tensor,     # scalar
    ) -> torch.Tensor:
        scale = logit_scale.exp()
        logits = scale * image_embeds @ text_embeds.T   # (B, B)
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.T, targets)
        return (loss_i2t + loss_t2i) / 2.0


# ---------------------------------------------------------------------------
# ID classification loss
# ---------------------------------------------------------------------------

class IDLoss(nn.Module):
    """Softmax cross-entropy over a trainable linear classifier.

    The linear head is registered as a module so its parameters are
    updated by the optimizer automatically.

    Args:
        embed_dim: Embedding dimension.
        num_classes: Total number of training identities.
    """

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.01)
        self.num_classes = num_classes

    def forward(
        self,
        image_embeds: torch.Tensor,   # (B, D)
        text_embeds: torch.Tensor,    # (B, D)
        pids: torch.Tensor,           # (B,) int, -1 means unknown
    ) -> torch.Tensor:
        mask = pids >= 0
        if mask.sum() == 0:
            return image_embeds.new_tensor(0.0)

        im_valid = image_embeds[mask]
        tx_valid = text_embeds[mask]
        labels = pids[mask]

        logits_img = self.classifier(im_valid)
        logits_txt = self.classifier(tx_valid)
        loss = (F.cross_entropy(logits_img, labels)
                + F.cross_entropy(logits_txt, labels)) / 2.0
        return loss


# ---------------------------------------------------------------------------
# Batch-hard triplet loss  (Phase 3)
# ---------------------------------------------------------------------------

class TripletLoss(nn.Module):
    """Batch-hard triplet loss with soft margin.

    For each anchor, the hardest positive and negative in the batch are
    selected based on L2 distance.

    Args:
        margin: Triplet margin (use 0 for soft margin / hinge).
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeds: torch.Tensor,  # (B, D) L2-normalised
        pids: torch.Tensor,    # (B,) int
    ) -> torch.Tensor:
        # Squared L2 via dot-product trick (embeds already normalised)
        dot = embeds @ embeds.T            # (B, B) cosine sims ∈ [-1, 1]
        dist = (2.0 - 2.0 * dot).clamp(min=0).sqrt()  # (B, B) ≥ 0

        # Build masks
        pid_eq = pids.unsqueeze(0) == pids.unsqueeze(1)   # (B, B)
        pid_ne = ~pid_eq

        # Hard positive: same pid, maximum distance
        pos_dist, _ = (dist * pid_eq.float()).max(dim=1)

        # Hard negative: different pid, minimum distance (mask same-pid with large val)
        neg_dist, _ = (dist + 1e4 * pid_eq.float()).min(dim=1)

        if self.margin > 0:
            loss = F.relu(pos_dist - neg_dist + self.margin)
        else:
            # Soft margin
            loss = torch.log1p(torch.exp(pos_dist - neg_dist))

        return loss.mean()
