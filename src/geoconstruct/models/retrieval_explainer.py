"""Retrieval-based explainer.

For each prediction, pick the class-template caption whose CLIP text embedding
has the highest cosine similarity to the fused image feature. Faithful by
construction: we never generate free text, we only retrieve from a fixed pool.

The pool is precomputed once from CLASS_TEMPLATES via CLIPBackbone.encode_texts.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from ..data.captions import CLASS_NAMES, CLASS_TEMPLATES

log = logging.getLogger(__name__)


class RetrievalExplainer:
    def __init__(self, template_features: dict[str, torch.Tensor],
                 template_texts: dict[str, list[str]]):
        # All templates concatenated into a flat tensor for fast retrieval.
        self.flat_text: list[str] = []
        self.flat_class: list[str] = []
        flat: list[torch.Tensor] = []
        for cls_name in CLASS_NAMES:
            feats = template_features[cls_name]      # (k, D), L2-normalized
            texts = template_texts[cls_name]
            for i in range(feats.shape[0]):
                flat.append(feats[i])
                self.flat_text.append(texts[i])
                self.flat_class.append(cls_name)
        self.bank = torch.stack(flat, dim=0)           # (K, D)

    @classmethod
    def from_backbone(cls, backbone) -> "RetrievalExplainer":
        """Build by encoding all CLASS_TEMPLATES with the given backbone."""
        feats: dict[str, torch.Tensor] = {}
        for cls_name, texts in CLASS_TEMPLATES.items():
            feats[cls_name] = backbone.encode_texts(texts)  # (k, D)
        return cls(template_features=feats, template_texts=CLASS_TEMPLATES)

    def explain(self, fused_feat: torch.Tensor, restrict_class: str | None = None
                ) -> tuple[str, str, float]:
        """Return (class_name, text, score). If restrict_class is given,
        only templates of that class are considered (use the model's predicted
        class for this in practice)."""
        if fused_feat.dim() == 1:
            q = fused_feat.unsqueeze(0)
        else:
            q = fused_feat
        # L2 normalize the query — the head outputs un-normalized vectors;
        # cosine requires normalization.
        q = F.normalize(q, dim=-1)
        sims = q @ self.bank.T  # (B, K)
        mask = torch.ones(sims.shape[-1], dtype=torch.bool)
        if restrict_class is not None:
            for k, c in enumerate(self.flat_class):
                if c != restrict_class:
                    mask[k] = False
            sims = sims.masked_fill(~mask.unsqueeze(0), float("-inf"))
        idx = int(sims[0].argmax().item())
        return self.flat_class[idx], self.flat_text[idx], float(sims[0, idx])
