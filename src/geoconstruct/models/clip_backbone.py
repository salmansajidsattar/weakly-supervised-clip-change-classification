"""Frozen CLIP wrapper using open_clip. Used only for one-time feature
extraction and (separately) for encoding text templates in the explainer.

We freeze all parameters. There is no fine-tuning path here by design.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import open_clip
import torch
import torch.nn as nn
from PIL import Image

log = logging.getLogger(__name__)


class CLIPBackbone(nn.Module):
    """Frozen CLIP image+text encoder.

    Default: ViT-B/32 OpenAI weights. Output dim is 512 for ViT-B/32.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
    ):
        super().__init__()
        log.info("Loading CLIP %s (pretrained=%s) on %s",
                 model_name, pretrained, device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.embed_dim = self._infer_embed_dim()
        log.info("CLIP embed_dim=%d", self.embed_dim)

    def _infer_embed_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            return int(self.model.encode_image(dummy).shape[-1])

    @torch.no_grad()
    def encode_images(self, paths: Iterable[str | Path], batch_size: int = 8):
        """Encode a list of image paths. Returns a (N, D) float32 tensor on CPU."""
        feats: list[torch.Tensor] = []
        batch: list[torch.Tensor] = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            batch.append(self.preprocess(img))
            if len(batch) == batch_size:
                feats.append(self._encode_batch(batch))
                batch = []
        if batch:
            feats.append(self._encode_batch(batch))
        return torch.cat(feats, dim=0).cpu()

    @torch.no_grad()
    def _encode_batch(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        x = torch.stack(tensors, dim=0).to(self.device)
        z = self.model.encode_image(x)
        z = z / z.norm(dim=-1, keepdim=True)
        return z.cpu()

    @torch.no_grad()
    def encode_texts(self, texts: Iterable[str]) -> torch.Tensor:
        """Encode a list of strings. Returns (N, D) float32 tensor on CPU,
        L2-normalized."""
        tokens = self.tokenizer(list(texts)).to(self.device)
        z = self.model.encode_text(tokens)
        z = z / z.norm(dim=-1, keepdim=True)
        return z.cpu()
