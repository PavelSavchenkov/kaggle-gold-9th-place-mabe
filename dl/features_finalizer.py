from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch  # type: ignore
import torch.nn as nn  # type: ignore

from common.constants import MOUSE_CATEGORICAL_FEATURES, VIDEO_CATEGORICAL_FEATURES
from common.helpers import get_default_cuda_float_dtype  # type: ignore
from dl.configs import FeaturesConfigDL


class FeaturesFinalizer(nn.Module):
    """
    Stacks handcrafted numeric features (from landmark feature extractor) + nan masks + categorical
    """

    config: FeaturesConfigDL

    def __init__(self, config: FeaturesConfigDL, stats_path: Path | str):
        super().__init__()
        self.config = config

        stats = np.load(stats_path)
        dtype = get_default_cuda_float_dtype()
        for attr in ["mean", "std"]:
            t = torch.from_numpy(stats[attr]).to(dtype=dtype)
            self.register_buffer(attr, t, persistent=True)

        self.cat_embeddings = nn.ModuleDict()
        for base_name, embedding_dim in self.config.categorical_dims.items():
            if base_name in VIDEO_CATEGORICAL_FEATURES:
                dim_to_embed = len(VIDEO_CATEGORICAL_FEATURES[base_name])
                self.cat_embeddings[base_name] = nn.Embedding(
                    dim_to_embed, embedding_dim
                )
            else:
                assert base_name in MOUSE_CATEGORICAL_FEATURES
                dim_to_embed = len(MOUSE_CATEGORICAL_FEATURES[base_name])
                self.cat_embeddings[base_name] = nn.Embedding(
                    dim_to_embed, embedding_dim
                )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        numeric = batch["numeric_feats"]  # [B, T, D_num]
        numeric = (numeric - self.mean) / self.std

        B, T, _ = numeric.shape
        final_chunks: list[torch.Tensor] = [numeric]

        # ---- masks ----
        if self.config.include_masks:
            for who in ["agent", "target"]:
                for c in ["x", "y"]:
                    key = f"{who}_{c}_mask"
                    assert key in batch
                    mask = batch[key].to(numeric.dtype)
                    assert mask.shape[:2] == numeric.shape[:2]
                    final_chunks.append(mask)

        # ---- categorical embeddings ----
        for base_name in sorted(self.cat_embeddings.keys()):
            emb = self.cat_embeddings[base_name]

            def add_embed(key):
                assert key in batch, f"missing key={key}"
                idx_t = batch[key].long().view(B)  # [B]
                emb_vec = emb(idx_t)  # [B, D_emb]
                emb_time = emb_vec.unsqueeze(1).expand(B, T, emb_vec.size(-1))
                final_chunks.append(emb_time.to(numeric.dtype))

            key = f"{base_name}_idx"
            if base_name in VIDEO_CATEGORICAL_FEATURES:
                add_embed(key)
            else:
                assert base_name in MOUSE_CATEGORICAL_FEATURES
                for who in ["agent", "target"]:
                    add_embed(f"{who}_{key}")

        # ---- concat everything ----
        feats = torch.cat(final_chunks, dim=-1)  # [B, T, D_final]

        out = dict(batch)
        out["feats"] = feats
        return out
