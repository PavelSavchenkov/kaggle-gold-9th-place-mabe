from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

from dl.base_mice_model import BaseMiceModel, binary_focal_loss_with_logits
from dl.configs import DL_TrainConfig
from dl.features_lookup_gpu import FeaturesLookupGPU
from dl.metrics import LAB_FEATURE_NAME  # type: ignore


class MiceConvTransformerModel(BaseMiceModel):
    def __init__(
        self,
        feats_lookup: FeaturesLookupGPU,
        stats_path: Path | str,
        train_config: DL_TrainConfig,
        verbose: bool = True,
    ):
        super().__init__(
            feats_config=train_config.features_config,
            aug_config=train_config.aug,
            feats_lookup=feats_lookup,
            stats_path=stats_path,
            verbose=verbose,
        )

        self.train_config = train_config.model_copy(deep=True)

        self.model_config = train_config.conv_transformer_config
        assert self.model_config is not None

        self.H = self.model_config.hidden_dim
        assert self.H % self.model_config.num_heads == 0

        # [B, T, D] -> [B, T, H]
        self.input_proj = nn.Linear(self.D, self.H)

        # shallow conv preprocessing
        assert self.model_config.kernel_size % 2 == 1
        conv_blocks = []
        dilations = [1] * self.model_config.n_conv_layers
        if self.model_config.dilations is not None:
            dilations = self.model_config.dilations
        assert len(dilations) == self.model_config.n_conv_layers
        for i in range(self.model_config.n_conv_layers):
            dilation = dilations[i]
            conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=self.H,
                        out_channels=self.H,
                        kernel_size=self.model_config.kernel_size,
                        padding=(self.model_config.kernel_size - 1) * dilation // 2,
                        dilation=dilation
                    ),
                    nn.BatchNorm1d(self.H),
                    nn.GELU(),
                    nn.Dropout(self.model_config.conv_dropout_prob),
                )
            )
        self.conv_blocks = nn.ModuleList(conv_blocks)

        max_seq_len = self.train_config.features_config.frames_in_window  # max T
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, self.H))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.H,
            nhead=self.model_config.num_heads,
            dim_feedforward=self.model_config.ff_dim,
            dropout=self.model_config.transformer_dropout_prob,
            activation="gelu",
            batch_first=True,  # [B, T, H]
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.model_config.num_transformer_layers,
        )

        self.pooled_H = self.H
        if self.model_config.extended_head:
            self.pooled_H *= 4

        self.head = nn.Linear(self.pooled_H, self.n_classes)

        assert self.train_config.self_supervised_config is None

    def forward(self, **batch) -> dict[str, Any]:
        # ----- feature pipeline -----
        batch = self.compute_features(batch)

        feats = batch["feats"]
        B, T, D = feats.shape

        h = self.input_proj(feats)

        h_conv = h.permute(0, 2, 1)  # [B, H, T]
        for conv in self.conv_blocks:
            residual = h_conv
            h_conv = conv(h_conv)
            h_conv = h_conv + residual
        h = h_conv.permute(0, 2, 1)  # [B, T, H]

        pos_emb = self.pos_embedding[:, :T, :]  # [1, T, H]
        h = h + pos_emb
        h = self.transformer(h)

        t0 = T // 2
        center_feats = h[:, t0, :]  # [B, H]

        if self.model_config.extended_head:  # type: ignore
            local = h[:, max(t0 - 2, 0) : min(t0 + 3, T), :].mean(dim=1)

            global_mean = h.mean(dim=1)
            global_max = h.max(dim=1).values

            pooled_feats = torch.cat(
                [center_feats, local, global_mean, global_max], dim=-1
            )
        else:
            pooled_feats = center_feats

        assert pooled_feats.size(1) == self.pooled_H

        logits = self.head(pooled_feats)

        labels = batch.get("labels")

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            labels_known = batch["labels_known"]
            loss_w = batch["loss_w"]

            if self.train_config.label_smoothing_config is not None:
                eps = self.train_config.label_smoothing_config.eps
                labels = labels * (1 - 2 * eps) + eps

            loss_matr = None  # [B, n_classes]
            if self.train_config.focal is not None:
                loss_matr = binary_focal_loss_with_logits(
                    logits=logits, targets=labels, gamma=self.train_config.focal.gamma
                )
            else:
                loss_matr = F.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none"
                )
            weight = loss_w * labels_known
            loss = (loss_matr * weight).sum() / weight.sum().clamp_min(1.0)

        return {
            "loss": loss,
            "logits": logits,
            "probs": torch.sigmoid(logits),
            LAB_FEATURE_NAME: batch[LAB_FEATURE_NAME],
        }
