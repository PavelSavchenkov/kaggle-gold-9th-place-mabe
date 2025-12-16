from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

from dl.base_mice_model import (
    BaseMiceModel,
    binary_focal_loss_with_logits,
    compute_pairwise_ranking_loss,
)
from dl.configs import DL_TrainConfig
from dl.features_lookup_gpu import FeaturesLookupGPU
from dl.layers import TemporalConvBlock, cosine_loss  # type: ignore
from dl.metrics import LAB_FEATURE_NAME  # type: ignore


class MiceTCNModel(BaseMiceModel):
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

        self.tcn_config = train_config.tcn_config
        assert self.tcn_config is not None
        self.H = self.tcn_config.hidden_dim

        self.input_proj = nn.Linear(self.D, self.H)

        blocks: list[nn.Module] = []
        for d in self.tcn_config.dilations:
            blocks.append(
                TemporalConvBlock(
                    in_channels=self.H,
                    out_channels=self.H,
                    kernel_size=self.tcn_config.kernel_size,
                    dilation=d,
                    dropout=self.tcn_config.dropout_prob,
                )
            )

        self.tcn = nn.Sequential(*blocks)

        self.pooled_H = self.H
        if self.tcn_config.extended_head:
            self.pooled_H *= 4

        if self.tcn_config.pooling_head_config_v0 is not None:
            pool_cfg = self.tcn_config.pooling_head_config_v0
            inner_h = int(self.pooled_H * pool_cfg.downsample_coef)
            self.head = nn.Sequential(
                nn.LayerNorm(self.pooled_H),
                nn.Linear(self.pooled_H, inner_h),
                nn.ReLU(),
                nn.Dropout(pool_cfg.dropout_prob),
                nn.Linear(inner_h, self.n_classes),
            )
        else:
            self.head = nn.Linear(self.pooled_H, self.n_classes)

        if self.train_config.self_supervised_config is not None:
            self.ssl_predictor = nn.Sequential(
                nn.Linear(self.H, self.H),
                nn.ReLU(),
                nn.Linear(self.H, self.H),
            )

    def forward(self, **batch) -> dict[str, Any]:
        assert self.tcn_config is not None
        # ----- feature pipeline -----
        batch = self.compute_features(batch)

        feats = batch["feats"]
        B, T, D = feats.shape

        # project to hidden_dim
        h = self.input_proj(feats)  # [B, T, H]

        # TCN expects [B, C, T]
        h = h.permute(0, 2, 1)  # [B, H, T]
        h = self.tcn(h)  # [B, H, T]

        # back to [B, T, H]
        h = h.permute(0, 2, 1)

        t0 = T // 2
        center_feats = h[:, t0, :]  # [B, H]

        if self.tcn_config.extended_head:  # type: ignore
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

        labels = batch.get("labels", None)

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
            assert loss is not None

            if self.tcn_config.ranking_loss_config is not None:
                cfg = self.tcn_config.ranking_loss_config
                ranking_loss = compute_pairwise_ranking_loss(
                    logits=logits,
                    labels=labels,
                    labels_known=labels_known,
                    margin=cfg.margin,
                )
                loss = loss * (1 - cfg.coef) + cfg.coef * ranking_loss

        if self.training and self.train_config.self_supervised_config is not None:
            pred = self.ssl_predictor(center_feats)

            losses = []
            for k in self.train_config.self_supervised_config.future_steps:
                t_future = min(t0 + k, T - 1)
                z_future = h[:, t_future, :]
                losses.append(cosine_loss(pred=pred, target=z_future))
            ssl_loss = sum(losses) / len(losses)
            ssl_loss *= self.train_config.self_supervised_config.loss_weight

            if loss is None:
                loss = ssl_loss
            else:
                loss += ssl_loss

        return {
            "loss": loss,
            "logits": logits,
            "probs": torch.sigmoid(logits),
            LAB_FEATURE_NAME: batch[LAB_FEATURE_NAME],
        }
