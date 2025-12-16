from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

from dl.base_mice_model import BaseMiceModel, binary_focal_loss_with_logits
from dl.configs import DL_TrainConfig
from dl.features_lookup_gpu import FeaturesLookupGPU
from dl.layers import TemporalConvBlock  # type: ignore
from dl.metrics import LAB_FEATURE_NAME  # type: ignore


class MiceMultiScaleTCNModel(BaseMiceModel):
    def __init__(
        self,
        feats_lookup: FeaturesLookupGPU,
        stats_path: Path | str,
        train_config: DL_TrainConfig,
    ):
        super().__init__(
            feats_config=train_config.features_config,
            aug_config=train_config.aug,
            feats_lookup=feats_lookup,
            stats_path=stats_path,
        )

        self.train_config = train_config.model_copy(deep=True)

        ms_cfg = train_config.multiscale_tcn_config
        assert ms_cfg is not None, "multiscale_tcn_config must be set"

        self.ms_cfg = ms_cfg

        self.input_proj = nn.Linear(self.D, ms_cfg.hidden_dim)

        # build branches
        branches: list[nn.Module] = []
        for branch_dilations in ms_cfg.branches_dilations:
            in_channels = ms_cfg.hidden_dim
            blocks: list[nn.Module] = []
            for d in branch_dilations:
                blocks.append(
                    TemporalConvBlock(
                        in_channels=in_channels,
                        out_channels=ms_cfg.hidden_dim,
                        kernel_size=ms_cfg.kernel_size,
                        dilation=d,
                        dropout=ms_cfg.dropout_prob,
                    )
                )
                in_channels = ms_cfg.hidden_dim
            branches.append(nn.Sequential(*blocks))

        self.branches = nn.ModuleList(branches)

        if ms_cfg.extended_head:
            per_branch_dim = 4 * ms_cfg.hidden_dim  # center + local + mean + max
        else:
            per_branch_dim = ms_cfg.hidden_dim

        self.head_in_dim = per_branch_dim * len(ms_cfg.branches_dilations)
        self.head = nn.Linear(self.head_in_dim, self.n_classes)

    def _pool_branch(self, h_t: torch.Tensor, extended_head: bool) -> torch.Tensor:
        """
        h_t: [B, T, H]

        Returns [B, H] or [B, 4H] depending on extended_head.
        """
        B, T, H = h_t.shape
        t0 = T // 2

        center = h_t[:, t0, :]  # [B, H]

        if not extended_head:
            return center

        # local window around center
        left = max(t0 - 2, 0)
        right = min(t0 + 3, T)  # exclusive
        local = h_t[:, left:right, :].mean(dim=1)  # [B, H]

        global_mean = h_t.mean(dim=1)  # [B, H]
        global_max = h_t.max(dim=1).values  # [B, H]

        return torch.cat([center, local, global_mean, global_max], dim=-1)

    def forward(self, **batch) -> dict[str, Any]:
        # ----- feature pipeline -----
        batch = self.compute_features(batch)
        feats = batch["feats"]  # [B, T, D]
        B, T, D = feats.shape

        # project to hidden_dim
        h = self.input_proj(feats)  # [B, T, H]

        # TCN expects [B, C, T]
        h_c = h.permute(0, 2, 1)  # [B, H, T]

        # pass through branches
        branch_feats: list[torch.Tensor] = []
        for branch in self.branches:
            out = branch(h_c)  # [B, H, T]
            out_t = out.permute(0, 2, 1)  # [B, T, H]
            branch_feat = self._pool_branch(out_t, self.ms_cfg.extended_head)
            branch_feats.append(branch_feat)

        # concat branches
        feats_all = torch.cat(branch_feats, dim=-1)  # [B, head_in_dim]

        logits = self.head(feats_all)  # [B, n_classes]

        labels = batch.get("labels", None)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            labels_known = batch["labels_known"]
            loss_w = batch["loss_w"]

            if self.train_config.focal is not None:
                loss_matr = binary_focal_loss_with_logits(
                    logits=logits,
                    targets=labels,
                    gamma=self.train_config.focal.gamma,
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
            LAB_FEATURE_NAME: batch[LAB_FEATURE_NAME],
        }
