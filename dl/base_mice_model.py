from __future__ import annotations

from pathlib import Path
from typing import Any

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

from common.constants import (
    ACTION_NAMES_IN_TEST,
    MOUSE_CATEGORICAL_FEATURES,
    VIDEO_CATEGORICAL_FEATURES,
)
from dl.configs import AugmentationsConfig, FeaturesConfigDL
from dl.features_finalizer import FeaturesFinalizer  # type: ignore
from dl.features_lookup_gpu import FeaturesLookupGPU
from dl.landmark_feature_extractor import LandmarkFeatureExtractor


def compute_pairwise_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    labels_known: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """
    Pairwise margin ranking loss over known labels.

    Assumes: at most one positive (label == 1) per row where labels_known == 1.

    For each sample i:
      - pos: single class c with labels[i, c] == 1 and labels_known[i, c] == 1
      - neg: all classes j with labels[i, j] == 0 and labels_known[i, j] == 1

    For each (pos, neg) pair we add:
        max(0, margin - logit_pos + logit_neg)

    Returns a scalar tensor (mean over all pairs), or 0 if no valid pairs.
    """
    device = logits.device
    dtype = logits.dtype

    pos_mask = (labels > 0.5) & (labels_known > 0.5)  # (B, C)
    neg_mask = (labels <= 0.5) & (labels_known > 0.5)  # (B, C)

    # Rows that actually have a pos and at least one neg
    pos_row_has = pos_mask.any(dim=1)  # (B,)
    neg_row_has = neg_mask.any(dim=1)  # (B,)
    valid_rows = pos_row_has & neg_row_has

    if not torch.any(valid_rows):
        return logits.new_tensor(0.0)

    # Only consider negatives in rows that also have a positive
    valid_neg_mask = neg_mask & valid_rows.unsqueeze(1)  # (B, C)

    # Indices of all (row, neg_class) pairs
    neg_rows, neg_cols = valid_neg_mask.nonzero(as_tuple=True)  # (#pairs,)

    # Positive logit for each row (1 pos per row assumed)
    # (logits * pos_mask) sums to the positive logit per row.
    pos_logits_per_row = (logits * pos_mask.to(dtype)).sum(dim=1)  # (B,)

    # Positive logit for each negative example (matched by row)
    pos_for_neg = pos_logits_per_row[neg_rows]  # (#pairs,)
    neg_logits = logits[neg_rows, neg_cols]  # (#pairs,)

    diff = margin - pos_for_neg + neg_logits  # (#pairs,)
    loss = F.relu(diff).mean()

    return loss


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    logits: [B, C]
    targets: [B, C] in {0, 1}
    returns: per-example, per-class loss [B, C] (no reduction)
    """
    # standard BCE, but keep per-element
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    eps = 1e-6

    # p_t = p if y = 1 else (1 - p)
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    p_t = p_t.clamp(min=eps, max=1.0 - eps)

    focal_factor = (1.0 - p_t) ** gamma

    return focal_factor * bce


class BaseMiceModel(nn.Module):
    def __init__(
        self,
        feats_config: FeaturesConfigDL,
        aug_config: AugmentationsConfig,
        feats_lookup: FeaturesLookupGPU,
        stats_path: Path | str,
        verbose: bool = True,
    ):
        super().__init__()
        self.features_config = feats_config
        self.features_lookup = feats_lookup  # not an nn.Module, just stored

        self.landmarks_feature_extractor = LandmarkFeatureExtractor(
            feats_config=self.features_config,
            aug_config=aug_config,
        )
        self.features_finalizer = FeaturesFinalizer(self.features_config, stats_path)

        self.n_classes = len(ACTION_NAMES_IN_TEST)

        self.device = torch.device("cuda")
        self.to(self.device)

        with torch.no_grad():
            dummy_batch = self._build_dummy_batch_post_lookup()
            dummy_out = self._run_post_lookup_pipeline(dummy_batch)
            self.D = dummy_out["feats"].shape[-1]
            if verbose:
                print(f"T: {self.features_config.frames_in_window}, D: {self.D}")

    def _build_dummy_batch_post_lookup(
        self,
    ) -> dict[str, torch.Tensor]:
        B = 1
        T = self.features_config.frames_in_window
        K = len(self.features_config.bodyparts)

        dummy: dict[str, torch.Tensor] = {
            "agent_x": torch.zeros(B, T, K, device=self.device),
            "agent_y": torch.zeros(B, T, K, device=self.device),
            "target_x": torch.zeros(B, T, K, device=self.device),
            "target_y": torch.zeros(B, T, K, device=self.device),
            "agent_x_mask": torch.zeros(B, T, K, device=self.device),
            "agent_y_mask": torch.zeros(B, T, K, device=self.device),
            "target_x_mask": torch.zeros(B, T, K, device=self.device),
            "target_y_mask": torch.zeros(B, T, K, device=self.device),
        }

        for base_name in sorted(self.features_config.categorical_dims.keys()):
            if base_name in VIDEO_CATEGORICAL_FEATURES:
                dummy[f"{base_name}_idx"] = torch.zeros(
                    B, dtype=torch.long, device=self.device
                )
            else:
                assert base_name in MOUSE_CATEGORICAL_FEATURES
                for who in ["agent", "target"]:
                    dummy[f"{who}_{base_name}_idx"] = torch.zeros(
                        B, dtype=torch.long, device=self.device
                    )

        return dummy

    def _run_post_lookup_pipeline(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        batch = self.landmarks_feature_extractor(batch)
        batch = self.features_finalizer(batch)
        return batch

    def compute_features(
        self,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        batch = self.features_lookup.preprocess_batch(batch, is_train=self.training)
        batch = self._run_post_lookup_pipeline(batch)
        return batch
