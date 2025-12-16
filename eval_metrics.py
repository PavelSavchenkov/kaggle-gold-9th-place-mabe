from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, log_loss

from common.metrics_common import calc_best_f1_threshold


def _safe_mean(values: List[float]) -> float:
    arr = np.array(
        [v for v in values if v is not None and not np.isnan(v)], dtype=float
    )
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def compute_eval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    index_df: pd.DataFrame,
    prefix: str = "",
) -> Dict[str, float]:
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_true and y_pred must be 2D")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} must match"
        )
    if y_true.shape[1] != len(class_names):
        raise ValueError("class_names length must match number of columns in y_true")
    if "lab_id" not in index_df.columns:
        raise ValueError("index_df must contain 'lab_id' column")
    labs = index_df["lab_id"].to_numpy()

    assert np.all((y_true == 0) | (y_true == 1))

    N, K = y_true.shape
    probs = y_pred

    best_f1_threshold = calc_best_f1_threshold(
        y_true=y_true.astype(int).ravel(), y_pred=probs.ravel()
    )

    unique_labs = list(pd.unique(labs))

    # Containers
    per_action_pr_auc: Dict[str, float] = {}
    per_action_logloss: Dict[str, float] = {}
    per_lab_pr_auc: Dict[str, float] = {}
    per_lab_logloss: Dict[str, float] = {}
    # F1 aggregated for prod-avg/total (best threshold only) and per-lab.
    # We still omit per-action F1 blocks to reduce noise.
    per_lab_f1_best: Dict[str, float] = {}

    # Precompute per-lab-per-action metrics, then aggregate in two axes
    # We'll store per-lab lists per metric for each action and vice versa.
    per_lab_action_pr_auc: Dict[str, Dict[str, float]] = {
        lab: {} for lab in unique_labs
    }
    per_lab_action_logloss: Dict[str, Dict[str, float]] = {
        lab: {} for lab in unique_labs
    }
    per_lab_action_f1_best: Dict[str, Dict[str, float]] = {
        lab: {} for lab in unique_labs
    }

    for j, action in enumerate(class_names):
        # For this action, compute per-lab metrics
        pr_auc_vals_for_action: List[float] = []
        logloss_vals_for_action: List[float] = []
        # no fixed-threshold F1 aggregation per action

        y_col = y_true[:, j]
        p_col = probs[:, j]

        for lab in unique_labs:
            mask = labs == lab
            if not np.any(mask):
                continue
            yt = y_col[mask]
            pt = p_col[mask]

            auc_v = float(average_precision_score(yt, pt))

            # Clip probabilities for numerical stability for log-loss
            eps = 1e-15
            pt_clip = np.clip(pt, eps, 1 - eps)
            logloss_v = float(log_loss(yt, pt_clip, labels=[0, 1]))

            preds_best = (pt >= float(best_f1_threshold)).astype(int)
            f1_v_best = float(f1_score(yt, preds_best, zero_division=0))

            pr_auc_vals_for_action.append(auc_v)
            per_lab_action_pr_auc[lab][action] = auc_v
            logloss_vals_for_action.append(logloss_v)
            per_lab_action_logloss[lab][action] = logloss_v
            per_lab_action_f1_best[lab][action] = f1_v_best

        # Aggregate per-action over labs where it occurred
        if pr_auc_vals_for_action:
            per_action_pr_auc[action] = _safe_mean(pr_auc_vals_for_action)
        if logloss_vals_for_action:
            per_action_logloss[action] = _safe_mean(logloss_vals_for_action)

    # Now aggregate per-lab over actions that occurred in that lab
    for lab in unique_labs:
        auc_list = list(per_lab_action_pr_auc[lab].values())
        logloss_list = list(per_lab_action_logloss[lab].values())
        f1_list_best = list(per_lab_action_f1_best[lab].values())
        if auc_list:
            per_lab_pr_auc[str(lab)] = _safe_mean(auc_list)
        if logloss_list:
            per_lab_logloss[str(lab)] = _safe_mean(logloss_list)
        if f1_list_best:
            per_lab_f1_best[str(lab)] = _safe_mean(f1_list_best)

    # prod-avg: average over labs of per-lab metrics (kept for continuity)
    prod_avg_pr_auc = _safe_mean(list(per_lab_pr_auc.values()))
    prod_avg_logloss = _safe_mean(list(per_lab_logloss.values()))
    prod_avg_f1_best = _safe_mean(list(per_lab_f1_best.values()))

    # Whole-dataset ("total"): compute metrics across all valid (non -1) entries
    # by flattening across labs and actions, without any averaging by group.
    valid_mask = y_true != -1
    yt_flat = y_true[valid_mask].astype(int)
    pt_flat = probs[valid_mask]

    # For PR AUC, ensure both classes are present; otherwise return NaN
    if yt_flat.size > 0 and np.any(yt_flat == 1) and np.any(yt_flat == 0):
        total_pr_auc = float(average_precision_score(yt_flat, pt_flat))
    else:
        total_pr_auc = float("nan")

    eps = 1e-15
    pt_clip_flat = np.clip(pt_flat, eps, 1 - eps)
    total_logloss = float(log_loss(yt_flat, pt_clip_flat, labels=[0, 1]))
    total_f1_best = float(
        f1_score(
            yt_flat, (pt_flat >= float(best_f1_threshold)).astype(int), zero_division=0
        )
    )

    # Build flat dict with keys in the specified block/key format
    out: Dict[str, float] = {}
    # prod-avg totals
    out[f"{prefix}prod-avg/pr-auc"] = float(prod_avg_pr_auc)
    out[f"{prefix}prod-avg/log-loss"] = float(prod_avg_logloss)
    out[f"{prefix}prod-avg/f1-best-th"] = float(prod_avg_f1_best)
    # whole-dataset totals
    out[f"{prefix}total/pr-auc"] = float(total_pr_auc)
    out[f"{prefix}total/log-loss"] = float(total_logloss)
    out[f"{prefix}total/f1-best-th"] = float(total_f1_best)

    # per-action
    for action, v in per_action_pr_auc.items():
        out[f"{prefix}per-action-pr-auc/{action}"] = float(v)
    for action, v in per_action_logloss.items():
        out[f"{prefix}per-action-log-loss/{action}"] = float(v)

    # per-lab
    for lab, v in per_lab_pr_auc.items():
        out[f"{prefix}per-lab-pr-auc/{lab}"] = float(v)
    for lab, v in per_lab_logloss.items():
        out[f"{prefix}per-lab-log-loss/{lab}"] = float(v)
    # per-lab F1 at best threshold (new block)
    for lab, v in per_lab_f1_best.items():
        out[f"{prefix}per-lab-f1-best-th/{lab}"] = float(v)

    return out
