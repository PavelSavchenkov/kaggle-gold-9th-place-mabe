from __future__ import annotations

from collections import defaultdict, namedtuple
from typing import ClassVar

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import average_precision_score, f1_score, log_loss


def calc_pr_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(average_precision_score(y_true=y_true, y_score=y_pred))


F1 = namedtuple("F1", ["th", "val"])


def calc_best_f1(y_true: np.ndarray, y_pred: np.ndarray) -> F1:
    pt = y_pred
    yt = y_true

    if not np.issubdtype(pt.dtype, np.floating):
        raise TypeError("y_pred must be floating probabilities")
    if not np.issubdtype(yt.dtype, np.integer):
        raise TypeError("y_true must be integer labels")

    min_prob = np.min(pt)
    max_prob = np.max(pt)
    if not (min_prob >= 0.0 - 1e-4 and max_prob <= 1.0 + 1e-4):
        raise ValueError(
            f"y_pred must be in [0, 1], but min_prob is {min_prob:.7f}, max_prob is {max_prob:.7f}"
        )
    if not np.all((yt == 0) | (yt == 1)):
        raise ValueError("y_true must contain only {0, 1}")

    if pt.size == 0:
        raise ValueError("Empty inputs")

    # Sort by predicted probability (descending)
    order = np.argsort(-pt)
    pt_sorted = pt[order]
    yt_sorted = yt[order].astype(bool)

    total_pos = yt_sorted.sum()

    # If no positives, F1 is always 0; any threshold is equivalent
    if total_pos == 0:
        return F1(th=0.5, val=0.0)

    # Cumulative TP when taking top k as positive
    cum_tp = np.cumsum(yt_sorted)
    # Number of predicted positives at position k is k+1

    n = pt_sorted.size

    # Indices of last occurrence of each unique prob
    # (these correspond to candidate thresholds)
    change_indices = np.nonzero(np.diff(pt_sorted))[0]
    cand_indices = np.append(change_indices, n - 1)

    tp = cum_tp[cand_indices]
    pred_pos = cand_indices + 1
    fp = pred_pos - tp
    fn = total_pos - tp

    denom = 2.0 * tp + fp + fn
    f1 = np.zeros_like(denom, dtype=float)
    mask = denom > 0
    f1[mask] = 2.0 * tp[mask] / denom[mask]

    best_idx = int(np.argmax(f1))
    best_thr = float(pt_sorted[cand_indices[best_idx]])
    return F1(th=best_thr, val=f1[best_idx])


def calc_best_f1_threshold(y_true, y_pred) -> float:
    return calc_best_f1(y_true, y_pred).th


def calc_nested_f1_with_th_per_fold(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    fold_id: np.ndarray,
    folds: list[int] | None = None,
) -> tuple[list[float], dict[int, float]]:
    if folds is None:
        folds = list(sorted(np.unique(fold_id)))

    f1_list = []
    th_per_fold = {}
    for f in folds:
        mask_f = fold_id == f
        th = calc_best_f1_threshold(y_true=y_true[~mask_f], y_pred=y_pred[~mask_f])
        f1 = float(f1_score(y_true=y_true[mask_f], y_pred=y_pred[mask_f] >= th))
        th_per_fold[f] = th
        f1_list.append(f1)
    return f1_list, th_per_fold


def calc_nested_f1(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    fold_id: np.ndarray,
    folds: list[int] | None = None,
) -> list[float]:
    return calc_nested_f1_with_th_per_fold(
        y_pred=y_pred, y_true=y_true, fold_id=fold_id, folds=folds
    )[0]


class OOF_Metrics(BaseModel):
    metric_fields: ClassVar[list[str]] = [
        "nested_f1",
        "oof_f1",
        "f1_per_fold",
        "oof_pr_auc",
        "pr_auc_per_fold",
    ]

    nested_f1: list[float] = Field(default_factory=list)
    oof_f1: float = -1.0
    f1_per_fold: list[float] = Field(default_factory=list)
    oof_pr_auc: float = -1.0
    pr_auc_per_fold: list[float] = Field(default_factory=list)

    oof_th: float = -1.0
    th_per_fold: dict[int, float] = Field(default_factory=dict)

    def __hash__(self) -> int:
        th_per_fold = [tuple(it) for it in self.th_per_fold.items()]
        return hash(
            (
                tuple(self.pr_auc_per_fold),
                self.oof_pr_auc,
                tuple(self.f1_per_fold),
                self.oof_f1,
                tuple(self.nested_f1),
                self.oof_th,
                tuple(th_per_fold),
            )
        )

    def get_folds(self) -> list[int]:
        return list(sorted(self.th_per_fold.keys()))

    @staticmethod
    def build(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        fold_id: np.ndarray,
        folds: list[int] | None = None,
    ):
        obj = OOF_Metrics()

        if folds is None:
            folds = [int(x) for x in sorted(set(np.unique(fold_id)))]
        else:
            assert folds == list(sorted(set(folds)))

        for f in folds:
            mask = fold_id == f
            pr_auc = calc_pr_auc(y_true=y_true[mask], y_pred=y_pred[mask])
            obj.pr_auc_per_fold.append(pr_auc)
            f1 = calc_best_f1(y_true=y_true[mask], y_pred=y_pred[mask]).val
            obj.f1_per_fold.append(f1)
        obj.oof_pr_auc = calc_pr_auc(y_true=y_true, y_pred=y_pred)
        f1 = calc_best_f1(y_true=y_true, y_pred=y_pred)
        obj.oof_f1 = f1.val
        obj.oof_th = f1.th

        obj.nested_f1, obj.th_per_fold = calc_nested_f1_with_th_per_fold(
            y_pred=y_pred, y_true=y_true, fold_id=fold_id, folds=folds
        )

        return obj


class DatasetScore(BaseModel):
    # metric_name (e.g. nested f1) -> metric value (avg labs avg actions)
    totals: dict[str, float] = Field(default_factory=dict)

    # lab name -> metric name -> metric value
    per_lab: dict[str, dict[str, float]] = Field(default_factory=dict)

    def to_str(self, include_labs: bool = False):
        s = []
        if include_labs:
            for lab in sorted(self.per_lab.keys()):
                s.append(f"{lab:20}")
                for f in OOF_Metrics.metric_fields:
                    value = self.per_lab[lab][f]
                    s.append(f"{'':10} {f:16}: {value:.5f}")
                s.append("\n")
        for f in OOF_Metrics.metric_fields:
            value = self.totals[f]
            s.append(f"{f:16}: {value:.5f}")
        return "\n".join(s)

    # @staticmethod
    # def from_oof_metrics_for_model(
    #     all_metrics: dict[tuple[str, str], OOF_Metrics_for_model],
    # ) -> "DatasetScore":
    #     all_metrics_reduced = {k: v.metrics for k, v in all_metrics.items()}
    #     return DatasetScore.from_oof_metrics(all_metrics=all_metrics_reduced)

    @staticmethod
    def from_oof_metrics(
        all_metrics: dict[tuple[str, str], OOF_Metrics],
    ) -> DatasetScore:
        fields = list(OOF_Metrics.metric_fields)
        per_lab = {}
        for f in fields:
            per_lab[f] = defaultdict(list)
        for (action, lab), oof in all_metrics.items():
            for f in fields:
                value = getattr(oof, f)
                if isinstance(value, list):
                    value = float(np.average(value))
                per_lab[f][lab].append(value)
        res = DatasetScore()
        for f in fields:
            avg_list = []
            for lab in per_lab[f].keys():
                avg = float(np.average(per_lab[f][lab]))
                # if f == "nested_f1":
                #     print(f"{lab:20}: {avg:.5f}")
                avg_list.append(avg)
                if lab not in res.per_lab:
                    res.per_lab[lab] = {}
                res.per_lab[lab][f] = float(avg)
            total_avg = float(np.average(avg_list))
            res.totals[f] = total_avg
        return res
