from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
from tqdm import tqdm

from common.helpers import get_annotation_by_video_meta, get_train_meta
from common.parse_utils import parse_behaviors_labeled_from_row
from dl.metrics import ProdMetricCompHelperOnline

# --------- calibration primitives ---------


@dataclass(frozen=True)
class ActionKey:
    lab: str
    action: str


@dataclass
class BinnedIsotonicCalibrator:
    """
    Monotone (non-decreasing) calibration map p -> p_cal, built from quantile bins + PAVA.
    """

    x: np.ndarray  # strictly increasing, shape (M,)
    y: np.ndarray  # shape (M,)

    def __call__(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float32)
        out = np.interp(p, self.x, self.y, left=self.y[0], right=self.y[-1]).astype(
            np.float32
        )
        return np.clip(out, 0.0, 1.0)


def _pava_non_decreasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n = y.size
    if n == 0:
        return y.astype(np.float32)

    means = []
    weights = []
    counts = []
    for yi, wi in zip(y, w):
        means.append(float(yi))
        weights.append(float(wi))
        counts.append(1)
        while len(means) >= 2 and means[-2] > means[-1]:
            m1, w1, c1 = means[-2], weights[-2], counts[-2]
            m2, w2, c2 = means[-1], weights[-1], counts[-1]
            ww = w1 + w2
            mm = (m1 * w1 + m2 * w2) / ww if ww > 0 else 0.5 * (m1 + m2)
            means[-2], weights[-2], counts[-2] = mm, ww, c1 + c2
            means.pop()
            weights.pop()
            counts.pop()

    out = np.empty(n, dtype=np.float64)
    idx = 0
    for m, c in zip(means, counts):
        out[idx : idx + c] = m
        idx += c
    return out.astype(np.float32)


def _merge_duplicate_x(
    x: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # assumes x sorted non-decreasing
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    xs, ys, ws = [], [], []
    i = 0
    n = x.size
    while i < n:
        j = i + 1
        while j < n and x[j] == x[i]:
            j += 1
        ww = w[i:j].sum()
        yy = (y[i:j] * w[i:j]).sum() / ww if ww > 0 else float(y[i:j].mean())
        xs.append(float(x[i]))
        ys.append(float(yy))
        ws.append(float(ww))
        i = j

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(ws, dtype=np.float32),
    )


def fit_binned_isotonic_calibrator(
    p: np.ndarray,
    y: np.ndarray,
    *,
    nbins: int = 50,
    min_samples: int = 500,
) -> BinnedIsotonicCalibrator:
    p = np.asarray(p, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    n = p.size
    if n < max(10, min_samples):
        return BinnedIsotonicCalibrator(
            x=np.asarray([0.0, 1.0], np.float32), y=np.asarray([0.0, 1.0], np.float32)
        )

    order = np.argsort(p)
    p = p[order]
    y = y[order]

    nb = int(min(max(2, nbins), n))  # at least 2 bins, at most n
    cuts = np.linspace(0, n, nb + 1, dtype=np.int32)

    bx = np.empty(nb, dtype=np.float32)
    by = np.empty(nb, dtype=np.float32)
    bw = np.empty(nb, dtype=np.float32)

    for i in range(nb):
        l, r = int(cuts[i]), int(cuts[i + 1])
        if r <= l:
            # should not happen with linspace, but be defensive
            bx[i] = float(p[min(l, n - 1)])
            by[i] = float(y[min(l, n - 1)])
            bw[i] = 1.0
            continue
        ps = p[l:r]
        ys = y[l:r]
        bx[i] = float(ps.mean())
        by[i] = float(ys.mean())
        bw[i] = float(r - l)

    # ensure strictly increasing x (merge duplicates)
    bx, by, bw = _merge_duplicate_x(bx, by, bw)

    # enforce monotone y via PAVA
    by_iso = _pava_non_decreasing(by, bw)

    # add endpoints to make interpolation stable
    x = np.concatenate([[0.0], bx, [1.0]]).astype(np.float32)
    yv = np.concatenate([[by_iso[0]], by_iso, [by_iso[-1]]]).astype(np.float32)

    # guarantee strict x increasing for np.interp
    eps = 1e-7
    x = np.maximum.accumulate(x + eps * np.arange(x.size, dtype=np.float32))

    return BinnedIsotonicCalibrator(x=x, y=yv)


def fit_best_f1_threshold(p: np.ndarray, y: np.ndarray) -> float:
    """
    Choose threshold t for rule pred = (p >= t) that maximizes F1 for this (action,lab) independently.
    """
    p = np.asarray(p, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    n = p.size
    if n == 0:
        return 1.0

    pos = int(y.sum())
    if pos == 0:
        return 1.0
    if pos == n:
        return 0.0

    order = np.argsort(p)[::-1]
    ps = p[order]
    ys = y[order]

    tp = np.cumsum(ys, dtype=np.int64)
    fp = np.cumsum(1 - ys, dtype=np.int64)
    fn = pos - tp

    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, (2.0 * tp / denom).astype(np.float32), 0.0)

    best_i = int(np.argmax(f1))
    t = float(ps[best_i] - 1e-12)  # so that this point stays predicted positive with >=
    return float(np.clip(t, 0.0, 1.0))


# --------- decoding + evaluation helpers (1 action per frame) ---------


def _select_argmax_one_action(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    probs: (T,K), thresholds: (K,)
    returns chosen: (T,) int, -1 means emit nothing
    """
    T, K = probs.shape
    masked = probs.copy()
    for j in range(K):
        masked[:, j] = np.where(masked[:, j] < thresholds[j], -np.inf, masked[:, j])
    best = np.argmax(masked, axis=1)
    best_val = masked[np.arange(T), best]
    return np.where(np.isfinite(best_val), best, -1).astype(np.int32)


def _update_helper_from_choice(
    helper, lab: str, actions: list[str], gt: np.ndarray, chosen: np.ndarray
):
    for j, action in enumerate(actions):
        helper.update(pred=(chosen == j), gt=gt[:, j], lab=lab, action=action)


# --------- main function ---------


@dataclass
class CalibThresholdModel:
    calibrators: dict[ActionKey, BinnedIsotonicCalibrator]
    thresholds: dict[ActionKey, float]
    nbins: int
    min_samples: int


def optimise_f1_oof_calibrate_probs_and_refit_thresholds(
    predictions,
    *,
    nbins: int = 50,
    min_samples_calib: int = 500,
    show_topn_thresholds: int = 10,
) -> CalibThresholdModel:
    """
    OOF procedure (5 folds assumed):
      - baseline OOF F1 (raw probs + original thresholds + argmax constraint)
      - tuned OOF F1: per (lab,action) fit prob-calibrator on 4 folds, apply it,
        then refit per (lab,action) threshold on calibrated probs (train folds),
        evaluate on the held-out fold with argmax constraint.

    Also fits on the whole dataset and reports full-data F1.

    Requires you already have:
      - get_train_meta()
      - parse_behaviors_labeled_from_row(...)
      - get_annotation_by_video_meta(...)
      - ProdMetricCompHelperOnline
      - PredictedProbsForBehavior with .key(), .fold, .lab_name, .behavior.action, .threshold, .probs
    """

    # --------- build GT ---------
    print("Building GT from meta/annotations...")
    meta = get_train_meta()
    gt_per_pred = {}

    for video_meta in tqdm(meta.to_dict(orient="records"), desc="Building gt_per_pred"):
        video_id = int(video_meta["video_id"])
        cnt_frames = int(video_meta["cnt_frames"])
        behs = parse_behaviors_labeled_from_row(row=video_meta)
        for beh in behs:
            key = (video_id, beh.agent, beh.target, beh.action)
            if key not in gt_per_pred:
                gt_per_pred[key] = np.zeros(cnt_frames, dtype=np.int8)

        annot = get_annotation_by_video_meta(video_meta=video_meta)
        for annot_row in annot.to_dict(orient="records"):
            key = (
                video_id,
                int(annot_row["agent_id"]),
                int(annot_row["target_id"]),
                annot_row["action"],
            )
            if key not in gt_per_pred:
                gt_per_pred[key] = np.zeros(cnt_frames, dtype=np.int8)
            l = int(annot_row["start_frame"])
            r = int(annot_row["stop_frame"])
            gt_per_pred[key][l:r] = 1

    # --------- group predictions into (video, agent, target) groups ---------
    groups = defaultdict(list)
    for pred in predictions:
        groups[(pred.video_id, pred.behavior.agent, pred.behavior.target)].append(pred)

    group_list = []
    for (video_id, agent, target), preds_list in tqdm(
        groups.items(), desc="Preparing group tensors"
    ):
        # fold / lab
        folds = [p.fold for p in preds_list if p.fold is not None]
        fold = int(folds[0]) if folds and all(f == folds[0] for f in folds) else -1
        lab = preds_list[0].lab_name

        # unique actions
        seen = set()
        uniq = []
        for p in preds_list:
            a = p.behavior.action
            if a not in seen:
                uniq.append(p)
                seen.add(a)
        preds_list = uniq

        actions = [p.behavior.action for p in preds_list]
        probs = np.stack(
            [np.asarray(p.probs, dtype=np.float32) for p in preds_list], axis=1
        )  # (T,K)
        thr0 = np.asarray(
            [float(p.threshold) for p in preds_list], dtype=np.float32
        )  # (K,)
        gt = np.stack(
            [np.asarray(gt_per_pred[p.key()], dtype=np.int8) for p in preds_list],
            axis=1,
        )  # (T,K)

        group_list.append(
            dict(
                video_id=int(video_id),
                agent=int(agent),
                target=int(target),
                fold=int(fold),
                lab=str(lab),
                actions=actions,
                probs=probs,
                thr0=thr0,
                gt=gt,
            )
        )

    folds = sorted({g["fold"] for g in group_list if g["fold"] >= 0})
    if not folds:
        raise ValueError(
            "No valid fold info found (pred.fold is None or -1 everywhere)."
        )

    print(f"Found folds: {folds}")

    # --------- baseline OOF (no tuning) ---------
    print("\nBaseline OOF (raw probs + original thresholds + argmax)...")
    baseline_fold_scores = []
    for f in folds:
        helper = ProdMetricCompHelperOnline()
        val_groups = [g for g in group_list if g["fold"] == f]
        for g in tqdm(val_groups, desc=f"[baseline] fold={f}", leave=False):
            chosen = _select_argmax_one_action(g["probs"], g["thr0"])
            _update_helper_from_choice(helper, g["lab"], g["actions"], g["gt"], chosen)
        score = float(helper.finalise())
        baseline_fold_scores.append(score)
        print(f"  fold {f}: f1={score:.6f}")

    baseline_mean = float(np.mean(baseline_fold_scores))
    print(f"Baseline mean across folds: {baseline_mean:.6f}")

    # --------- tuned OOF: calibrate + refit thresholds ---------
    print(
        "\nTuned OOF (calibrate probs per (lab,action) on train folds + refit thresholds)..."
    )
    tuned_fold_scores = []

    for f in folds:
        train_groups = [g for g in group_list if g["fold"] != f and g["fold"] >= 0]
        val_groups = [g for g in group_list if g["fold"] == f]

        # collect raw (p,y) per (lab,action) from train
        p_map = defaultdict(list)
        y_map = defaultdict(list)

        for g in tqdm(train_groups, desc=f"[train collect] fold={f}", leave=False):
            lab = g["lab"]
            for j, action in enumerate(g["actions"]):
                key = ActionKey(lab=lab, action=action)
                p_map[key].append(g["probs"][:, j])
                y_map[key].append(g["gt"][:, j])

        # fit calibrators
        calibrators: dict[ActionKey, BinnedIsotonicCalibrator] = {}
        for key in tqdm(list(p_map.keys()), desc=f"[fit calib] fold={f}", leave=False):
            p = np.concatenate(p_map[key], axis=0)
            y = np.concatenate(y_map[key], axis=0)
            calibrators[key] = fit_binned_isotonic_calibrator(
                p, y, nbins=nbins, min_samples=min_samples_calib
            )

        # fit thresholds on calibrated train data
        thresholds: dict[ActionKey, float] = {}
        for key in tqdm(list(p_map.keys()), desc=f"[fit thr] fold={f}", leave=False):
            p = np.concatenate(p_map[key], axis=0)
            y = np.concatenate(y_map[key], axis=0)
            p_cal = calibrators[key](p)
            thresholds[key] = fit_best_f1_threshold(p_cal, y)

        # evaluate on val fold with argmax constraint
        helper = ProdMetricCompHelperOnline()
        for g in tqdm(val_groups, desc=f"[eval tuned] fold={f}", leave=False):
            lab = g["lab"]
            probs = g["probs"]
            K = probs.shape[1]

            probs_cal = np.empty_like(probs, dtype=np.float32)
            thr = g["thr0"].copy()

            for j, action in enumerate(g["actions"]):
                key = ActionKey(lab=lab, action=action)
                calib = calibrators.get(key, None)
                if calib is None:
                    probs_cal[:, j] = probs[:, j]
                else:
                    probs_cal[:, j] = calib(probs[:, j])
                if key in thresholds:
                    thr[j] = float(thresholds[key])

            chosen = _select_argmax_one_action(probs_cal, thr)
            _update_helper_from_choice(helper, lab, g["actions"], g["gt"], chosen)

        score = float(helper.finalise())
        tuned_fold_scores.append(score)
        print(
            f"  fold {f}: tuned f1={score:.6f} (baseline {baseline_fold_scores[folds.index(f)]:.6f})"
        )

        mean_so_far = float(np.mean(tuned_fold_scores))
        print(
            f"    partial tuned mean after {len(tuned_fold_scores)}/{len(folds)} folds: {mean_so_far:.6f}"
        )

    tuned_mean = float(np.mean(tuned_fold_scores))
    print(
        f"Tuned mean across folds: {tuned_mean:.6f}  (baseline mean {baseline_mean:.6f})"
    )

    # --------- fit on ALL data (in-sample) ---------
    print("\nFit calibration + thresholds on FULL dataset and evaluate FULL f1...")
    p_map_all = defaultdict(list)
    y_map_all = defaultdict(list)
    for g in tqdm(group_list, desc="[full collect]"):
        lab = g["lab"]
        for j, action in enumerate(g["actions"]):
            key = ActionKey(lab=lab, action=action)
            p_map_all[key].append(g["probs"][:, j])
            y_map_all[key].append(g["gt"][:, j])

    calibrators_all: dict[ActionKey, BinnedIsotonicCalibrator] = {}
    thresholds_all: dict[ActionKey, float] = {}

    for key in tqdm(list(p_map_all.keys()), desc="[full fit calib]"):
        p = np.concatenate(p_map_all[key], axis=0)
        y = np.concatenate(y_map_all[key], axis=0)
        calibrators_all[key] = fit_binned_isotonic_calibrator(
            p, y, nbins=nbins, min_samples=min_samples_calib
        )

    # show some thresholds as we fit them
    for key in tqdm(list(p_map_all.keys()), desc="[full fit thr]"):
        p = np.concatenate(p_map_all[key], axis=0)
        y = np.concatenate(y_map_all[key], axis=0)
        p_cal = calibrators_all[key](p)
        thresholds_all[key] = fit_best_f1_threshold(p_cal, y)

    # full-data evaluation
    helper_full = ProdMetricCompHelperOnline()
    for g in tqdm(group_list, desc="[full eval]"):
        lab = g["lab"]
        probs = g["probs"]
        K = probs.shape[1]

        probs_cal = np.empty_like(probs, dtype=np.float32)
        thr = g["thr0"].copy()

        for j, action in enumerate(g["actions"]):
            key = ActionKey(lab=lab, action=action)
            probs_cal[:, j] = (
                calibrators_all[key](probs[:, j])
                if key in calibrators_all
                else probs[:, j]
            )
            if key in thresholds_all:
                thr[j] = float(thresholds_all[key])

        chosen = _select_argmax_one_action(probs_cal, thr)
        _update_helper_from_choice(helper_full, lab, g["actions"], g["gt"], chosen)

    full_score = float(helper_full.finalise())
    print(f"FULL-data f1 (calib+thr fit on full): {full_score:.6f}")

    # Optional: show a few thresholds (largest changes vs original median, roughly)
    if show_topn_thresholds > 0:
        items = list(thresholds_all.items())
        print(
            f"\nExample learned thresholds (showing {min(show_topn_thresholds, len(items))}):"
        )
        for key, t in items[:show_topn_thresholds]:
            print(f"  lab={key.lab:>12s}  action={key.action:>24s}  thr={t:.4f}")

    return CalibThresholdModel(
        calibrators=calibrators_all,
        thresholds=thresholds_all,
        nbins=nbins,
        min_samples=min_samples_calib,
    )
