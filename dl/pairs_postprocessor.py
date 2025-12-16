from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm
from itertools import product

from common.config_utils import base_model_to_file
from common.helpers import get_annotation_by_video_meta, get_train_meta
from common.metrics_common import calc_best_f1_threshold
from common.parse_utils import parse_behaviors_labeled_from_row
from dl.metrics import ProdMetricCompHelperOnline


def _fit_action_thresholds_on_folds(
    predictions,
    gt_per_pred: dict[tuple[int, int, int, str], np.ndarray],
    *,
    train_folds: set[int],
    desc: str,
) -> dict[tuple[str, str], float]:
    """
    Returns thresholds[(lab, action)] fitted on *train_folds* only using calc_best_f1_threshold.
    """
    probs_by_key: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    gt_by_key: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)

    for pred in predictions:
        f = pred.fold
        if f is None or int(f) not in train_folds:
            continue
        key = (pred.lab_name, pred.behavior.action)
        probs_by_key[key].append(np.asarray(pred.probs, dtype=np.float32))
        gt_by_key[key].append(np.asarray(gt_per_pred[pred.key()], dtype=np.int8))

    keys = list(probs_by_key.keys())
    out: dict[tuple[str, str], float] = {}

    for k in tqdm(keys, desc=desc, leave=False):
        y_pred = np.concatenate(probs_by_key[k], axis=0).astype(np.float32, copy=False)
        y_true = np.concatenate(gt_by_key[k], axis=0).astype(np.int8, copy=False)

        # Optional safety for degenerate labels (keep if you want):
        # if y_true.sum() == 0:
        #     out[k] = float(y_pred.max() + 1e-6)
        #     continue

        out[k] = calc_best_f1_threshold(y_true=y_true, y_pred=y_pred)

    return out


def _thresholds_for_group(
    g,
    thresholds_by_action: dict[tuple[str, str], float] | None,
) -> np.ndarray:
    """
    Build (K,) thresholds for this group from thresholds_by_action[(lab, action)].
    Falls back to g.thresholds[j] when missing.
    """
    if thresholds_by_action is None:
        return g.thresholds
    thr = np.asarray(g.thresholds, dtype=np.float32).copy()
    for j, action in enumerate(g.actions):
        thr[j] = float(thresholds_by_action.get((g.lab, action), thr[j]))
    return thr


# ----------------------------
# Pairwise resolver training
# ----------------------------

_PAIRKEY_RE = re.compile(r"lab='(?P<lab>[^']*)'\s+a='(?P<a>[^']*)'\s+b='(?P<b>[^']*)'")


@dataclass(frozen=True)
class PairKey:
    lab: str = ""
    a: str = ""  # action name, lexicographically smaller
    b: str = ""  # action name, lexicographically larger


@dataclass
class PairwiseResolverParams:
    """
    Directed compare rule:
      choose i over j  iff  (prob_i - prob_j) > t[(lab,i,j)]
    where t[(lab,i,j)] = +raw_t[(lab,min(i,j),max(i,j))] or its negation.

    'raw_t' is stored only for (a<b). Counts stored per (a<b).
    """

    alpha: float = 0.0
    min_samples: int = sys.maxsize
    raw_t: dict[PairKey, float] = field(
        default_factory=dict
    )  # (a<b) -> threshold on (prob_a - prob_b)
    raw_n: dict[PairKey, int] = field(
        default_factory=dict
    )  # (a<b) -> number of training samples used

    def directed_threshold(self, lab: str, i: str, j: str) -> float:
        if i == j:
            return 0.0
        a, b = (i, j) if i < j else (j, i)
        key = PairKey(lab=lab, a=a, b=b)
        n = self.raw_n.get(key, 0)
        if n < self.min_samples:
            return 0.0  # fallback to argmax
        t = self.raw_t.get(key, 0.0) * self.alpha
        return t if (i < j) else -t

    @staticmethod
    def _encode_key(k: PairKey) -> str:
        return f"{k.lab}|{k.a}|{k.b}"

    @staticmethod
    def _decode_key(s: str) -> PairKey:
        s = s.strip()
        if "|" in s:
            lab, a, b = s.split("|", 2)
            return PairKey(lab=lab, a=a, b=b)
        m = _PAIRKEY_RE.fullmatch(s)
        if m:
            return PairKey(**m.groupdict())
        raise ValueError(f"Bad PairKey string: {s!r}")

    def to_json_file(self, file_path: str | Path) -> None:
        p = Path(file_path)
        data = {
            "alpha": self.alpha,
            "min_samples": self.min_samples,
            "raw_t": {self._encode_key(k): v for k, v in self.raw_t.items()},
            "raw_n": {self._encode_key(k): v for k, v in self.raw_n.items()},
        }
        p.write_text(json.dumps(data, indent=2, sort_keys=True))

    @classmethod
    def from_json_file(cls, file_path: str | Path) -> PairwiseResolverParams:
        p = Path(file_path)
        d = json.loads(p.read_text())

        def parse_map(m, cast):
            if not m:
                return {}
            if isinstance(m, dict):
                return {cls._decode_key(k): cast(v) for k, v in m.items()}
            raise TypeError(f"Expected object for raw_*; got {type(m).__name__}")

        return cls(
            alpha=float(d.get("alpha", 0.0)),
            min_samples=int(d.get("min_samples", sys.maxsize)),
            raw_t=parse_map(d.get("raw_t", {}), float),
            raw_n=parse_map(d.get("raw_n", {}), int),
        )


@dataclass
class GroupData:
    video_id: int
    agent: int
    target: int
    fold: int
    lab: str
    actions: list[str]  # aligned with columns
    thresholds: np.ndarray  # (K,)
    probs: np.ndarray  # (T,K)
    gt: np.ndarray  # (T,K), 0/1


@dataclass
class PairSample:
    fold: int
    lab: str
    a: str
    b: str
    delta: float  # prob_a - prob_b where a<b
    y: int  # 1 if a is the true action, 0 if b is the true action


def _build_gt_per_pred(predictions) -> dict[tuple[int, int, int, str], np.ndarray]:
    """
    Uses your existing pipeline helpers:
      - get_train_meta()
      - parse_behaviors_labeled_from_row(...)
      - get_annotation_by_video_meta(...)
    """
    meta = get_train_meta()

    gt_per_pred: dict[tuple[int, int, int, str], np.ndarray] = {}
    for video_meta in tqdm(
        meta.to_dict(orient="records"), desc="Building gt from meta.csv..."
    ):
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
                # In case meta parsing missed something, be defensive.
                gt_per_pred[key] = np.zeros(cnt_frames, dtype=np.int8)
            l = int(annot_row["start_frame"])
            r = int(annot_row["stop_frame"])
            gt_per_pred[key][l:r] = 1

    # (Optional) sanity: ensure every pred has gt
    missing = 0
    for pred in predictions:
        if pred.key() not in gt_per_pred:
            missing += 1
    if missing:
        print(
            f"[WARN] Missing GT for {missing} predictions (they will crash if evaluated)."
        )

    return gt_per_pred


def _group_predictions_to_groupdata(
    predictions, gt_per_pred: dict[tuple[int, int, int, str], np.ndarray]
) -> list[GroupData]:
    groups = defaultdict(list)
    for pred in predictions:
        groups[(pred.video_id, pred.behavior.agent, pred.behavior.target)].append(pred)

    out: list[GroupData] = []
    for (video_id, agent, target), preds_list in tqdm(
        groups.items(), desc="Preparing group tensors..."
    ):
        # Assume consistent fold & lab per group (common in OOF video splits). Be defensive if not.
        folds = [p.fold for p in preds_list if p.fold is not None]
        fold = int(folds[0]) if folds and all(f == folds[0] for f in folds) else -1

        labs = [p.lab_name for p in preds_list]
        lab = labs[0] if all(x == labs[0] for x in labs) else labs[0]  # keep first

        # Ensure unique actions
        actions = [p.behavior.action for p in preds_list]
        if len(actions) != len(set(actions)):
            # If duplicates exist, keep the first occurrence.
            seen = set()
            new_preds = []
            for p in preds_list:
                if p.behavior.action not in seen:
                    new_preds.append(p)
                    seen.add(p.behavior.action)
            preds_list = new_preds
            actions = [p.behavior.action for p in preds_list]

        # Build matrices
        probs = np.stack(
            [np.asarray(p.probs, dtype=np.float32) for p in preds_list], axis=1
        )  # (T,K)
        thresholds = np.asarray(
            [float(p.threshold) for p in preds_list], dtype=np.float32
        )  # (K,)

        # GT aligned to the same columns
        gt_cols = []
        for p in preds_list:
            gt_cols.append(np.asarray(gt_per_pred[p.key()], dtype=np.int8))
        gt = np.stack(gt_cols, axis=1)  # (T,K)

        out.append(
            GroupData(
                video_id=video_id,
                agent=int(agent),
                target=int(target),
                fold=fold,
                lab=lab,
                actions=actions,
                thresholds=thresholds,
                probs=probs,
                gt=gt,
            )
        )
    return out


def _evaluate_independent_f1(predictions, gt_per_pred) -> float:
    helper = ProdMetricCompHelperOnline()
    for pred in tqdm(predictions, desc="Independent f1 (no 1-action constraint)..."):
        helper.update(
            pred=(np.asarray(pred.probs) >= float(pred.threshold)),
            gt=gt_per_pred[pred.key()],
            lab=pred.lab_name,
            action=pred.behavior.action,
        )
    return float(helper.finalise())


def _select_argmax_one_action(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    probs: (T,K), thresholds: (K,)
    Returns chosen: (T,) int, -1 means 'emit nothing'
    """
    T, K = probs.shape
    masked = probs.copy()
    for j in range(K):
        masked[:, j] = np.where(masked[:, j] < thresholds[j], -np.inf, masked[:, j])

    best = np.argmax(masked, axis=1)  # (T,)
    best_val = masked[np.arange(T), best]
    chosen = np.where(np.isfinite(best_val), best, -1).astype(np.int32)
    return chosen


def _select_pairwise_tournament(
    probs: np.ndarray,
    thresholds: np.ndarray,
    actions: list[str],
    lab: str,
    params: PairwiseResolverParams,
) -> np.ndarray:
    """
    Pairwise tournament among candidates above thresholds.
    Returns chosen: (T,) int, -1 means none.
    """
    T, K = probs.shape
    active = probs >= thresholds[None, :]

    # Precompute directed threshold matrix for this group's actions
    tmat = np.zeros((K, K), dtype=np.float32)
    for i, ai in enumerate(actions):
        for j, aj in enumerate(actions):
            if i == j:
                continue
            tmat[i, j] = float(params.directed_threshold(lab=lab, i=ai, j=aj))

    chosen = np.full((T,), -1, dtype=np.int32)

    for t in range(T):
        cands = np.flatnonzero(active[t])
        if cands.size == 0:
            continue
        if cands.size == 1:
            chosen[t] = int(cands[0])
            continue

        p = probs[t, cands].astype(np.float32)  # (C,)
        diff = p[:, None] - p[None, :]  # (C,C)
        tsub = tmat[np.ix_(cands, cands)]  # (C,C)
        wins = (diff > tsub).sum(axis=1)  # (C,)

        maxw = wins.max()
        best_local = np.flatnonzero(wins == maxw)
        if best_local.size > 1:
            # tie-break by highest prob
            pp = p[best_local]
            best_local = best_local[np.argmax(pp)][None]
        chosen[t] = int(cands[int(best_local[0])])

    return chosen


def _update_helper_from_group_choice(
    helper: ProdMetricCompHelperOnline,
    group: GroupData,
    chosen: np.ndarray,  # (T,), -1 means none
):
    for j, action in enumerate(group.actions):
        pred_j = chosen == j
        helper.update(pred=pred_j, gt=group.gt[:, j], lab=group.lab, action=action)


def _evaluate_groups_argmax(groups: list[GroupData], fold: int | None = None) -> float:
    helper = ProdMetricCompHelperOnline()
    for g in groups:
        if fold is not None and g.fold != fold:
            continue
        chosen = _select_argmax_one_action(g.probs, g.thresholds)
        _update_helper_from_group_choice(helper, g, chosen)
    return float(helper.finalise())


def _extract_pair_samples(
    groups: list["GroupData"],
    *,
    use_ge2_active: bool = False,
    thresholds_by_action: dict[tuple[str, str], float] | None = None,
    desc: str = "Extracting pairwise samples...",
) -> list["PairSample"]:
    """
    Same as before, but `active` is computed using refit thresholds if provided.
    """
    samples: list[PairSample] = []

    for g in tqdm(groups, desc=desc, leave=False):
        if g.fold < 0:
            continue  # need real folds for OOF

        probs = g.probs  # (T,K)
        thr = _thresholds_for_group(g, thresholds_by_action)  # (K,)
        gt = g.gt  # (T,K) 0/1

        active = probs >= thr[None, :]  # (T,K) bool
        active_cnt = active.sum(axis=1)  # (T,)
        gt_sum = gt.sum(axis=1)  # (T,)

        frames_single_gt = np.flatnonzero(gt_sum == 1)
        if frames_single_gt.size == 0:
            continue

        gt_idx_all = np.argmax(gt, axis=1).astype(np.int32)  # (T,)
        gt_is_active = active[np.arange(gt.shape[0]), gt_idx_all]

        if use_ge2_active:
            frames = np.flatnonzero((active_cnt >= 2) & (gt_sum == 1) & gt_is_active)
            if frames.size == 0:
                continue

            mask = active[frames]  # (F,K)
            rr, cc = np.nonzero(mask)
            gt_cols = gt_idx_all[frames][rr]

            keep = cc != gt_cols
            rr = rr[keep]
            cc = cc[keep]
            gt_cols = gt_cols[keep]

            for r, j, gi in zip(rr, cc, gt_cols):
                t = int(frames[int(r)])
                j = int(j)
                gi = int(gi)

                gname = g.actions[gi]
                jname = g.actions[j]

                pg = float(probs[t, gi])
                pj = float(probs[t, j])

                if gname < jname:
                    a, b = gname, jname
                    delta = pg - pj
                    yv = 1
                else:
                    a, b = jname, gname
                    delta = pj - pg
                    yv = 0

                samples.append(
                    PairSample(
                        fold=g.fold, lab=g.lab, a=a, b=b, delta=float(delta), y=int(yv)
                    )
                )
        else:
            frames2 = np.flatnonzero((active_cnt == 2) & (gt_sum == 1) & gt_is_active)
            if frames2.size == 0:
                continue

            rr, cc = np.nonzero(active[frames2])
            cc = cc.reshape(-1, 2)

            cand0 = cc[:, 0]
            cand1 = cc[:, 1]

            gt2 = gt[frames2, :]
            true_in_cands = (
                gt2[np.arange(frames2.size), cand0]
                + gt2[np.arange(frames2.size), cand1]
            ) == 1
            frames2 = frames2[true_in_cands]
            cc = cc[true_in_cands]
            if frames2.size == 0:
                continue

            p0 = probs[frames2, cc[:, 0]]
            p1 = probs[frames2, cc[:, 1]]
            y0 = gt[frames2, cc[:, 0]].astype(np.int32)

            for k in range(frames2.size):
                i = int(cc[k, 0])
                j = int(cc[k, 1])
                ai = g.actions[i]
                aj = g.actions[j]

                if ai < aj:
                    a, b = ai, aj
                    delta = float(p0[k] - p1[k])
                    yv = int(y0[k])
                else:
                    a, b = aj, ai
                    delta = float(p1[k] - p0[k])
                    yv = int(1 - y0[k])

                samples.append(
                    PairSample(fold=g.fold, lab=g.lab, a=a, b=b, delta=delta, y=int(yv))
                )

    return samples


def _fit_threshold_for_pair(deltas: np.ndarray, y: np.ndarray) -> float:
    """
    Fit t for rule: predict a if (delta > t), else b
      delta = prob_a - prob_b  (a<b)
      y = 1 if a is true, 0 if b is true

    Objective: maximize average of local F1(a) and F1(b) on this pair's samples.
    """
    deltas = np.asarray(deltas, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    n = deltas.size
    if n == 0:
        return 0.0

    order = np.argsort(deltas)
    d = deltas[order]
    yy = y[order]

    total_a = int(yy.sum())
    total_b = int(n - total_a)

    # suffix sums for y (true a) to compute tp_a when predicting 'a' for indices > k
    suf_true_a = np.concatenate([[0], np.cumsum(yy[::-1])]).astype(np.int32)  # len n+1
    # suf_true_a[m] = sum of last m elements of yy; so for cut at k -> region (k+1..n-1) has size m=n-(k+1)
    # We'll iterate k in [-1..n-1], where predicted_a = indices > k.
    best_score = -1.0
    best_t = 0.0

    # Helper to compute f1 safely
    def f1(tp: int, fp: int, fn: int) -> float:
        denom = 2 * tp + fp + fn
        return 0.0 if denom == 0 else (2.0 * tp / denom)

    # Consider k=-1 => predict_a for all
    for k in range(-1, n):
        # predicted_a indices: (k+1..n-1)
        m = n - (k + 1)
        tp_a = int(suf_true_a[m])
        fp_a = int(m - tp_a)
        fn_a = int(total_a - tp_a)

        # predicted_b for the rest: (0..k)
        tp_b = int((k + 1) - (total_a - tp_a))  # predicted_b & true_b
        # safer:
        true_a_in_pred_b = int(total_a - tp_a)
        tp_b = int((k + 1) - true_a_in_pred_b)
        fp_b = int(true_a_in_pred_b)
        fn_b = int(total_b - tp_b)

        score = 0.5 * (f1(tp_a, fp_a, fn_a) + f1(tp_b, fp_b, fn_b))

        # choose a threshold value corresponding to this cut
        # if k == -1 -> t = -inf (predict a always); use very small
        # if k == n-1 -> t = +inf (predict b always); use very large
        if score > best_score:
            best_score = score
            if k == -1:
                best_t = float(d[0] - 1e-6)
            elif k == n - 1:
                best_t = float(d[-1] + 1e-6)
            else:
                best_t = float(0.5 * (d[k] + d[k + 1]))

    return best_t


def _fit_pairwise_params_from_samples(
    samples: Iterable[PairSample],
) -> tuple[dict[PairKey, float], dict[PairKey, int]]:
    by_pair_delta: dict[PairKey, list[float]] = defaultdict(list)
    by_pair_y: dict[PairKey, list[int]] = defaultdict(list)

    for s in samples:
        key = PairKey(lab=s.lab, a=s.a, b=s.b)
        by_pair_delta[key].append(s.delta)
        by_pair_y[key].append(s.y)

    raw_t: dict[PairKey, float] = {}
    raw_n: dict[PairKey, int] = {}

    for key in by_pair_delta.keys():
        deltas = np.asarray(by_pair_delta[key], dtype=np.float32)
        y = np.asarray(by_pair_y[key], dtype=np.int32)
        raw_n[key] = int(deltas.size)
        raw_t[key] = float(_fit_threshold_for_pair(deltas=deltas, y=y))

    return raw_t, raw_n


def maximise_f1_one_action_pairwise_oof(
    predictions,
    *,
    alpha_grid: tuple[float, ...] = (0.75, 1.0),
    min_samples_grid: tuple[int, ...] = (5, 10, 20),
    use_ge2_active_samples: bool = False,
    save_json_path: str | None = None,
) -> PairwiseResolverParams:
    """
    Clean CV:
      For each val fold:
        1) fit per-(lab, action) thresholds on the 4 train folds
        2) fit pairwise deltas on train folds using those thresholds
        3) apply thresholds + deltas to val fold

    Prints baseline CV first: argmax with refit thresholds, averaged over 5 folds.

    For each (alpha, min_samples) prints per-fold F1 and average.
    """
    print("Building GT...")
    gt_per_pred = _build_gt_per_pred(predictions)

    print("Preparing group tensors...")
    groups = _group_predictions_to_groupdata(predictions, gt_per_pred)

    folds = sorted({g.fold for g in groups if g.fold >= 0})
    if not folds:
        print(
            "[WARN] No fold info found in predictions. Will train on all data and return in-sample params."
        )
        folds = [-1]

    # Keep original diagnostics unchanged
    ind_f1 = _evaluate_independent_f1(predictions, gt_per_pred)
    print(f"Independent f1 (actions independent): {ind_f1:.6f}")

    argmax_f1_full = _evaluate_groups_argmax(groups, fold=None)
    print(f"Argmax f1 (1 action/frame, full): {argmax_f1_full:.6f}")

    if folds == [-1]:
        # no OOF possible
        pair_samples = _extract_pair_samples(
            groups,
            use_ge2_active=use_ge2_active_samples,
            thresholds_by_action=None,
            desc="Extracting pairwise samples (no folds)...",
        )
        raw_t, raw_n = _fit_pairwise_params_from_samples(pair_samples)
        params = PairwiseResolverParams(
            alpha=1.0,
            min_samples=min(min_samples_grid) if min_samples_grid else 0,
            raw_t=raw_t,
            raw_n=raw_n,
        )
        print("[WARN] No OOF evaluation done (missing folds). Returning fitted params.")
        return params

    # ---------------------------------------------------------
    # Baseline CV: refit thresholds on 4 folds, apply on val fold (argmax)
    # ---------------------------------------------------------
    thr_by_val_fold: dict[int, dict[tuple[str, str], float]] = {}
    baseline_scores: list[float] = []

    for val_fold in tqdm(folds, desc="Baseline CV (refit thresholds + argmax)"):
        train_folds = {f for f in folds if f != val_fold}
        thr_map = _fit_action_thresholds_on_folds(
            predictions,
            gt_per_pred,
            train_folds=train_folds,
            desc=f"Fitting thresholds (train folds != {val_fold})",
        )
        thr_by_val_fold[val_fold] = thr_map

        helper = ProdMetricCompHelperOnline()
        for g in groups:
            if g.fold != val_fold:
                continue
            thr = _thresholds_for_group(g, thr_map)
            chosen = _select_argmax_one_action(g.probs, thr)
            _update_helper_from_group_choice(helper, g, chosen)

        score = float(helper.finalise())
        baseline_scores.append(score)

    baseline_avg = float(np.mean(baseline_scores)) if baseline_scores else 0.0
    baseline_str = ", ".join(f"{f}:{s:.6f}" for f, s in zip(folds, baseline_scores))
    print(
        f"Baseline CV (refit thresholds, argmax) folds=[{baseline_str}] avg={baseline_avg:.6f}"
    )

    # ---------------------------------------------------------
    # Precompute per-val-fold pairwise raw_t/raw_n
    # (trained on 4 folds, using thresholds fitted on those 4 folds)
    # ---------------------------------------------------------
    raw_by_val_fold: dict[int, tuple[dict[PairKey, float], dict[PairKey, int]]] = {}

    for val_fold in tqdm(folds, desc="Fitting pairwise deltas per validation fold"):
        thr_map = thr_by_val_fold[val_fold]
        train_groups = [g for g in groups if g.fold != val_fold]

        train_samples = _extract_pair_samples(
            train_groups,
            use_ge2_active=use_ge2_active_samples,
            thresholds_by_action=thr_map,
            desc=f"Extracting pair samples (train folds != {val_fold})",
        )
        raw_by_val_fold[val_fold] = _fit_pairwise_params_from_samples(train_samples)

    # ---------------------------------------------------------
    # Grid-search alpha + min_samples (report per fold + avg)
    # ---------------------------------------------------------
    best_score = -1.0
    best_alpha = alpha_grid[0]
    best_min_samples = min_samples_grid[0] if min_samples_grid else 0

    combos = list(product(alpha_grid, min_samples_grid))
    for alpha, min_s in tqdm(combos, desc="Grid-search (alpha, min_samples)"):
        fold_scores: list[float] = []

        for val_fold in folds:
            thr_map = thr_by_val_fold[val_fold]
            raw_t, raw_n = raw_by_val_fold[val_fold]
            fold_params = PairwiseResolverParams(
                alpha=float(alpha), min_samples=int(min_s), raw_t=raw_t, raw_n=raw_n
            )

            helper = ProdMetricCompHelperOnline()
            for g in groups:
                if g.fold != val_fold:
                    continue
                thr = _thresholds_for_group(g, thr_map)
                chosen = _select_pairwise_tournament(
                    probs=g.probs,
                    thresholds=thr,
                    actions=g.actions,
                    lab=g.lab,
                    params=fold_params,
                )
                _update_helper_from_group_choice(helper, g, chosen)

            fold_scores.append(float(helper.finalise()))

        avg = float(np.mean(fold_scores)) if fold_scores else 0.0
        folds_str = ", ".join(f"{f}:{s:.6f}" for f, s in zip(folds, fold_scores))
        print(
            f"[CV] alpha={alpha:.2f}, min_samples={min_s:4d} folds=[{folds_str}] avg={avg:.6f}"
        )

        if avg > best_score:
            best_score = avg
            best_alpha = float(alpha)
            best_min_samples = int(min_s)

    print(
        f"Best OOF: avg_f1={best_score:.6f} (alpha={best_alpha:.2f}, min_samples={best_min_samples})"
    )

    # ---------------------------------------------------------
    # Refit on ALL folds with best hyperparams, using thresholds refit on ALL folds
    # ---------------------------------------------------------
    thr_all = _fit_action_thresholds_on_folds(
        predictions,
        gt_per_pred,
        train_folds=set(folds),
        desc="Fitting thresholds (ALL folds)",
    )

    pair_samples_all = _extract_pair_samples(
        [g for g in groups if g.fold >= 0],
        use_ge2_active=use_ge2_active_samples,
        thresholds_by_action=thr_all,
        desc="Extracting pair samples (ALL folds)",
    )

    raw_t_all, raw_n_all = _fit_pairwise_params_from_samples(pair_samples_all)
    final_params = PairwiseResolverParams(
        alpha=best_alpha,
        min_samples=best_min_samples,
        raw_t=raw_t_all,
        raw_n=raw_n_all,
    )

    if save_json_path is not None:
        final_params.to_json_file(save_json_path)
        print(f"Saved params to: {save_json_path}")

    return final_params
