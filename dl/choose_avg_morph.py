from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pydantic import BaseModel, Field
from tqdm import tqdm

from common.helpers import get_annotation_by_video_meta, get_train_meta
from common.parse_utils import parse_behaviors_labeled_from_row
from common.submission_common import PredictedProbsForBehavior
from dl.metrics import ProdMetricCompHelperOnline

# ---------------------------------------------------------------------------
# Postprocessing params
# ---------------------------------------------------------------------------


class PostprocessingParams(BaseModel):
    """
    Postprocessing configuration.

    moving_average_window_30fps / morph_kernel_30fps are defined as window
    sizes (in frames) for a **30 fps** video. At a different FPS, they are
    rescaled and converted to an odd number of frames.

    - moving_average_window_30fps: int or None
    - morph_kernel_30fps: int or None
    - morph_first: whether to apply morph BEFORE averaging
    """

    moving_average_window_30fps: Optional[int] = Field(
        default=None,
        ge=1,
        description="Moving-average window length (frames) defined at 30fps.",
    )
    morph_kernel_30fps: Optional[int] = Field(
        default=None,
        ge=1,
        description="Morphological kernel length (frames) defined at 30fps.",
    )
    morph_first: bool = Field(
        default=False,
        description="If True, apply dilation+erosion before averaging; "
        "otherwise average first.",
    )

    class Config:
        frozen = True  # hashable & safe to use as dict keys

    # --- helpers for converting 30fps lengths -> odd length at given fps ---

    def _length_in_frames(
        self, base_len_30fps: Optional[int], fps: float
    ) -> Optional[int]:
        if base_len_30fps is None or base_len_30fps <= 0:
            return None
        scale = fps / 30.0
        k = int(round(base_len_30fps * scale))
        if k <= 1:
            return None
        if k % 2 == 0:  # must be odd
            k += 1
        return k

    def moving_average_window_frames(self, fps: float) -> Optional[int]:
        return self._length_in_frames(self.moving_average_window_30fps, fps)

    def morph_kernel_frames(self, fps: float) -> Optional[int]:
        return self._length_in_frames(self.morph_kernel_30fps, fps)

    # --- main application API ---

    def apply(self, probs: np.ndarray, fps: float) -> np.ndarray:
        """
        Apply the postprocessing to a 1D probability array.

        Returns a new float32 array of the same shape.
        """
        x = np.asarray(probs, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError(
                f"PostprocessingParams.apply expects a 1D array, got shape {x.shape}"
            )

        w = self.moving_average_window_frames(fps)
        k = self.morph_kernel_frames(fps)

        if w is None and k is None:
            # nothing to do
            return x

        def avg_op(arr: np.ndarray) -> np.ndarray:
            if w is None:
                return arr
            kernel = np.ones(w, dtype=np.float32) / float(w)
            # mode="same" preserves the probability array length
            return np.convolve(arr, kernel, mode="same")

        def morph_op(arr: np.ndarray) -> np.ndarray:
            """
            Grey-scale closing (dilation then erosion) with an odd window k.
            """
            if k is None:
                return arr

            pad = k // 2

            # dilation (max filter)
            padded = np.pad(arr, pad_width=pad, mode="edge")
            win = sliding_window_view(padded, k)
            dilated = win.max(axis=-1)

            # erosion (min filter)
            padded_d = np.pad(dilated, pad_width=pad, mode="edge")
            win2 = sliding_window_view(padded_d, k)
            eroded = win2.min(axis=-1)

            return eroded

        if self.morph_first:
            return avg_op(morph_op(x))

        return morph_op(avg_op(x))

    def apply_to_prediction(
        self, pred: "PredictedProbsForBehavior", fps: float
    ) -> np.ndarray:
        """
        Convenience wrapper: apply to a PredictedProbsForBehavior instance.
        """
        return self.apply(pred.probs, fps=fps)


def _params_to_dict(p: PostprocessingParams) -> Dict:
    """
    For JSON serialisation; pydantic v2 only.
    """
    return p.model_dump()


# ---------------------------------------------------------------------------
# Internal helper data structures
# ---------------------------------------------------------------------------


@dataclass
class _GroupRecord:
    """
    One (video, agent, target, action, lab, fold) sample of probabilities+gt
    used for grid search. All records in a group share (lab, action, fold).
    """

    probs: np.ndarray  # shape: (T,)
    gt: np.ndarray  # shape: (T,), 0/1
    threshold: float
    fps: float


GroupKey = Tuple[str, str, int]  # (lab, action, fold)


def _effective_threshold(pred: "PredictedProbsForBehavior") -> float:
    """
    Always use the global threshold from PredictedProbsForBehavior.
    """
    return float(pred.threshold)


# ---------------------------------------------------------------------------
# F1 utilities (delegate to ProdMetricCompHelperOnline)
# ---------------------------------------------------------------------------


def _compute_f1_from_counts(stats_param: Dict[str, Dict[str, Dict[str, int]]]) -> float:
    """
    stats_param[lab][action] = {'tp': int, 'fp': int, 'fn': int}

    Use ProdMetricCompHelperOnline.finalise() to compute the F1 so that
    the weighting matches the rest of the codebase.
    """

    helper = ProdMetricCompHelperOnline()
    helper.stats = stats_param  # type: ignore[attr-defined]
    return float(helper.finalise())


# ---------------------------------------------------------------------------
# Ground-truth construction
# ---------------------------------------------------------------------------


def _build_gt_per_pred() -> Dict[Tuple[int, int, int, str], np.ndarray]:
    """
    Rebuild gt_per_pred the same way as in analyse_segments_dl.
    """

    meta = get_train_meta()
    gt_per_pred: Dict[Tuple[int, int, int, str], np.ndarray] = {}

    for video_meta in tqdm(
        meta.to_dict(orient="records"), desc="Building gt_per_pred from meta.csv..."
    ):
        video_id = int(video_meta["video_id"])
        cnt_frames = int(video_meta["cnt_frames"])
        behs = parse_behaviors_labeled_from_row(row=video_meta)

        # Initialise all possible keys for this video
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
            assert key in gt_per_pred
            l = int(annot_row["start_frame"])
            r = int(annot_row["stop_frame"])
            gt_per_pred[key][l:r] = 1

    return gt_per_pred


# ---------------------------------------------------------------------------
# Baseline F1 computations
# ---------------------------------------------------------------------------


def _compute_baseline_f1_independent(
    predictions: Sequence["PredictedProbsForBehavior"],
    gt_per_pred: Dict[Tuple[int, int, int, str], np.ndarray],
) -> float:
    """
    Reference F1 where each action is treated independently and thresholded.
    """
    f1_helper = ProdMetricCompHelperOnline()

    for pred in tqdm(predictions, desc="F1 baseline: independent actions..."):
        key = pred.key()
        assert key in gt_per_pred, f"Missing GT for key {key}"
        gt = gt_per_pred[key]
        thr = _effective_threshold(pred)
        pred_bin = pred.probs >= thr

        f1_helper.update(
            pred=pred_bin,
            gt=gt,
            lab=pred.lab_name,
            action=pred.behavior.action,
        )

    return float(f1_helper.finalise())


def _compute_baseline_f1_argmax(
    predictions: Sequence["PredictedProbsForBehavior"],
    gt_per_pred: Dict[Tuple[int, int, int, str], np.ndarray],
) -> float:
    """
    F1 when at each (video_id, agent, target, frame) we pick argmax over actions.
    """
    # Group predictions by (video_id, agent, target)
    prediction_groups: Dict[Tuple[int, int, int], List[PredictedProbsForBehavior]] = (
        defaultdict(list)
    )
    for pred in predictions:
        prediction_groups[
            (pred.video_id, pred.behavior.agent, pred.behavior.target)
        ].append(pred)

    f1_helper = ProdMetricCompHelperOnline()

    for group in tqdm(
        prediction_groups.values(),
        desc="F1 baseline: argmax over actions per frame...",
    ):
        if not group:
            continue

        # Ensure deterministic ordering
        group.sort(key=lambda p: p.behavior.action)

        gt_list = []
        probs_list = []
        for pred in group:
            gt = gt_per_pred[pred.key()]
            gt_list.append(gt[:, None])
            probs_list.append(pred.probs[:, None])

        gt_mat = np.concatenate(gt_list, axis=1)  # (T, A)
        probs_mat = np.concatenate(probs_list, axis=1)  # (T, A)
        assert gt_mat.shape == probs_mat.shape

        # Argmax over actions for each frame
        winners = np.argmax(probs_mat, axis=1)  # (T,)
        pred_mat = np.zeros_like(probs_mat, dtype=bool)
        pred_mat[np.arange(len(winners)), winners] = True

        # Feed back into per-lab per-action stats
        for j, pred in enumerate(group):
            f1_helper.update(
                pred=pred_mat[:, j],
                gt=gt_mat[:, j],
                lab=pred.lab_name,
                action=pred.behavior.action,
            )

    return float(f1_helper.finalise())


def _compute_cv_baseline_f1_no_post(
    predictions: Sequence["PredictedProbsForBehavior"],
    gt_per_pred: Dict[Tuple[int, int, int, str], np.ndarray],
) -> Tuple[Dict[int, float], float]:
    """
    Per-fold F1 without postprocessing (only thresholding).
    fold is taken from pred.fold; folds with fold in {None, -1} are ignored.
    """
    per_fold_helpers: Dict[int, ProdMetricCompHelperOnline] = {}

    for pred in tqdm(predictions, desc="Per-fold baseline F1 (no postprocessing)..."):
        fold = getattr(pred, "fold", None)
        if fold is None or fold == -1:
            continue
        fold = int(fold)

        key = pred.key()
        assert key in gt_per_pred, f"Missing GT for key {key}"
        gt = gt_per_pred[key]
        thr = _effective_threshold(pred)
        pred_bin = pred.probs >= thr

        helper = per_fold_helpers.get(fold)
        if helper is None:
            helper = ProdMetricCompHelperOnline()
            per_fold_helpers[fold] = helper

        helper.update(
            pred=pred_bin,
            gt=gt,
            lab=pred.lab_name,
            action=pred.behavior.action,
        )

    per_fold_f1: Dict[int, float] = {
        fold: float(helper.finalise()) for fold, helper in per_fold_helpers.items()
    }

    avg_f1 = float(np.mean(list(per_fold_f1.values()))) if per_fold_f1 else 0.0
    return per_fold_f1, avg_f1


# ---------------------------------------------------------------------------
# Group construction for grid search
# ---------------------------------------------------------------------------


def _build_groups_for_grid_search(
    predictions: Sequence["PredictedProbsForBehavior"],
    gt_per_pred: Dict[Tuple[int, int, int, str], np.ndarray],
) -> Dict[GroupKey, List[_GroupRecord]]:
    """
    Group predictions by (lab, action, fold), skipping fold in {None, -1}.
    """
    groups: Dict[GroupKey, List[_GroupRecord]] = defaultdict(list)

    for pred in tqdm(
        predictions, desc="Assigning predictions to (lab, action, fold) groups..."
    ):
        fold = getattr(pred, "fold", None)
        if fold is None or fold == -1:
            continue

        key_gt = pred.key()
        assert key_gt in gt_per_pred, f"Missing GT for key {key_gt}"
        gt = gt_per_pred[key_gt]
        probs = np.asarray(pred.probs, dtype=np.float32)
        assert gt.shape == probs.shape, f"Shape mismatch for key {key_gt}"

        assert pred.fps is not None
        fps = float(pred.fps)  # per-pred FPS

        group_key: GroupKey = (pred.lab_name, pred.behavior.action, int(fold))
        groups[group_key].append(
            _GroupRecord(
                probs=probs,
                gt=gt.astype(np.bool_),
                threshold=_effective_threshold(pred),
                fps=fps,
            )
        )

    return groups


# ---------------------------------------------------------------------------
# Worker for multiprocessing
# ---------------------------------------------------------------------------


def _evaluate_group_worker(
    group_key: GroupKey,
    records: Sequence[_GroupRecord],
    params_list: Sequence[PostprocessingParams],
) -> List[Tuple[GroupKey, int, int, int, int]]:
    """
    Evaluate all PostprocessingParams in params_list for a single (lab, action, fold)
    group and return TP/FP/FN for each param.

    Returns a list of:
        (group_key, param_index, tp, fp, fn)
    """
    results: List[Tuple[GroupKey, int, int, int, int]] = []

    for param_idx, params in enumerate(params_list):
        tp = fp = fn = 0

        for rec in records:
            probs_post = params.apply(rec.probs, fps=rec.fps)
            pred_bin = (probs_post >= rec.threshold).astype(bool)
            gt = rec.gt

            tp += int(np.logical_and(pred_bin, gt).sum())
            fp += int(np.logical_and(pred_bin, ~gt).sum())
            fn += int(np.logical_and(~pred_bin, gt).sum())

        results.append((group_key, param_idx, tp, fp, fn))

    return results


# ---------------------------------------------------------------------------
# Leak-free fold evaluation
# ---------------------------------------------------------------------------


def _compute_leak_free_cv_f1(
    counts_by_param_lab_action_fold: Dict[
        Tuple[int, str, str, int], Dict[str, int]
    ],  # (param_idx, lab, action, fold) -> {'tp','fp','fn'}
    param_count: int,
) -> Tuple[Dict[int, float], float]:
    """
    For each fold:
      - choose best params for each (lab, action) using all other folds
      - evaluate on this fold with those params
      - compute F1 on that fold

    Returns:
        per_fold_f1, avg_f1
    """
    # Collect folds and (lab, action) pairs
    folds = sorted({fold for (_, _, _, fold) in counts_by_param_lab_action_fold.keys()})
    all_lab_actions = {
        (lab, action) for (_, lab, action, _) in counts_by_param_lab_action_fold.keys()
    }

    per_fold_f1: Dict[int, float] = {}

    for fold in tqdm(folds, desc="Leak-free fold evaluation..."):
        stats_fold: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(dict)

        for lab, action in all_lab_actions:
            best_param_idx: Optional[int] = None
            best_train_f1 = -1.0

            # choose best param using all folds except this one
            for p_idx in range(param_count):
                tp_train = fp_train = fn_train = 0

                for other_fold in folds:
                    if other_fold == fold:
                        continue
                    s = counts_by_param_lab_action_fold.get(
                        (p_idx, lab, action, other_fold)
                    )
                    if s is None:
                        continue
                    tp_train += s["tp"]
                    fp_train += s["fp"]
                    fn_train += s["fn"]

                denom_train = 2 * tp_train + fp_train + fn_train
                if denom_train == 0:
                    continue

                f1_train = 2.0 * tp_train / denom_train
                if f1_train > best_train_f1:
                    best_train_f1 = f1_train
                    best_param_idx = p_idx

            if best_param_idx is None:
                # no training signal for this (lab, action)
                continue

            # now apply chosen param to this fold
            s_test = counts_by_param_lab_action_fold.get(
                (best_param_idx, lab, action, fold)
            )
            if s_test is None:
                continue

            stats_fold[lab][action] = dict(s_test)

        f1_fold = _compute_f1_from_counts(stats_fold)
        per_fold_f1[fold] = f1_fold

    avg_f1 = float(np.mean(list(per_fold_f1.values()))) if per_fold_f1 else 0.0
    return per_fold_f1, avg_f1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def evaluate_postprocessing_and_save(
    predictions: Sequence[PredictedProbsForBehavior],
    param_grid: Sequence[PostprocessingParams],
    json_out_path: str,
    max_workers: int = 1,
) -> None:
    """
    Main function:

    - rebuild gt_per_pred (from meta / annotations)
    - compute reference F1:
        * independent actions (thresholded)
        * argmax per (video_id, agent, target, frame)
    - build groups by (lab, action, fold) where fold not in {None, -1}
    - run grid search over postprocessing params, computing
      (action, lab, fold, params) -> (tp, fp, fn, f1)
      optionally in parallel processes
    - report:
        * best F1 with a single params choice for entire dataset
        * best F1 if we choose params separately per (lab, action)
        * leak-free CV F1:
            - avg F1 over folds (no postprocessing)
            - avg F1 with per-(lab,action) params chosen on other folds only
    - save all detailed results to json_out_path

    Note: FPS is taken from each prediction individually via pred.fps.
    """
    if not param_grid:
        raise ValueError("param_grid must not be empty.")

    print("Step 1/6: Building ground truth...")
    gt_per_pred = _build_gt_per_pred()

    print("Step 2/6: Computing baseline F1 metrics...")
    baseline_f1_indep = _compute_baseline_f1_independent(predictions, gt_per_pred)
    baseline_f1_argmax = _compute_baseline_f1_argmax(predictions, gt_per_pred)

    print(f"Baseline F1 (independent actions): {baseline_f1_indep:.5f}")
    print(f"Baseline F1 (argmax over actions): {baseline_f1_argmax:.5f}")

    # CV baseline (no postprocessing)
    per_fold_f1_no_post, avg_f1_no_post = _compute_cv_baseline_f1_no_post(
        predictions, gt_per_pred
    )
    print("Per-fold F1 without postprocessing:")
    for fold, f1 in sorted(per_fold_f1_no_post.items()):
        print(f"  fold={fold}: F1={f1:.5f}")
    print(f"Average F1 over folds (no postprocessing): {avg_f1_no_post:.5f}")

    print("Step 3/6: Building groups for grid search...")
    groups = _build_groups_for_grid_search(predictions, gt_per_pred)
    print(f"  total (lab, action, fold) groups: {len(groups)}")

    if not groups:
        print("No groups with valid folds were found; nothing to tune.")
        # still save baselines
        payload = {
            "param_grid": [_params_to_dict(p) for p in param_grid],
            "baseline": {
                "f1_independent": baseline_f1_indep,
                "f1_argmax": baseline_f1_argmax,
                "per_fold_no_post": per_fold_f1_no_post,
                "avg_f1_no_post": avg_f1_no_post,
            },
            "grid_results": [],
            "best_global": None,
            "best_per_action_lab_global_f1": None,
            "leak_free": None,
        }
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("Done.")
        return

    print("Step 4/6: Running grid search over postprocessing params...")

    # counts_total[param_idx][lab][action] = {'tp','fp','fn'} across ALL folds
    counts_total: Dict[int, Dict[str, Dict[str, Dict[str, int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}))
    )
    # counts_by_param_lab_action_fold[(param_idx, lab, action, fold)]
    counts_by_param_lab_action_fold: Dict[Tuple[int, str, str, int], Dict[str, int]] = (
        defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    )

    params_list = list(param_grid)
    n_params = len(params_list)

    # we'll update this frequently in tqdm
    def _current_best_from_counts() -> Tuple[Optional[int], float]:
        best_idx: Optional[int] = None
        best_f1 = -1.0
        for p_idx, stats in counts_total.items():
            f1 = _compute_f1_from_counts(stats)
            if f1 > best_f1:
                best_f1 = f1
                best_idx = p_idx
        if best_idx is None:
            return None, 0.0
        return best_idx, best_f1

    # Grid search (possibly with multiprocessing)
    group_items = list(groups.items())

    if max_workers is not None and max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(_evaluate_group_worker, gk, recs, params_list)
                for gk, recs in group_items
            ]

            with tqdm(
                total=len(futures),
                desc=f"Grid search over {n_params} postprocessing configs (mp)",
            ) as pbar:
                for fut in as_completed(futures):
                    group_results = fut.result()
                    # integrate counts
                    for group_key, param_idx, tp, fp, fn in group_results:
                        lab, action, fold = group_key

                        s_total = counts_total[param_idx][lab][action]
                        s_total["tp"] += tp
                        s_total["fp"] += fp
                        s_total["fn"] += fn

                        s_group = counts_by_param_lab_action_fold[
                            (param_idx, lab, action, fold)
                        ]
                        s_group["tp"] += tp
                        s_group["fp"] += fp
                        s_group["fn"] += fn

                    # update progress + best so far
                    best_idx, best_f1 = _current_best_from_counts()
                    if best_idx is not None:
                        pbar.set_postfix(best_f1=f"{best_f1:.5f}", best_param=best_idx)
                    pbar.update(1)
    else:
        with tqdm(
            total=len(group_items),
            desc=f"Grid search over {n_params} postprocessing configs",
        ) as pbar:
            for group_key, recs in group_items:
                group_results = _evaluate_group_worker(group_key, recs, params_list)

                for group_key2, param_idx, tp, fp, fn in group_results:
                    lab, action, fold = group_key2

                    s_total = counts_total[param_idx][lab][action]
                    s_total["tp"] += tp
                    s_total["fp"] += fp
                    s_total["fn"] += fn

                    s_group = counts_by_param_lab_action_fold[
                        (param_idx, lab, action, fold)
                    ]
                    s_group["tp"] += tp
                    s_group["fp"] += fp
                    s_group["fn"] += fn

                best_idx, best_f1 = _current_best_from_counts()
                if best_idx is not None:
                    pbar.set_postfix(best_f1=f"{best_f1:.5f}", best_param=best_idx)
                pbar.update(1)

    # Final best single params for entire dataset
    best_global_param_idx, best_global_f1 = _current_best_from_counts()
    best_global_params = (
        params_list[best_global_param_idx]
        if best_global_param_idx is not None
        else None
    )
    print("Step 5/6: Aggregating grid-search results...")

    if best_global_params is not None:
        print(
            f"Best F1 with single global params: {best_global_f1:.5f} "
            f"(param index {best_global_param_idx}, params={_params_to_dict(best_global_params)})"
        )
    else:
        print("Could not determine best global params (no stats).")

    # --- best per (lab, action) F1 (using all folds) ---

    counts_per_lab_action_param: Dict[Tuple[str, str, int], Dict[str, int]] = (
        defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    )

    for (param_idx, lab, action, fold), s in counts_by_param_lab_action_fold.items():
        key = (lab, action, param_idx)
        stats = counts_per_lab_action_param[key]
        stats["tp"] += s["tp"]
        stats["fp"] += s["fp"]
        stats["fn"] += s["fn"]

    best_param_per_lab_action: Dict[Tuple[str, str], Tuple[int, float]] = {}
    for (lab, action, param_idx), s in counts_per_lab_action_param.items():
        denom = 2 * s["tp"] + s["fp"] + s["fn"]
        f1 = 0.0 if denom == 0 else 2.0 * s["tp"] / denom
        key = (lab, action)
        current = best_param_per_lab_action.get(key)
        if current is None or f1 > current[1]:
            best_param_per_lab_action[key] = (param_idx, f1)

    # Build stats dict as if we had chosen best params per (lab, action)
    stats_best_per_lab_action: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(dict)
    for (lab, action), (p_idx, _) in best_param_per_lab_action.items():
        stats = counts_per_lab_action_param[(lab, action, p_idx)]
        stats_best_per_lab_action[lab][action] = dict(stats)

    best_per_action_lab_global_f1 = _compute_f1_from_counts(stats_best_per_lab_action)
    print(
        f"Best F1 with separate params per (lab, action): "
        f"{best_per_action_lab_global_f1:.5f}"
    )

    # --- leak-free CV evaluation using fold structure ---

    print("Step 6/6: Computing leak-free cross-validated F1...")
    per_fold_f1_leak_free, avg_f1_leak_free = _compute_leak_free_cv_f1(
        counts_by_param_lab_action_fold, param_count=n_params
    )

    print("Per-fold leak-free F1 with tuned params:")
    for fold, f1 in sorted(per_fold_f1_leak_free.items()):
        print(f"  fold={fold}: F1={f1:.5f}")

    print(
        f"Average F1 over folds (no postprocessing): {avg_f1_no_post:.5f}\n"
        f"Average F1 over folds (leak-free tuned params): {avg_f1_leak_free:.5f}"
    )

    # -----------------------------------------------------------------------
    # Save everything to JSON
    # -----------------------------------------------------------------------
    print(f"Saving detailed results to {json_out_path!r}...")

    grid_results_json: List[Dict] = []
    for (param_idx, lab, action, fold), s in counts_by_param_lab_action_fold.items():
        denom = 2 * s["tp"] + s["fp"] + s["fn"]
        f1 = 0.0 if denom == 0 else 2.0 * s["tp"] / denom

        grid_results_json.append(
            {
                "lab": lab,
                "action": action,
                "fold": fold,
                "param_index": param_idx,
                "params": _params_to_dict(params_list[param_idx]),
                "tp": s["tp"],
                "fp": s["fp"],
                "fn": s["fn"],
                "f1": f1,
            }
        )

    payload = {
        "param_grid": [_params_to_dict(p) for p in params_list],
        "baseline": {
            "f1_independent": baseline_f1_indep,
            "f1_argmax": baseline_f1_argmax,
            "per_fold_no_post": per_fold_f1_no_post,
            "avg_f1_no_post": avg_f1_no_post,
        },
        "grid_results": grid_results_json,
        "best_global": {
            "param_index": best_global_param_idx,
            "params": (
                _params_to_dict(best_global_params)
                if best_global_params is not None
                else None
            ),
            "f1": best_global_f1,
        },
        "best_per_action_lab_global_f1": best_per_action_lab_global_f1,
        "best_param_per_lab_action": {
            f"{lab}::{action}": {
                "param_index": idx,
                "params": _params_to_dict(params_list[idx]),
                "f1": f1,
            }
            for (lab, action), (idx, f1) in best_param_per_lab_action.items()
        },
        "leak_free": {
            "per_fold": per_fold_f1_leak_free,
            "avg_f1_leak_free": avg_f1_leak_free,
        },
    }

    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Done.")
