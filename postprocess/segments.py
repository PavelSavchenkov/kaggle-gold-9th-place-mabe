from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.signal import medfilt
from sklearn.metrics import f1_score
from tqdm import tqdm

from common.helpers import get_annotation_by_video_meta, get_train_meta
from common.parse_utils import BehaviorLabeled
from common.submission_common import PredictedProbsForBehavior
from eval_metrics import calc_best_f1_threshold
from postprocess.submission_utils import (
    Submission,
    SubmissionSmoothing,
    SubmissionThreshold,
)


def moving_median(x: np.ndarray, window: int) -> np.ndarray:
    return medfilt(x, kernel_size=window)


def analyze_segments(
    all_probs: list[PredictedProbsForBehavior], submission: Submission
):
    train_meta = get_train_meta()
    video_meta_by_video_id = {}
    annot_by_video_id = {}
    print(f"Reading train meta...")
    for video_meta in train_meta[train_meta.has_annotation].to_dict(orient="records"):
        video_id = int(video_meta["video_id"])
        video_meta_by_video_id[video_id] = video_meta
        annot = get_annotation_by_video_meta(video_meta)
        annot_by_video_id[video_id] = annot

    @dataclass
    class Item:
        probs: np.ndarray
        gt: np.ndarray
        beh: BehaviorLabeled
        threshold: float
        threshold_fold: float

    print(f"Building prob maps...")
    item_map = {}  # lab -> action -> fold -> list[probs]
    for probs in all_probs:
        beh = probs.behavior
        action = beh.action
        lab = video_meta_by_video_id[probs.video_id]["lab_id"]
        annot = annot_by_video_id[probs.video_id]
        fold = probs.fold
        assert fold != -1
        if lab not in item_map:
            item_map[lab] = {}
        if action not in item_map[lab]:
            item_map[lab][action] = defaultdict(list)

        cnt_frames = probs.probs.shape[0]
        gt = np.zeros((cnt_frames,), dtype=np.int8)
        for annot_row in annot[
            (annot.agent_id == beh.agent)
            & (annot.target_id == beh.target)
            & (annot.action == action)
        ].to_dict(orient="records"):
            l = annot_row["start_frame"]
            r = annot_row["stop_frame"]
            gt[l:r] = 1

        item_map[lab][action][fold].append(
            Item(
                probs=probs.probs,
                gt=gt,
                beh=beh,
                threshold=probs.threshold,
                threshold_fold=probs.threshold_fold,
            )
        )

    print(f"Calculating original f1...")
    for th_field in ["threshold", "threshold_fold"]:
        f1_per_valid = []
        for valid in tqdm(range(5)):
            f1_global = []
            for lab in sorted(item_map.keys()):
                f1_for_lab = []
                for action in sorted(item_map[lab].keys()):
                    pred_list = []
                    gt_list = []
                    if valid not in item_map[lab][action]:
                        # print(
                        #     f"[WARNING] action={action}, lab={lab} does not have fold={valid}"
                        # )
                        continue
                    for item in item_map[lab][action][valid]:
                        th = getattr(item, th_field)
                        assert th >= 0
                        probs = item.probs
                        pred_list.append(probs >= th)
                        gt_list.append(item.gt)
                    pred = np.concatenate(pred_list, axis=0)
                    gt = np.concatenate(gt_list, axis=0)
                    f1 = f1_score(y_true=gt, y_pred=pred)
                    f1_for_lab.append(f1)
                f1_for_lab = float(np.mean(f1_for_lab))
                f1_global.append(f1_for_lab)
            f1_global = float(np.mean(f1_global))
            f1_per_valid.append(f1_global)
        f1_avg = float(np.mean(f1_per_valid))
        print(f"{th_field}: f1={f1_avg:.5f}")
        print(f1_per_valid)
        print()

    # submission.thresholds.clear()
    # submission.smoothing.clear()
    # WINDOWS = [3, 5, 7, 9, 11]
    # print(
    #     f"Calculating f1 if refit threshold and windows on full inference data in oof fashion..."
    # )
    # f1_global = []
    # for lab in tqdm(list(sorted(item_map.keys()))):
    #     f1_for_lab = []
    #     for action in sorted(item_map[lab].keys()):
    #         best_f1 = -1.0
    #         best_window = None
    #         best_th = None
    #         for window in WINDOWS:
    #             f1_per_valid = []
    #             for valid in range(5):
    #                 probs_list = defaultdict(list)
    #                 gt_list = defaultdict(list)
    #                 for fold in range(5):
    #                     if fold not in item_map[lab][action]:
    #                         continue
    #                     for item in item_map[lab][action][fold]:
    #                         if fold == valid:
    #                             dst = "valid"
    #                         else:
    #                             dst = "train"
    #                         probs = moving_average(item.probs, window)
    #                         probs_list[dst].append(probs)
    #                         gt_list[dst].append(item.gt)
    #                 if not probs_list["valid"]:
    #                     continue
    #                 gt = {}
    #                 for dst in ["valid", "train"]:
    #                     gt[dst] = np.concatenate(gt_list[dst], axis=0)
    #                 probs = {}
    #                 for dst in ["valid", "train"]:
    #                     probs[dst] = np.concatenate(probs_list[dst], axis=0)
    #                 th = calc_best_f1_threshold(
    #                     y_true=gt["train"], y_pred=probs["train"]
    #                 )
    #                 f1 = f1_score(y_true=gt["valid"], y_pred=probs["valid"] >= th)
    #                 f1_per_valid.append(f1)
    #             f1 = float(np.mean(f1_per_valid))
    #             if f1 > best_f1:
    #                 best_f1 = f1
    #                 best_th = th
    #                 best_window = window
    #         assert best_th is not None
    #         assert best_window is not None
    #         submission.thresholds.append(
    #             SubmissionThreshold(lab=lab, action=action, threshold=best_th)
    #         )
    #         submission.smoothing.append(
    #             SubmissionSmoothing(lab=lab, action=action, window=best_window)
    #         )
    #         f1_for_lab.append(best_f1)
    #     f1_for_lab = float(np.mean(f1_for_lab))
    #     f1_global.append(f1_for_lab)
    # f1_global = float(np.mean(f1_global))
    # print(
    #     f"If select best window and then threshold for each (lab, action): f1={f1_global:.5f}"
    # )
    # print()

    print(
        f"Calculating f1 if refit threshold on full inference data on all folds at once..."
    )
    submission.thresholds.clear()
    thresholds = {}
    f1_per_valid = []
    for valid in tqdm(range(5)):
        f1_global = []
        for lab in sorted(item_map.keys()):
            f1_for_lab = []
            for action in sorted(item_map[lab].keys()):
                window = submission.get_smoothing_window(lab=lab, action=action)
                assert window is not None
                print(f"lab={lab:20}, action={action:15}: window={window}")
                train_probs = []
                train_gt = []
                valid_probs = []
                valid_gt = []
                for fold in range(5):
                    if fold not in item_map[lab][action]:
                        continue
                    for item in item_map[lab][action][fold]:
                        probs = item.probs
                        probs = moving_average(probs, window=window)
                        if fold == valid:
                            valid_probs.append(probs)
                            valid_gt.append(item.gt)
                        train_probs.append(probs)
                        train_gt.append(item.gt)
                if not valid_probs:
                    continue
                train_probs = np.concatenate(train_probs, axis=0)
                train_gt = np.concatenate(train_gt, axis=0)
                valid_probs = np.concatenate(valid_probs, axis=0)
                valid_gt = np.concatenate(valid_gt, axis=0)
                th = calc_best_f1_threshold(y_true=train_gt, y_pred=train_probs)
                thresholds[(action, lab)] = SubmissionThreshold(
                    action=action, lab=lab, threshold=th
                )
                f1 = f1_score(y_true=valid_gt, y_pred=valid_probs >= th)
                f1_for_lab.append(f1)
            f1_for_lab = float(np.mean(f1_for_lab))
            f1_global.append(f1_for_lab)
        f1_global = float(np.mean(f1_global))
        f1_per_valid.append(f1_global)
    f1_avg = float(np.mean(f1_per_valid))
    print(f"f1={f1_avg:.5f}")
    print(f1_per_valid)
    print()

    submission.thresholds = list(sorted(thresholds.values()))
