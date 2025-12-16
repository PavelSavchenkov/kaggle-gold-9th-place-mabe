import array
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from common.constants import LABS_IN_TEST_PER_ACTION, VIDEO_CATEGORICAL_FEATURES
from dl.oof_utils import OOF_Data


class GBDT_DL_Style:
    root: Path
    folds: list[int]
    best_step_per_lab_cv: dict[tuple[str, int], int]
    best_step_per_cv: dict[int, int]
    action: str
    steps: list[int]  # same for all folds

    def __init__(self, model_path: Path | str):
        self.root = Path(model_path)
        self.folds = []
        self.best_step_per_lab_cv = {}
        self.best_step_per_cv = {}
        self.all_steps_per_lab_cv = {}
        self.steps = []
        for cv_dir in self.root.rglob("cv*"):
            cv = int(cv_dir.name[2:])
            self.folds.append(cv)
            best_step_per_lab = json.load(open(cv_dir / "best_step_per_lab.json"))
            self.action = best_step_per_lab["action"]
            del best_step_per_lab["action"]
            steps = best_step_per_lab["steps"]
            del best_step_per_lab["steps"]
            if not self.steps:
                self.steps = steps
            else:
                assert (
                    self.steps == steps
                ), f"self.steps: {self.steps}, steps from cv={cv}: {steps}"
            self.best_step_per_cv[cv] = best_step_per_lab["best_step"]
            del best_step_per_lab["best_step"]
            for lab, (step, _) in best_step_per_lab.items():
                assert step > 0
                self.best_step_per_lab_cv[(lab, cv)] = step
        assert hasattr(self, "action")

    def get_action_lab_set(self) -> set[tuple[int, int]]:
        action_lab_set = set()
        for lab, _ in self.best_step_per_lab_cv.keys():
            action_lab_set.add((self.action, lab))
        return action_lab_set

    def best_step_per_action_lab_cv(self, action: str, lab: str, cv: int) -> int:
        assert self.action == action
        return self.best_step_per_lab_cv[(lab, cv)]

    def get_median_step(
        self, action: str, lab: str | None, folds: list[int], verbose: bool = False
    ) -> int:
        assert self.action == action
        steps = []
        for f in folds:
            assert (
                f in self.folds
            ), f"root: {self.root}, self.folds: {self.folds}, f: {f}"
            if lab is not None:
                key = (lab, f)
                if key not in self.best_step_per_lab_cv:
                    if verbose:
                        print(f"[WARNING] No {key} in {self.root} | cv{f}")
                    continue
                step = self.best_step_per_lab_cv[key]
            else:
                key = f
                if key not in self.best_step_per_cv:
                    if verbose:
                        print(f"[WARNING] No key={key} in {self.root} | cv{f}")
                    continue
                step = self.best_step_per_cv[key]
            steps.append(step)
        assert steps, f"root: {self.root}, lab: {lab}, folds: {folds}"
        return statistics.median_low(steps)

    def get_oof_data_with_lab_mask(
        self,
        action: str,
        lab: str | None,
        folds: list[int],
        steps: list[int],
        verbose: bool = False,
        float_dtype: type[np.floating[Any]] = np.float32,
    ) -> tuple[OOF_Data, np.ndarray]:
        y_pred_list = []
        fold_id_list = []
        lab_mask_list = []
        step_per_fold = {}
        for fold, step in zip(folds, steps):
            step_per_fold[fold] = step
            pred, lab_mask = self.get_pred_or_none_with_lab_mask(
                fold=fold, step=step, action=action, lab=lab, float_dtype=float_dtype
            )
            if pred is None:
                if verbose:
                    print(
                        f"[WARNING] No lab={lab} prediction in {self.root} | cv{fold} | step {step}"
                    )
                continue
            fold_id = np.ones(pred.shape, dtype=np.int8) * fold
            y_pred_list.append(pred)
            fold_id_list.append(fold_id)
            lab_mask_list.append(lab_mask)
        y_pred = np.concatenate(y_pred_list, axis=0)
        fold_id = np.concatenate(fold_id_list, axis=0)
        lab_mask = np.concatenate(lab_mask_list, axis=0)
        return (
            OOF_Data(
                y_true=np.array(()),
                y_pred=y_pred,
                fold_id=fold_id,
                step_per_fold=step_per_fold,
            ),
            lab_mask,
        )

    def get_oof_data(
        self,
        action: str,
        lab: str | None,
        folds: list[int],
        steps: list[int],
        verbose: bool = False,
        float_dtype: type[np.floating[Any]] = np.float32,
    ) -> OOF_Data:
        return self.get_oof_data_with_lab_mask(
            action=action,
            lab=lab,
            folds=folds,
            steps=steps,
            verbose=verbose,
            float_dtype=float_dtype,
        )[0]

    def get_pred_path(self, fold: int, step: int, action: str, lab: str) -> Path:
        assert self.action == action
        return (
            self.root
            / f"cv{fold}"
            / f"checkpoint-{step}"
            / "preds"
            / f"{action}-{lab}.npy"
        )

    def get_pred(
        self,
        fold: int,
        step: int,
        action: str,
        lab: str,
        float_dtype: type[np.floating[Any]] = np.float32,
    ) -> np.ndarray:
        pred = self.get_pred_or_none(
            fold=fold, step=step, action=action, lab=lab, float_dtype=float_dtype
        )
        assert pred is not None
        return pred

    def get_pred_or_none(
        self,
        fold: int,
        step: int,
        action: str,
        lab: str | None,
        float_dtype: type[np.floating[Any]] = np.float32,
    ) -> np.ndarray | None:
        return self.get_pred_or_none_with_lab_mask(
            fold=fold, step=step, action=action, lab=lab, float_dtype=float_dtype
        )[0]

    def get_pred_or_none_with_lab_mask(
        self,
        fold: int,
        step: int,
        action: str,
        lab: str | None,
        float_dtype: type[np.floating[Any]] = np.float32,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        if lab is not None:
            pred_npy = self.get_pred_path(fold=fold, step=step, action=action, lab=lab)
            if not pred_npy.exists():
                return (None, np.array(()))
            lab_idx = VIDEO_CATEGORICAL_FEATURES["lab_id"].index(lab)
            pred = np.load(pred_npy).astype(float_dtype)
            lab_mask = np.ones(pred.shape, dtype=np.int8) * lab_idx
            return (pred, lab_mask)
        else:
            pred_list = []
            lab_mask_list = []
            for cur_lab in LABS_IN_TEST_PER_ACTION[self.action]:
                pred = self.get_pred_or_none(
                    fold=fold,
                    step=step,
                    action=action,
                    lab=cur_lab,
                    float_dtype=float_dtype,
                )
                if pred is None:
                    continue
                pred_list.append(pred)
                lab_idx = VIDEO_CATEGORICAL_FEATURES["lab_id"].index(cur_lab)
                lab_mask = np.ones(pred.shape, dtype=np.int8) * lab_idx
                lab_mask_list.append(lab_mask)
            if not pred_list:
                return (None, np.array(()))
            pred = np.concatenate(pred_list, axis=0)
            lab_mask = np.concatenate(lab_mask_list, axis=0)
            return (pred.astype(dtype=float_dtype, copy=False), lab_mask)
