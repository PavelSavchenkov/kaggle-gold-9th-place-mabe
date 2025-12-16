from collections import defaultdict
from ctypes import c_void_p
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from common.config_utils import base_model_from_file, base_model_to_file
from common.constants import ACTION_NAMES_IN_TEST, LABS_IN_TEST_PER_ACTION
from common.metrics_common import OOF_Metrics
from common.paths import MODELS_ROOT
from dl.configs import DL_TrainConfig
from dl.postprocess import (
    Ckpt,
    full_training_steps_from_config_dl,
    get_best_pr_auc_ckpt_for_each_action,
    get_best_pr_auc_ckpt_for_each_action_lab,
    get_train_config_dl,
    iter_all_checkpoints_for_model_cv,
    iter_all_configs_for_model_name_dl,
    iter_all_dl_model_names,
    ratio_to_step,
    step_to_ratio,
)


@dataclass
class OOF_Data:
    y_true: np.ndarray
    y_pred: np.ndarray
    fold_id: np.ndarray
    step_per_fold: dict[int, int]

    def print_info(self):
        def np_info(name: str) -> str:
            arr = getattr(self, name)
            return f"{name}: {arr.shape}, {arr.dtype}"

        info = "OOF_Data | "
        for name in ["y_true", "y_pred", "fold_id"]:
            info += np_info(name)
            info += " | "
        info += ", step_per_fold: { "
        for fold, step in sorted(self.step_per_fold.items()):
            info += f"{fold}: {step}, "
        info += "}"
        print(info)


class OOF_Full(BaseModel):
    metrics: OOF_Metrics = Field(default_factory=OOF_Metrics)
    step_per_fold: dict[int, int] = Field(default_factory=dict)
    name: str = ""


class OOF_All_Actions_Labs(BaseModel):
    oof_map: dict[str, dict[str, OOF_Full]] = Field(default_factory=dict)


class OOF_All_Actions(BaseModel):
    oof_map: dict[str, OOF_Full] = Field(default_factory=dict)


def oof_save_path(name: str, median: bool = False, per_action: bool = False) -> Path:
    fname = "oof" if not median else "oof_median"
    if per_action:
        fname += "_per_action"
    fname += ".json"
    return MODELS_ROOT / name / fname


def lazy_calc_oof_all_actions(name: str, median: bool = False):
    dst = oof_save_path(name, median=median, per_action=True)
    if dst.exists():
        return

    oof_map = {}
    for action in ACTION_NAMES_IN_TEST:
        oof_data = get_oof_data(name=name, action=action, lab=None, median_ckpt=median)
        # oof_data.print_info()
        oof_metrics = OOF_Metrics.build(
            y_pred=oof_data.y_pred, y_true=oof_data.y_true, fold_id=oof_data.fold_id
        )
        oof_full = OOF_Full(
            metrics=oof_metrics, step_per_fold=oof_data.step_per_fold, name=name
        )
        oof_map[action] = oof_full

    oof_all = OOF_All_Actions(oof_map=oof_map)
    # print(oof_all.model_dump(mode="json"))
    base_model_to_file(oof_all, dst)


def lazy_calc_oof_all_actions_labs(name: str, median: bool = False):
    dst = oof_save_path(name, median=median)
    if dst.exists():
        return

    oof_map = {}
    for action in ACTION_NAMES_IN_TEST:
        oof_map[action] = {}
        for lab in LABS_IN_TEST_PER_ACTION[action]:
            oof_data = get_oof_data(
                name=name, action=action, lab=lab, median_ckpt=median
            )
            # oof_data.print_info()
            oof_metrics = OOF_Metrics.build(
                y_pred=oof_data.y_pred, y_true=oof_data.y_true, fold_id=oof_data.fold_id
            )
            oof_full = OOF_Full(
                metrics=oof_metrics, step_per_fold=oof_data.step_per_fold, name=name
            )
            oof_map[action][lab] = oof_full

    oof_all = OOF_All_Actions_Labs(oof_map=oof_map)
    # print(oof_all.model_dump(mode="json"))
    base_model_to_file(oof_all, dst)


def get_oof_data_from_folds_ratios(
    name: str,
    action: str,
    lab: str | None,
    folds: list[int],
    ratios: list[float],
    verbose: bool = True,
    float_dtype: type[np.floating[Any]] = np.float32,
) -> OOF_Data:
    assert len(folds) == len(ratios)
    assert list(sorted(folds)) == list(sorted(set(folds)))
    assert folds

    configs_list = []
    ckpt_list = []
    for fold, ratio in zip(folds, ratios):
        config = get_train_config_dl(name=name, cv=fold)
        assert config is not None
        step = ratio_to_step(config=config, ratio=ratio)
        ckpt = Ckpt.from_config_and_step(config=config, step=step)
        ckpt_list.append(ckpt)
        configs_list.append(config)

    y_true_list = []
    y_pred_list = []
    fold_id_list = []
    step_per_fold = {}

    for ckpt, config in zip(ckpt_list, configs_list):
        step_per_fold[config.cv()] = ckpt.step
        if lab is not None:
            gt_path = ckpt.gt_npy_path(action=action, lab=lab)
            preds_path = ckpt.preds_npy_path(action=action, lab=lab)

            y_true = np.load(gt_path).astype(np.int8)
            y_pred = np.load(preds_path).astype(float_dtype)
        else:
            y_true_per_lab = []
            y_pred_per_lab = []
            for lab_id in LABS_IN_TEST_PER_ACTION[action]:
                gt_path = ckpt.gt_npy_path(action=action, lab=lab_id)
                preds_path = ckpt.preds_npy_path(action=action, lab=lab_id)
                # assert gt_path.exists() == preds_path.exists()
                if not gt_path.exists() or not preds_path.exists():
                    continue
                cur_y_true = np.load(gt_path).astype(np.int8)
                cur_y_pred = np.load(preds_path).astype(float_dtype)
                y_true_per_lab.append(cur_y_true)
                y_pred_per_lab.append(cur_y_pred)
            y_true = np.concatenate(y_true_per_lab, axis=0)
            y_pred = np.concatenate(y_pred_per_lab, axis=0)

        fold_id = np.ones(y_true.shape, dtype=np.int8) * config.cv()

        assert y_true.shape == y_pred.shape

        if (y_true == 1).all() or (y_true == 0).all():
            if verbose:
                print(
                    f"[WARNING] Degenerate y_true in {config.name}/cv{config.cv()}, action: {action}, lab: {lab}"
                )
            continue

        y_true_list.append(y_true)
        y_pred_list.append(y_pred)
        fold_id_list.append(fold_id)

    assert (
        y_true_list
    ), f"name={name}, action={action}, lab={lab}, folds={folds}, ratios={ratios}"
    assert y_pred_list

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    fold_id = np.concatenate(fold_id_list, axis=0)

    return OOF_Data(
        y_true=y_true, y_pred=y_pred, fold_id=fold_id, step_per_fold=step_per_fold
    )


def get_median_ratio_by_pr_auc(
    name: str, action: str, lab: str | None, folds: list[int] | None = None
) -> float:
    if folds is None:
        folds = list(range(5))
    assert folds
    ratios = []
    for f in folds:
        config = get_train_config_dl(name=name, cv=f)
        assert config is not None
        if lab is not None:
            best_map = get_best_pr_auc_ckpt_for_each_action_lab(config.save_dir())
            key = (action, lab)
        else:
            best_map = get_best_pr_auc_ckpt_for_each_action(config.save_dir())
            key = action
        if key not in best_map:
            continue
        _, ckpt = best_map[key]  # type: ignore
        ratio = step_to_ratio(config=config, step=ckpt.step)
        ratios.append(ratio)
    assert ratios
    return float(np.median(ratios))


def get_oof_data(
    name: str,
    action: str,
    lab: str | None,
    median_ckpt: bool = False,
    folds: list[int] | None = None,
    verbose: bool = True,
    float_dtype: type[np.floating[Any]] = np.float32,
) -> OOF_Data:
    if folds is None:
        folds = list(range(5))

    configs_list: list[DL_TrainConfig] = [get_train_config_dl(name=name, cv=f) for f in folds]  # type: ignore
    configs_list.sort(key=lambda cfg: cfg.cv())

    ratios = []
    configs_filtered = []
    for config in configs_list:
        if lab is not None:
            map_best = get_best_pr_auc_ckpt_for_each_action_lab(
                save_dir=config.save_dir()
            )
            key = (action, lab)
        else:
            map_best = get_best_pr_auc_ckpt_for_each_action(save_dir=config.save_dir())
            key = action
        if key not in map_best:
            if verbose:
                print(f"[WARNING] No {key} in {config.name}/cv{config.cv()}")
            continue
        _, ckpt = map_best[key]  # type: ignore

        ratios.append(step_to_ratio(config=config, step=ckpt.step))
        configs_filtered.append(config)
    assert len(ratios) > 1

    del configs_list
    configs_list = configs_filtered

    assert len(configs_list) == len(ratios)

    if median_ckpt:
        median_ratio = float(np.median(ratios))
        ratios = [median_ratio] * len(configs_list)

    folds = [config.cv() for config in configs_list]
    return get_oof_data_from_folds_ratios(
        name=name,
        action=action,
        lab=lab,
        folds=folds,
        ratios=ratios,
        verbose=verbose,
        float_dtype=float_dtype,
    )


def read_oof_map(
    name: str, median: bool = False, per_action: bool = False
) -> dict | None:
    oof_path = oof_save_path(name=name, median=median, per_action=per_action)
    if not oof_path.exists():
        return None
    oof_cls = OOF_All_Actions if per_action else OOF_All_Actions_Labs
    return base_model_from_file(oof_cls, oof_path).oof_map


def get_best_oof_per_action_lab(return_list: bool = False, median: bool = False):
    best_per = defaultdict(list)
    for name in iter_all_dl_model_names():
        oof_map = read_oof_map(name=name, median=median)
        if oof_map is None:
            continue
        for action in oof_map.keys():
            for lab in oof_map[action].keys():
                oof_full = oof_map[action][lab]
                best_per[(action, lab)].append(oof_full)

    single_best_per = {}
    for k, oof_list in best_per.items():
        oof_list.sort(
            key=lambda oof_full: np.mean(oof_full.metrics.nested_f1), reverse=True
        )
        single_best_per[k] = oof_list[0]
    return best_per if return_list else single_best_per
