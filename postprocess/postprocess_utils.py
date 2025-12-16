import json
from collections import defaultdict, namedtuple
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import average_precision_score, f1_score
from tqdm import tqdm

from common.config_utils import base_model_from_file, base_model_to_file
from common.constants import (
    ACTION_NAMES_IN_TEST,
    ALL_ACTION_NAMES,
    DEFAULT_DURATION_CAP,
    LAB_NAMES_IN_TEST,
    LABS_IN_TEST_PER_ACTION,
)
from common.folds_split_utils import train_test_split
from common.helpers import get_model_cv_path, get_train_meta
from common.metrics_common import (
    OOF_Metrics,
    calc_best_f1,
    calc_nested_f1_with_th_per_fold,
    calc_pr_auc,
)
from gbdt.callbacks import CustomEvalPeriodicLogic
from gbdt.configs import GBDT_TrainConfig
from gbdt.helpers import (
    get_any_train_config,
    get_best_f1_value,
    get_best_logloss_step,
    get_best_pr_auc_step,
    get_best_pr_auc_value,
    get_feats_test_for_model,
    get_oof_metrics_path,
    get_pred_test_np,
    get_pred_test_path,
    get_test_ground_truth_np,
    get_test_ground_truth_path,
    get_test_index_df,
    get_test_index_path,
    get_train_config,
    is_fully_trained,
    read_metrics,
)
from gbdt.model import GBDT_Model
from gbdt.rebalance_utils import DurationStats, get_labs_in_test_with_action
from gbdt.train import (
    fill_test_folder,
    try_fill_all_test_folders_for_model,
    try_fill_test_fold_for_model_cv,
)

CheckpointInfo = namedtuple("CheckpointInfo", ["pr_auc", "f1", "name", "step", "fold"])
ModelPostStats = namedtuple(
    "ModelPostStats",
    ["name", "action", "configs", "metrics", "folds", "labs_in_test_per_fold"],
)


class OOF_Metrics_for_model(BaseModel):
    name: str = ""
    folds: list[int] = Field(default_factory=list)
    steps: list[int] = Field(default_factory=list)
    metrics: OOF_Metrics = Field(default_factory=lambda: OOF_Metrics())

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.folds), tuple(self.steps), self.metrics))

    @staticmethod
    def build(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        fold_id: np.ndarray,
        folds: list[int],
    ):
        obj = OOF_Metrics_for_model()

        obj.metrics = OOF_Metrics.build(
            y_pred=y_pred, y_true=y_true, fold_id=fold_id, folds=folds
        )

        obj.folds = list(folds)

        return obj

    @lru_cache
    def oof_data(self, lab: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_pred_list = []
        y_true_list = []
        fold_id_list = []
        for f, step in zip(self.folds, self.steps):
            y_pred = get_pred_test_np(self.name, f, step, dtype=np.float32)
            y_true = get_test_ground_truth_np(self.name, f, dtype=np.int8)
            assert (
                y_pred.shape == y_true.shape
            ), f"y_pred.shape={y_pred.shape}, y_true.shape={y_true.shape}, model={self.name}, cv={f}"
            fold_id = np.ones(y_pred.shape, np.int8) * f
            if lab is not None:
                index = get_test_index_df(self.name, f, usecols=("lab_id",))
                lab_mask = (index["lab_id"] == lab).to_numpy()
                y_pred = y_pred[lab_mask]
                y_true = y_true[lab_mask]
                fold_id = fold_id[lab_mask]
            y_pred_list.append(y_pred)
            y_true_list.append(y_true)
            fold_id_list.append(fold_id)
        y_pred = np.concatenate(y_pred_list, axis=0)
        y_true = np.concatenate(y_true_list, axis=0)
        fold_id = np.concatenate(fold_id_list, axis=0)
        return y_pred, y_true, fold_id


class ALL_OOF_Metrics_for_model(BaseModel):
    name: str = Field(default_factory=lambda: "")
    action: str = Field(default_factory=lambda: "")
    total: OOF_Metrics_for_model | None = Field(default=None)
    per_lab: dict[str, OOF_Metrics_for_model] = Field(default_factory=dict)

    @staticmethod
    def build(stats: ModelPostStats):
        assert stats.folds == list(sorted(set(stats.folds)))

        obj = ALL_OOF_Metrics_for_model()

        obj.name = str(stats.name)
        obj.action = str(stats.action)

        y_true_list = []
        fold_id_list = []
        lab_id_list = []
        for cv in stats.folds:
            y_true = get_test_ground_truth_np(stats.name, cv)
            y_true_list.append(y_true)
            fold_id = np.full(shape=y_true.shape, fill_value=cv, dtype=np.int32)
            fold_id_list.append(fold_id)
            lab_id_series = get_test_index_df(stats.name, cv, usecols=("lab_id",))[
                "lab_id"
            ]
            lab_id_list.append(lab_id_series)
        y_true = np.concatenate(y_true_list, axis=0)
        fold_id = np.concatenate(fold_id_list, axis=0)
        lab_id_series = pd.concat(lab_id_list, ignore_index=True)

        lab_ids = list(sorted(lab_id_series.unique()))  # type: ignore
        lab_ids.append(None)

        for lab in lab_ids:
            folds = []
            y_pred_list = []
            steps = []
            for f, metrics in zip(stats.folds, stats.metrics):
                if lab is not None and lab not in stats.labs_in_test_per_fold[f]:
                    continue
                folds.append(f)
                step = get_best_pr_auc_step(metrics=metrics, lab=lab)
                steps.append(step)
                y_pred = get_pred_test_np(stats.name, f, step, np.float32)
                y_pred_list.append(y_pred)
            assert y_pred_list, f"name={stats.name}, lab={lab}"

            y_pred = np.concatenate(y_pred_list, axis=0)
            mask_folds = np.isin(fold_id, folds)
            y_true_cur = y_true[mask_folds]
            fold_id_cur = fold_id[mask_folds]
            lab_id_series_cur = lab_id_series[mask_folds]
            assert y_pred.shape == y_true_cur.shape
            if lab is not None:
                mask = (lab_id_series_cur == lab).to_numpy()
                y_pred = y_pred[mask]
                y_true_cur = y_true_cur[mask]
                fold_id_cur = fold_id_cur[mask]
            oof = OOF_Metrics_for_model.build(
                y_pred=y_pred, y_true=y_true_cur, fold_id=fold_id_cur, folds=folds
            )
            oof.steps = list(steps)
            oof.name = str(obj.name)
            if lab is not None:
                obj.per_lab[lab] = oof
            else:
                obj.total = oof

        return obj


def get_model_post_stats(
    name: str, folds_to_consider: list[int]
) -> ModelPostStats | None:
    configs = []
    metrics = []
    folds = []
    for f in folds_to_consider:
        if not is_fully_trained(name, f):
            continue
        metrics.append(read_metrics(name, f))
        configs.append(get_train_config(name, f))
        folds.append(f)

    if not configs:
        return None

    action = configs[0].action
    # do not skip only if test was really absent
    if len(configs) != len(folds_to_consider):
        if action != "tussle":  # only tussle does not have all folds
            return None
        # print(f"Checking if should skip for name = {name}")
        for f in set(folds_to_consider).difference(set(folds)):
            config = configs[0].model_copy(deep=True)
            config.data_split_config.test_fold = f
            _, test_meta = train_test_split_from_train_config(config)
            if not test_meta.empty:
                return None

    labs_in_test_per_fold = {}
    for idx, f in enumerate(folds):
        _, test_meta = train_test_split_from_train_config(configs[idx])
        labs_in_test_per_fold[f] = set(test_meta.lab_id.unique())

    return ModelPostStats(
        name=name,
        action=action,
        configs=configs,
        metrics=metrics,
        folds=folds,
        labs_in_test_per_fold=labs_in_test_per_fold,
    )


def train_test_split_from_train_config(
    config: GBDT_TrainConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        get_train_meta(), config=config.data_split_config, force_recalc=False
    )


def top_models_by_oof_per_action_per_lab(
    take_filter=None,
    return_list: bool = False,
    folds: list[int] | None = None,
    max_cnt: int | None = None,
):
    train_logs = Path("train_logs")
    res_dict = defaultdict(list)
    for model_path in tqdm(list(train_logs.iterdir())):
        name = model_path.name
        if take_filter is not None and not take_filter(name):
            continue
        try_calc_oof_metrics(name, rewrite=False, verbose=True)
        oof_path = get_oof_metrics_path(name, folds=folds)
        if not oof_path.exists():
            continue
        oof = base_model_from_file(ALL_OOF_Metrics_for_model, oof_path)
        for lab in sorted(oof.per_lab.keys()):
            key = (oof.action, lab)
            res_dict[key].append(oof.per_lab[lab])

    res_single = {}
    for key in sorted(res_dict.keys()):
        res_dict[key].sort(key=lambda oof: np.mean(oof.metrics.nested_f1), reverse=True)
        # res_dict[key].sort(key=lambda oof: oof.metrics.pr_auc_per_fold[1], reverse=True)
        if max_cnt is not None:
            res_dict[key] = res_dict[key][:max_cnt]
        res_single[key] = res_dict[key][0]
    return res_dict if return_list else res_single


def top_models_by_oof_per_action(
    take_filter=None, return_list: bool = False, folds: list[int] | None = None
):
    train_logs = Path("train_logs")
    res_dict = defaultdict(list)
    for model_path in tqdm(list(train_logs.iterdir())):
        name = model_path.name
        if take_filter is not None and not take_filter(name):
            continue
        try_calc_oof_metrics(name, rewrite=False, verbose=True)
        oof_path = get_oof_metrics_path(name, folds=folds)
        if not oof_path.exists():
            continue
        oof = base_model_from_file(ALL_OOF_Metrics_for_model, oof_path)

        # # !!!!!!!!!!!!!!!!!!
        # # print(f"DIRTY HACK TO REPLACE PR-AUC STEPS WITH LOGLOSS")
        # new_steps = []
        # for f in oof.total.folds: # type: ignore
        #     metrics = read_metrics(name, f)
        #     step = get_best_logloss_step(metrics)
        #     new_steps.append(step)
        # oof.total.steps = new_steps
        # # !!!!!!!!!!!!!!!!!!
        res_dict[oof.action].append(oof.total)

    res_single = {}
    for key in sorted(res_dict.keys()):
        res_dict[key].sort(key=lambda oof: np.mean(oof.metrics.nested_f1), reverse=True)
        res_single[key] = res_dict[key][0]
    return res_dict if return_list else res_single


def top_models_per_action(
    fold: int,
    return_list: bool = False,
    take_filter=None,
):
    if take_filter is None:
        take_filter = lambda _: True

    train_logs = Path("train_logs")

    res_dict = defaultdict(list)

    for model_path in tqdm(list(train_logs.iterdir())):
        name = model_path.name
        if take_filter is not None and not take_filter(name):
            continue
        stats = get_model_post_stats(name, folds_to_consider=[fold])
        if stats is None:
            continue
        metrics = stats.metrics[0]
        pr_auc = get_best_pr_auc_value(metrics)
        f1 = get_best_f1_value(metrics)
        step = get_best_pr_auc_step(metrics)
        info = CheckpointInfo(name=name, fold=[fold], step=step, pr_auc=pr_auc, f1=f1)
        res_dict[stats.action].append(info)

    for k in res_dict:
        res_dict[k].sort(key=lambda info: info.pr_auc, reverse=True)
    res_dict_one = {}
    for k in res_dict:
        res_dict_one[k] = res_dict[k][0]
    return res_dict if return_list else res_dict_one


def try_calc_oof_metrics(name: str, rewrite: bool = False, verbose: bool = False):
    base_config = get_any_train_config(name)
    if base_config is None:
        return

    all_folds = list(range(base_config.data_split_config.num_folds))
    folds_for_oof = []
    for valid in all_folds:
        folds = [fold for fold in all_folds if fold != valid]
        path = get_oof_metrics_path(name, folds=folds)
        if not path.exists() or rewrite:
            folds_for_oof.append(folds)
    path = get_oof_metrics_path(name, folds=None)
    if not path.exists() or rewrite:
        folds_for_oof.append(list(all_folds))

    if not folds_for_oof:
        return

    for folds in folds_for_oof:
        stats = get_model_post_stats(name, folds_to_consider=folds)
        if stats is None:
            return

        if verbose:
            print(f"Calculating OOF for {name}, folds: {folds}")

        oof = ALL_OOF_Metrics_for_model.build(stats)
        if len(folds) == len(all_folds):
            assert folds == all_folds
            folds = None
        base_model_to_file(oof, get_oof_metrics_path(name, folds=folds))


def cut_lightgbm_iterations_to(name: str, n_estimators: int) -> str | None:
    base_config = get_any_train_config(name)
    if base_config is None:
        return None
    if base_config.lightgbm_config.n_estimators <= n_estimators:
        return None
    dst_name = f"{name}-{n_estimators}-auto"

    dst_model_path = Path("train_logs") / dst_name
    dst_model_path.mkdir(exist_ok=True, parents=True)

    src_path = Path("train_logs") / name
    for cv_dir in src_path.iterdir():
        cv = cv_dir.name
        if "cv" not in cv:
            continue
        if not is_fully_trained(name, cv):
            continue
        if is_fully_trained(dst_name, cv):
            continue
        model = GBDT_Model.from_path(get_model_cv_path(name, cv))
        config = get_train_config(name, cv)

        dst_ckpt_path = get_model_cv_path(dst_name, cv)
        dst_ckpt_path.mkdir(exist_ok=True, parents=True)
        base_model_to_file(config, dst_ckpt_path / "train_config.json")

        feats = get_feats_test_for_model(name, cv)
        eval_logic = CustomEvalPeriodicLogic(
            model=model,
            eval_feats=feats,
            log_config=config.logging_config,
            use_wandb=False,
        )
        for epoch in range(n_estimators):
            eval_logic.after_iteration(epoch, model.get_booster(), preds=None)
        model.custom_eval_logic = eval_logic

        save_dir = dst_ckpt_path
        (save_dir / "final_model").mkdir(exist_ok=True, parents=True)

        metrics = model.get_after_train_metrics()
        json.dump(metrics, (save_dir / "final_model" / "metrics.json").open("w"))

        model.save(save_dir / "final_model")
        model.print_post_train_stats()

        fill_test_folder(name=dst_name, cv=cv, feats=feats, verbose=True)

    return dst_name


def is_full_fix_model_for_submission_needed(name: str, cv: int) -> bool:
    if is_fully_trained(name, cv):
        config = get_train_config(name, cv)
        if config.test_downsample_params.duration_cap != DEFAULT_DURATION_CAP:
            return True
    return False


def full_fix_model_for_submission(name: str, cv: int):
    if not is_full_fix_model_for_submission_needed(name, cv):
        return
    print(f"FIXING: {name}/cv{cv}")
    try_fill_test_fold_for_model_cv(
        name=name,
        cv=cv,
        rewrite=True,
        recalc_features=False,
        verbose=True,
        duration_cap=DEFAULT_DURATION_CAP,
    )
