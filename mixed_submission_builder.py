import json
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from common.config_utils import base_model_from_dict, base_model_from_file
from common.constants import (
    ACTION_NAMES_IN_TEST,
    LABS_IN_TEST_PER_ACTION,
    VIDEO_CATEGORICAL_FEATURES,
)
from common.ensemble_building_primitives import (
    EnsembleObjective,
    build_ensembles_beam_search,
)
from common.metrics_common import DatasetScore, OOF_Metrics
from common.paths import MODELS_ROOT
from dl.ensembling_dl import EnsembleApproachDL, build_ensemble_map_per_approach
from dl.gbdt_dl_stype import GBDT_DL_Style
from dl.metrics import ProdMetricCompHelper
from dl.oof_utils import (
    OOF_Data,
    OOF_Full,
    get_median_ratio_by_pr_auc,
    get_oof_data,
    read_oof_map,
)
from dl.postprocess import (
    Ckpt,
    get_action_lab_set_from_config_dl,
    get_actions_set_from_config_dl,
    get_best_pr_auc_ckpt_for_each_action,
    get_best_pr_auc_ckpt_for_each_action_lab,
    get_train_config_dl,
    iter_all_dl_model_names,
    ratio_to_step,
)
from dl.submission import Submission as SubmissionDL
from dl.submission import SubmissionModel as SubmissionModelDL
from dl.submission import SubmissionThreshold as SubmissionThresholdDL
from postprocess.submission_utils import Submission as SubmissionGBDT
from postprocess.submission_utils import SubmissionModel as SubmissionModelGBDT
from postprocess.submission_utils import SubmissionThreshold as SubmissionThresholdGBDT


def is_name_gbdt(name: str) -> bool:
    return name.startswith("gbdt-")


def build_mixed_ensemble_submissions(
    dl_submission_path: Path | str,
    gbdt_submission_path: Path | str,
    approach: EnsembleApproachDL,
    take_filter_dl: Callable | None = None,
    overwrite: bool = False,
    ref_gbdt_submission_path: Path | str = "submissions/e-nested-f1-all",
):
    ref_gbdt_submission_path = Path(ref_gbdt_submission_path)
    assert ref_gbdt_submission_path.exists()
    ref_gbdt_submission = base_model_from_file(
        SubmissionGBDT,
        ref_gbdt_submission_path / "submission.json",
    )

    def get_all_folds_of_gbdt_model(name: str) -> list[int]:
        folds = set()
        for model in ref_gbdt_submission.models:
            if model.name == name:
                folds.update(set(model.folds))
        return list(sorted(folds))

    dl_submission_path = Path(dl_submission_path)
    gbdt_submission_path = Path(gbdt_submission_path)
    if not overwrite and (dl_submission_path.exists() or gbdt_submission_path.exists()):
        raise ValueError(
            f"Both {dl_submission_path} and {gbdt_submission_path} should not exist."
        )

    submission_dl = SubmissionDL()
    submission_gbdt = SubmissionGBDT()
    oof_metrics_map = {}

    dl_names = list(iter_all_dl_model_names(take_filter=take_filter_dl))
    dl_names.sort()

    gbdt_names = [p.name for p in MODELS_ROOT.rglob("gbdt-*")]
    gbdt_names.sort()

    gbdt_obj_by_name: dict[str, GBDT_DL_Style] = {}
    for name in gbdt_names:
        gbdt_obj_by_name[name] = GBDT_DL_Style(model_path=MODELS_ROOT / name)

    names = dl_names + gbdt_names

    for action in ACTION_NAMES_IN_TEST:
        for lab in LABS_IN_TEST_PER_ACTION[action]:
            y_pred_list = []
            y_true = None
            fold_id = None
            folds = []
            relevant_names = []
            step_of_gbdt_model = {}
            for name in tqdm(names, desc=f"Iterate dl models for {(action, lab)} ..."):
                if not is_name_gbdt(name):
                    relevant_names.append(name)
                    oof_data = get_oof_data(
                        name=name,
                        action=action,
                        lab=lab,
                        median_ckpt=True,
                        verbose=False,
                    )
                    y_pred_list.append(oof_data.y_pred)
                    if fold_id is None:
                        fold_id = oof_data.fold_id
                        folds = list(sorted(set(np.unique(fold_id.astype(int)))))
                    else:
                        assert (fold_id == oof_data.fold_id).all()
                    if y_true is None:
                        y_true = oof_data.y_true
                    else:
                        assert (y_true == oof_data.y_true).all()
                else:
                    if (action, lab) not in gbdt_obj_by_name[name].get_action_lab_set():
                        continue
                    assert folds
                    assert fold_id is not None
                    assert y_true is not None
                    median_step = gbdt_obj_by_name[name].get_median_step(
                        action=action, lab=lab, folds=folds
                    )
                    assert median_step in gbdt_obj_by_name[name].steps
                    steps = [median_step] * len(folds)
                    oof_data = gbdt_obj_by_name[name].get_oof_data(
                        action=action, lab=lab, folds=folds, steps=steps
                    )
                    if fold_id.shape != oof_data.fold_id.shape:
                        continue
                    step_of_gbdt_model[name] = median_step
                    relevant_names.append(name)
                    assert (fold_id == oof_data.fold_id).all()
                    y_pred_list.append(oof_data.y_pred)

            assert y_true is not None
            assert fold_id is not None
            assert y_pred_list
            assert relevant_names
            assert len(y_pred_list) == len(relevant_names)

            print(
                f"Building ensemble for {(action, lab)}, approach: {approach.to_str()} ..."
            )
            es = build_ensembles_beam_search(
                y_preds=y_pred_list,
                y_true=y_true,
                num_bins=approach.num_bins,
                pool_size=approach.pool_size,
                max_models=approach.max_models,
                objective=approach.objective,
                fold_id=fold_id,
                allow_neg_weights=approach.allow_neg,
                verbose=False,
            )
            e = es[approach.max_models]
            pred_e = e.infer(preds_list=y_pred_list)
            oof_metrics = OOF_Metrics.build(
                y_pred=pred_e, y_true=y_true, fold_id=fold_id, folds=folds
            )
            oof_metrics_map[(action, lab)] = oof_metrics
            th = oof_metrics.oof_th
            # th_per_fold = oof_metrics.th_per_fold

            submission_dl.thresholds.append(
                SubmissionThresholdDL(action=action, lab=lab, th=th)
            )
            submission_gbdt.thresholds.append(
                SubmissionThresholdGBDT(action=action, lab=lab, threshold=th)
            )
            for idx, w in zip(e.model_idx, e.weights):
                name = relevant_names[idx]
                print(f"{w:.4f} * {name}")
                if not is_name_gbdt(name):
                    oof_map = read_oof_map(name=name, median=True)
                    assert oof_map is not None
                    oof_full = oof_map[action][lab]
                    submission_dl.models.append(
                        SubmissionModelDL(
                            action=action, lab=lab, coef=w, oof_full=oof_full
                        )
                    )
                else:
                    median_step = step_of_gbdt_model[name]
                    name = name[len("gbdt-") :]
                    folds = get_all_folds_of_gbdt_model(name=name)
                    steps = [median_step] * len(folds)
                    assert (MODELS_ROOT / name).exists()
                    submission_gbdt.models.append(
                        SubmissionModelGBDT(
                            action=action,
                            lab=lab,
                            name=name,
                            coef=w,
                            folds=folds,
                            steps=steps,
                        )
                    )
            print()

    submission_dl.dataset_score = DatasetScore.from_oof_metrics(oof_metrics_map)
    submission_gbdt.dataset_score = DatasetScore.from_oof_metrics(oof_metrics_map)
    print(submission_dl.dataset_score.to_str())
    submission_dl.write_dir(dst=dl_submission_path, overwrite=overwrite)
    if overwrite and gbdt_submission_path.exists():
        shutil.rmtree(gbdt_submission_path)
    submission_gbdt.write_dir(dst=gbdt_submission_path)


def build_mixed_ensemble_submissions_per_action(
    dl_submission_path: Path | str,
    gbdt_submission_path: Path | str,
    approach: EnsembleApproachDL,
    take_filter_dl: Callable | None = None,
    overwrite: bool = False,
    ref_gbdt_submission_path: Path | str = "submissions/e-nested-f1-all",
):
    ref_gbdt_submission_path = Path(ref_gbdt_submission_path)
    assert ref_gbdt_submission_path.exists()
    ref_gbdt_submission = base_model_from_file(
        SubmissionGBDT,
        ref_gbdt_submission_path / "submission.json",
    )

    def get_all_folds_of_gbdt_model(name: str) -> list[int]:
        folds = set()
        for model in ref_gbdt_submission.models:
            if model.name == name:
                folds.update(set(model.folds))
        return list(sorted(folds))

    dl_submission_path = Path(dl_submission_path)
    gbdt_submission_path = Path(gbdt_submission_path)
    if not overwrite and (dl_submission_path.exists() or gbdt_submission_path.exists()):
        raise ValueError(
            f"Both {dl_submission_path} and {gbdt_submission_path} should not exist."
        )

    submission_dl = SubmissionDL()
    submission_gbdt = SubmissionGBDT()
    oof_metrics_map = {}

    dl_names = list(iter_all_dl_model_names(take_filter=take_filter_dl))
    dl_names.sort()

    gbdt_names = [p.name for p in MODELS_ROOT.rglob("gbdt-*")]
    gbdt_names.sort()

    gbdt_obj_by_name: dict[str, GBDT_DL_Style] = {}
    for name in gbdt_names:
        gbdt_obj_by_name[name] = GBDT_DL_Style(model_path=MODELS_ROOT / name)

    names = dl_names + gbdt_names

    for action in ACTION_NAMES_IN_TEST:
        y_pred_list = []
        y_true = None
        fold_id = None
        lab_mask = None
        folds = []
        relevant_names = []
        step_of_gbdt_model = {}
        for name in tqdm(names, desc=f"Iterate dl models for action={action} ..."):
            if not is_name_gbdt(name):
                relevant_names.append(name)
                oof_data = get_oof_data(
                    name=name,
                    action=action,
                    lab=None,
                    median_ckpt=True,
                    verbose=False,
                )
                y_pred_list.append(oof_data.y_pred)
                if fold_id is None:
                    fold_id = oof_data.fold_id
                    folds = list(sorted(set(np.unique(fold_id.astype(int)))))
                else:
                    assert (fold_id == oof_data.fold_id).all()
                if y_true is None:
                    y_true = oof_data.y_true
                else:
                    assert (y_true == oof_data.y_true).all()
            else:
                if gbdt_obj_by_name[name].action != action:
                    continue
                assert folds
                assert fold_id is not None
                assert y_true is not None
                median_step = gbdt_obj_by_name[name].get_median_step(
                    action=action, lab=None, folds=folds
                )
                assert median_step in gbdt_obj_by_name[name].steps
                steps = [median_step] * len(folds)
                oof_data, cur_lab_mask = gbdt_obj_by_name[
                    name
                ].get_oof_data_with_lab_mask(
                    action=action, lab=None, folds=folds, steps=steps
                )
                if fold_id.shape != oof_data.fold_id.shape:
                    continue
                step_of_gbdt_model[name] = median_step
                relevant_names.append(name)
                assert (fold_id == oof_data.fold_id).all()
                y_pred_list.append(oof_data.y_pred)
                if lab_mask is None:
                    lab_mask = cur_lab_mask
                else:
                    assert (lab_mask == cur_lab_mask).all()

        assert y_true is not None
        assert fold_id is not None
        assert lab_mask is not None
        assert y_pred_list
        assert relevant_names
        assert len(y_pred_list) == len(relevant_names)

        print(
            f"Building ensemble for action={action}, approach: {approach.to_str()} ..."
        )
        es = build_ensembles_beam_search(
            y_preds=y_pred_list,
            y_true=y_true,
            num_bins=approach.num_bins,
            pool_size=approach.pool_size,
            max_models=approach.max_models,
            objective=approach.objective,
            fold_id=fold_id,
            allow_neg_weights=approach.allow_neg,
            verbose=False,
        )
        e = es[approach.max_models]
        pred_e = e.infer(preds_list=y_pred_list)
        oof_metrics = OOF_Metrics.build(
            y_pred=pred_e, y_true=y_true, fold_id=fold_id, folds=folds
        )

        lab_idx_all = list(sorted(set(np.unique(lab_mask))))
        for lab_idx in lab_idx_all:
            cur_lab_mask = lab_mask == lab_idx
            cur_pred_e = pred_e[cur_lab_mask]
            cur_y_true = y_true[cur_lab_mask]
            cur_fold_id = fold_id[cur_lab_mask]
            lab = VIDEO_CATEGORICAL_FEATURES["lab_id"][lab_idx]
            oof_metrics_map[(action, lab)] = OOF_Metrics.build(
                y_pred=cur_pred_e, y_true=cur_y_true, fold_id=cur_fold_id
            )
        th = oof_metrics.oof_th

        submission_dl.thresholds.append(
            SubmissionThresholdDL(action=action, lab=None, th=th)
        )
        submission_gbdt.thresholds.append(
            SubmissionThresholdGBDT(action=action, lab=None, threshold=th)
        )
        for idx, w in zip(e.model_idx, e.weights):
            name = relevant_names[idx]
            print(f"{w:.4f} * {name}")
            if not is_name_gbdt(name):
                oof_map = read_oof_map(name=name, median=True, per_action=True)
                assert oof_map is not None
                oof_full = oof_map[action]
                submission_dl.models.append(
                    SubmissionModelDL(
                        action=action, lab=None, coef=w, oof_full=oof_full
                    )
                )
            else:
                median_step = step_of_gbdt_model[name]
                name = name[len("gbdt-") :]
                folds = get_all_folds_of_gbdt_model(name=name)
                steps = [median_step] * len(folds)
                assert (MODELS_ROOT / name).exists()
                assert folds
                assert len(folds) == len(steps)
                submission_gbdt.models.append(
                    SubmissionModelGBDT(
                        action=action,
                        lab=None,
                        name=name,
                        coef=w,
                        folds=folds,
                        steps=steps,
                    )
                )
        print()

    submission_dl.dataset_score = DatasetScore.from_oof_metrics(oof_metrics_map)
    submission_gbdt.dataset_score = DatasetScore.from_oof_metrics(oof_metrics_map)
    print(submission_dl.dataset_score.to_str())
    submission_dl.write_dir(dst=dl_submission_path, overwrite=overwrite)
    if overwrite and gbdt_submission_path.exists():
        shutil.rmtree(gbdt_submission_path)
    submission_gbdt.write_dir(dst=gbdt_submission_path)
