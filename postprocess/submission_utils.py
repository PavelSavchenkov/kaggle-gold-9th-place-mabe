from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
from tqdm import tqdm

from common.config_utils import base_model_from_str, base_model_to_file
from common.constants import (
    ACTION_NAMES_IN_TEST,
    ACTIONS_PER_ENSEMBLE_APPROACH,
    LABS_IN_TEST_PER_ACTION,
)
from common.ensemble_building_primitives import EnsembleObjective
from common.helpers import copy_all_source_code_to
from common.metrics_common import DatasetScore
from common.parse_utils import BehaviorLabeled
from gbdt.configs import FeaturesConfig
from postprocess.ensemble_utils import (
    EnsembleApproach,
    build_ensembles_from_models_per_approach,
)
from postprocess.postprocess_utils import OOF_Metrics, OOF_Metrics_for_model


@lru_cache
def best_ensemble_approach_for_action(action: str) -> EnsembleApproach:
    if action == "tussle":
        return EnsembleApproach(
            max_models=3,
            objective=EnsembleObjective.nested_f1,
            num_weight_bins=10,
            pool_size=1,
        )
    for app_str, actions in ACTIONS_PER_ENSEMBLE_APPROACH.items():
        if action in actions:
            return base_model_from_str(EnsembleApproach, app_str)
    # return EnsembleApproach(max_models=3, objective=EnsembleObjective.nested_f1, num_weight_bins=10, pool_size=1)
    raise ValueError(f"Could not find best ensemble approach for action={action}")


class SubmissionModel(BaseModel):
    features_config: FeaturesConfig | None = None
    features_config_hash: str | None = None

    action: str = ""
    lab: str | None = ""
    name: str = ""
    steps: list[int] = Field(default_factory=list)
    folds: list[int] = Field(default_factory=list)
    coef: float = 1.0


class SubmissionThreshold(BaseModel):
    action: str = ""
    lab: str | None = ""
    threshold: float = -1.0
    threshold_per_fold: dict[int, float] = Field(default_factory=dict)

    def __lt__(self, other: SubmissionThreshold) -> bool:
        return (self.lab, self.action) < (other.lab, other.action)


class SubmissionSmoothing(BaseModel):
    lab: str
    action: str
    window: int


class Submission(BaseModel):
    dataset_score: DatasetScore = Field(default_factory=lambda: DatasetScore())
    models: list[SubmissionModel] = Field(default_factory=list)
    thresholds: list[SubmissionThreshold] = Field(default_factory=list)
    smoothing: list[SubmissionSmoothing] = Field(default_factory=list)

    @staticmethod
    def build_ensemble(
        models_dict: dict[tuple[str, str], list[OOF_Metrics_for_model]],
        verbose: bool = False,
    ) -> "Submission":
        missing_keys = set()
        for action in ACTION_NAMES_IN_TEST:
            for lab in LABS_IN_TEST_PER_ACTION[action]:
                key = (action, lab)
                if key not in models_dict:
                    missing_keys.add(key)
        assert not missing_keys, f"missing keys: {missing_keys}"

        all_oof_metrics = {}
        submission = Submission()

        for (action, lab), models in tqdm(
            models_dict.items(), "Model lists per (action, lab)"
        ):
            approach = best_ensemble_approach_for_action(action=action)
            best_single_model = models[0]
            if verbose:
                print(
                    f"Building ensemble: action={action}, lab={lab}, Approach: {approach.to_str()} ..."
                )

            e = build_ensembles_from_models_per_approach(
                models=models,
                lab=lab,
                e_approaches=[approach],
                verbose=verbose,
            )[approach]
            oof_metrics, model_idx, model_coefs = e
            if verbose:
                for f in OOF_Metrics.metric_fields:
                    old_value = getattr(best_single_model.metrics, f)
                    if isinstance(old_value, list):
                        old_value = np.mean(old_value)
                    new_value = getattr(oof_metrics, f)
                    if isinstance(new_value, list):
                        new_value = np.mean(new_value)
                    print(f"{f:15}: {old_value:.5f} --> {new_value:.5f}")
                print("---")

            submission_threshold = SubmissionThreshold(
                action=action,
                lab=lab,
                threshold=oof_metrics.oof_th,
                threshold_per_fold=oof_metrics.th_per_fold,
            )
            submission.thresholds.append(submission_threshold)
            all_oof_metrics[(action, lab)] = oof_metrics
            for i, coef in zip(model_idx, model_coefs):
                model = models[i]
                if len(model.folds) != 5:
                    print(
                        f"less than 5 folds: {action}, {lab}, {model.folds}, {model.name}"
                    )
                submission_model = SubmissionModel(
                    action=action,
                    lab=lab,
                    name=model.name,
                    steps=model.steps,
                    folds=model.folds,
                    coef=coef,
                )
                submission.models.append(submission_model)

        submission.dataset_score = DatasetScore.from_oof_metrics(
            all_metrics=all_oof_metrics
        )

        return submission

    @staticmethod
    def build_ensembles_per_approach(
        models_dict: dict[tuple[str, str], list[OOF_Metrics_for_model]],
        e_approaches: list[EnsembleApproach],
        verbose: bool = False,
    ) -> dict[EnsembleApproach, "Submission"]:
        # missing_keys = set()
        # for action in ACTION_NAMES_IN_TEST:
        #     for lab in LABS_IN_TEST_PER_ACTION[action]:
        #         key = (action, lab)
        #         if key not in models_dict:
        #             missing_keys.add(key)
        # assert not missing_keys, f"missing keys: {missing_keys}"

        all_oof_metrics = {}
        submissions = {}

        for (action, lab), models in tqdm(
            models_dict.items(), "Model lists per (action, lab)"
        ):
            best_single_model = models[0]
            if verbose:
                print(f"Building ensemble: action={action}, lab={lab} ...")

            es_per_approach = build_ensembles_from_models_per_approach(
                models=models,
                lab=lab,
                e_approaches=e_approaches,
                verbose=verbose,
            )
            if verbose:
                for app, e in es_per_approach.items():
                    oof_metrics, _, _ = e
                    print(f"approach: {app.to_str()}")
                    for f in OOF_Metrics.metric_fields:
                        old_value = getattr(best_single_model.metrics, f)
                        if isinstance(old_value, list):
                            old_value = np.mean(old_value)
                        new_value = getattr(oof_metrics, f)
                        if isinstance(new_value, list):
                            new_value = np.mean(new_value)
                        print(f"{f:15}: {old_value:.5f} --> {new_value:.5f}")
                    print("---")

            for app, e in es_per_approach.items():
                if app not in submissions:
                    submissions[app] = Submission()
                oof_metrics, model_idx, model_coefs = e
                submission_threshold = SubmissionThreshold(
                    action=action,
                    lab=lab,
                    threshold=oof_metrics.oof_th,
                    threshold_per_fold=oof_metrics.th_per_fold,
                )
                submissions[app].thresholds.append(submission_threshold)
                if app not in all_oof_metrics:
                    all_oof_metrics[app] = {}
                all_oof_metrics[app][(action, lab)] = oof_metrics
                for i, coef in zip(model_idx, model_coefs):
                    model = models[i]
                    # if len(model.folds) != 5:
                    #     print(
                    #         f"less than 5 folds: {action}, {lab}, {model.folds}, {model.name}"
                    #     )
                    submission_model = SubmissionModel(
                        action=action,
                        lab=lab,
                        name=model.name,
                        steps=model.steps,
                        folds=model.folds,
                        coef=coef,
                    )
                    submissions[app].models.append(submission_model)

        for app in e_approaches:
            submissions[app].dataset_score = DatasetScore.from_oof_metrics(
                all_oof_metrics[app]
            )

        return submissions

    @staticmethod
    def build_ensembles_per_approach_per_action(
        models_dict: dict[str, list[OOF_Metrics_for_model]],
        e_approaches: list[EnsembleApproach],
        verbose: bool = False,
    ) -> dict[EnsembleApproach, "Submission"]:
        # missing_keys = set()
        # for action in ACTION_NAMES_IN_TEST:
        #     for lab in LABS_IN_TEST_PER_ACTION[action]:
        #         key = (action, lab)
        #         if key not in models_dict:
        #             missing_keys.add(key)
        # assert not missing_keys, f"missing keys: {missing_keys}"

        all_oof_metrics = {}
        submissions = {}

        for action, models in tqdm(models_dict.items(), "Model lists per action"):
            best_single_model = models[0]
            if verbose:
                print(f"Building ensemble: action={action} ...")

            es_per_approach = build_ensembles_from_models_per_approach(
                models=models,
                lab=None,
                e_approaches=e_approaches,
                verbose=verbose,
            )
            if verbose:
                for app, e in es_per_approach.items():
                    oof_metrics, _, _ = e
                    print(f"approach: {app.to_str()}")
                    for f in OOF_Metrics.metric_fields:
                        old_value = getattr(best_single_model.metrics, f)
                        if isinstance(old_value, list):
                            old_value = np.mean(old_value)
                        new_value = getattr(oof_metrics, f)
                        if isinstance(new_value, list):
                            new_value = np.mean(new_value)
                        print(f"{f:15}: {old_value:.5f} --> {new_value:.5f}")
                    print("---")

            for app, e in es_per_approach.items():
                for lab in LABS_IN_TEST_PER_ACTION[action]:
                    if app not in submissions:
                        submissions[app] = Submission()
                    oof_metrics, model_idx, model_coefs = e
                    submission_threshold = SubmissionThreshold(
                        action=action, lab=lab, threshold=oof_metrics.oof_th
                    )
                    submissions[app].thresholds.append(submission_threshold)
                    if app not in all_oof_metrics:
                        all_oof_metrics[app] = {}
                    all_oof_metrics[app][(action, lab)] = oof_metrics
                    for i, coef in zip(model_idx, model_coefs):
                        model = models[i]
                        # if len(model.folds) != 5:
                        #     print(
                        #         f"less than 5 folds: {action}, {lab}, {model.folds}, {model.name}"
                        #     )
                        submission_model = SubmissionModel(
                            action=action,
                            lab=lab,
                            name=model.name,
                            steps=model.steps,
                            folds=model.folds,
                            coef=coef,
                        )
                        submissions[app].models.append(submission_model)

        for app in e_approaches:
            submissions[app].dataset_score = DatasetScore.from_oof_metrics(
                all_oof_metrics[app]
            )

        return submissions

    def get_threshold(self, lab: str, action: str) -> float:
        for th in self.thresholds:
            if th.lab == lab and th.action == action:
                return th.threshold
        for th in self.thresholds:
            if th.action == action and th.lab is None:
                return th.threshold
        raise ValueError(f"Could not find threshold for lab={lab}, action={action}")

    def get_threshold_for_fold(self, lab: str, action: str, fold: int) -> float:
        for th in self.thresholds:
            if th.lab == lab and th.action == action:
                return th.threshold_per_fold[fold]
        raise ValueError(f"Could not find threshold for lab={lab}, action={action}")

    def get_smoothing_window(self, lab: str, action: str) -> int | None:
        for sm in self.smoothing:
            if sm.lab == lab and sm.action == action:
                return sm.window
        return None

    def fill_from(self, lab: str, action: str, other: Submission):
        for model in other.models:
            if model.action == action and model.lab == lab:
                self.models.append(model.model_copy(deep=True))
        for th in other.thresholds:
            if th.action == action and th.lab == lab:
                self.thresholds.append(th.model_copy(deep=True))

    def write_dir(self, dst: Path | str):
        dst = Path(dst)
        dst.mkdir(exist_ok=True, parents=True)
        for model in tqdm(self.models, desc="Writing submission models"):
            copy_model_file_to_submission(dst, model.name)
        base_model_to_file(self, dst / "submission.json")

    def get_models_for(self, lab: str, action: str) -> list[SubmissionModel]:
        res = []
        for model in self.models:
            if model.lab == lab and model.action == action:
                res.append(model)
        if not res:
            for model in self.models:
                if model.action == action:
                    assert model.lab is None
                    res.append(model)
        assert res, f"No models was found for action={action}, lab={lab}"
        return res

    def sanity_check_labs_actions_present(self):
        was_model = set()
        was_th = set()
        for model in self.models:
            was_model.add((model.action, model.lab))
        for th in self.thresholds:
            was_th.add((th.action, th.lab))

        for action in ACTION_NAMES_IN_TEST:
            for lab in LABS_IN_TEST_PER_ACTION[action]:
                key = (action, lab)
                assert key in was_model, key
                assert key in was_th, key


def copy_model_file_to_submission(
    dst_dir: Path | str, name: str, delete_test: bool = True
):
    dst_dir = Path(dst_dir)
    dst_model_dir = dst_dir / name
    if dst_model_dir.exists():
        return
    shutil.copytree(Path("train_logs") / name, dst_model_dir)
    if delete_test:
        for dir in dst_model_dir.rglob("*"):
            if dir.is_dir() and dir.name == "test":
                shutil.rmtree(dir)


def make_submission(
    submission_name: str,
    models_dict: dict[tuple[str, str], OOF_Metrics_for_model],
    rewrite: bool = False,
):
    dataset_score = DatasetScore.from_oof_metrics_for_model(all_metrics=models_dict)  # type: ignore
    print(dataset_score.to_str(include_labs=True))

    dst = Path(submission_name)
    if dst.exists() and not rewrite:
        raise RuntimeError(f"{str(dst)} should not exist")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    copy_all_source_code_to(dst / "code")

    missing_keys = set()
    for action in ACTION_NAMES_IN_TEST:
        for lab in LABS_IN_TEST_PER_ACTION[action]:
            key = (action, lab)
            if key not in models_dict:
                missing_keys.add(key)
    assert not missing_keys, f"missing keys: {missing_keys}"

    all_actions = set()
    submission = Submission(dataset_score=dataset_score)
    for (action, lab), model in tqdm(models_dict.items()):
        submission_model = SubmissionModel(
            action=action,
            lab=lab,
            name=model.name,
            steps=model.steps,
            folds=model.folds,
        )
        submission_threshold = SubmissionThreshold(
            action=action, lab=lab, threshold=model.metrics.oof_th
        )
        if len(model.folds) != 5:
            print(f"less than 5 folds: {action}, {lab}, {model.folds}, {model.name}")
        submission.models.append(submission_model)
        submission.thresholds.append(submission_threshold)
        all_actions.add(action)

    print(all_actions.symmetric_difference(ACTION_NAMES_IN_TEST))
    submission.write_dir(dst)


def make_ensemble_submission(
    submission_name: str,
    models_dict: dict[tuple[str, str], list[OOF_Metrics_for_model]],
    rewrite: bool = False,
):
    dst = Path(submission_name)
    if dst.exists() and not rewrite:
        raise RuntimeError(f"{str(dst)} should not exist")
    if dst.exists():
        print(f"[FOLDER DELETE] {dst}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    # copy_all_source_code_to(dst / "code")

    # app = EnsembleApproach(
    #     max_models=3,
    #     objective=EnsembleObjective.nested_f1,
    #     num_weight_bins=10,
    #     pool_size=1,
    # )
    # submission = Submission.build_ensembles_per_approach(
    #     models_dict=models_dict,
    #     e_approaches=[app],
    # )[app]
    submission = Submission.build_ensemble(
        models_dict=models_dict,
        verbose=True,
    )
    print(submission.dataset_score.to_str())

    submission.write_dir(dst)


class Stats:
    total_features_time: float = 0.0
    total_features_rows: int = 0

    total_prediction_time: float = 0.0
    total_prediction_rows_x_trees: int = 0

    def submit_features(self, elapsed: float, rows: int):
        self.total_features_time += elapsed
        self.total_features_rows += rows

    def submit_predict(self, elapsed: float, rows_x_trees: int):
        self.total_prediction_time += elapsed
        self.total_prediction_rows_x_trees += rows_x_trees

    def print(self, final: bool = False):
        eps = 1e-10
        total_time = self.total_features_time + self.total_prediction_time
        feats_perc = self.total_features_time / (total_time + eps) * 100
        pred_perc = self.total_prediction_time / (total_time + eps) * 100
        print(f"------------ AGG STATS ------------")
        print(
            f"[AGG FEATURES ({feats_perc:4.1f}%)] per 1M rows            : {self.total_features_time * (1_000_000 / (self.total_features_rows + eps)):.3f} s"
        )
        print(
            f"[AGG PREDICT  ({pred_perc:4.1f}%)] per 1M rows x 300 trees: {self.total_prediction_time * (1_000_000 * 300) / (self.total_prediction_rows_x_trees + eps):.3f} s"
        )
        if final:
            print(f"[TOTAL FEATURES]: {timedelta(seconds=self.total_features_time)}")
            print(f"[TOTAL  PREDICT]: {timedelta(seconds=self.total_prediction_time)}")
