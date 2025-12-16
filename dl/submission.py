from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
from tqdm import tqdm

from common.config_utils import base_model_from_file, base_model_to_file
from common.ensemble_building_primitives import build_ensembles_beam_search
from common.metrics_common import DatasetScore, OOF_Metrics
from common.paths import MODELS_ROOT
from dl.configs import DL_TrainConfig
from dl.ensembling_dl import EnsembleApproachDL
from dl.oof_utils import OOF_Full, get_oof_data, get_oof_data_from_folds_ratios
from dl.postprocess import get_train_config_dl, step_to_ratio


class SubmissionThreshold(BaseModel):
    action: str = ""
    lab: str | None = ""
    th: float = -1


class SubmissionModel(BaseModel):
    action: str = ""
    lab: str | None = ""
    coef: float = 1.0
    oof_full: OOF_Full = Field(default_factory=OOF_Full)

    def name(self) -> str:
        return self.oof_full.name

    def threshold(self) -> float:
        return self.oof_full.metrics.oof_th

    def folds(self) -> list[int]:
        return self.oof_full.metrics.get_folds()

    def get_step(self, fold: int) -> int:
        return self.oof_full.step_per_fold[fold]

    def get_cv_dir(self, models_dir: Path | str, fold: int) -> Path:
        models_dir = Path(models_dir)
        return models_dir / self.name() / f"cv{fold}"

    def get_config(self, models_dir: Path | str, fold: int) -> DL_TrainConfig:
        models_dir = Path(models_dir)
        return base_model_from_file(
            DL_TrainConfig,
            self.get_cv_dir(models_dir=models_dir, fold=fold) / "train_config.json",
        )

    def get_ckpt_dir(self, models_dir: Path | str, fold: int) -> Path:
        step = self.get_step(fold)
        return self.get_cv_dir(models_dir=models_dir, fold=fold) / f"checkpoint-{step}"


class Submission(BaseModel):
    models: list[SubmissionModel] = Field(default_factory=list)
    dataset_score: DatasetScore = Field(default_factory=DatasetScore)
    thresholds: list[SubmissionThreshold] = Field(default_factory=list)

    def get_model_for(self, action: str, lab: str) -> SubmissionModel:
        res = None
        for model in self.models:
            if model.action == action and model.lab == lab:
                assert res is None
                res = model
        if res is None:
            for model in self.models:
                if model.action == action:
                    assert model.lab is None
                    assert res is None
                    res = model
        if res is None:
            raise ValueError(f"Could not find the model for action={action}, lab={lab}")
        return res

    def get_all_models_for(self, action: str, lab: str) -> list[SubmissionModel]:
        res = []
        for model in self.models:
            if model.action == action and model.lab == lab:
                res.append(model)
        if not res:
            for model in self.models:
                if model.action == action and model.lab is None:
                    res.append(model)
        if not res:
            raise ValueError(
                f"Could not find any models for action={action}, lab={lab}"
            )
        return res

    def get_threshold_for(self, action: str, lab: str) -> float:
        th = None
        for threshold in self.thresholds:
            if threshold.action == action and threshold.lab == lab:
                assert th is None
                th = threshold.th
        if th is None:
            for threshold in self.thresholds:
                if threshold.action == action and threshold.lab is None:
                    assert th is None
                    th = threshold.th
        assert th is not None, f"No threshold for {action}, {lab}"
        assert th >= 0.0, f"th={th}"
        return th

    @staticmethod
    def from_oof_map(oof_map: dict[tuple[str, str], OOF_Full]) -> Submission:
        submission = Submission()
        oof_metrics_map = {}
        for action, lab in sorted(oof_map.keys()):
            oof_full = oof_map[(action, lab)]
            submission.models.append(
                SubmissionModel(action=action, lab=lab, coef=1.0, oof_full=oof_full)
            )
            oof_metrics_map[(action, lab)] = oof_full.metrics
        submission.dataset_score = DatasetScore.from_oof_metrics(oof_metrics_map)
        return submission

    @staticmethod
    def from_oof_map_to_ensemble(
        oof_map: dict[tuple[str, str], list[OOF_Full]], approach: EnsembleApproachDL
    ) -> Submission:
        submission = Submission()
        oof_metrics_map = {}
        for action, lab in sorted(oof_map.keys()):
            oof_list = oof_map[(action, lab)]
            y_pred_list = []
            y_true = None
            fold_id = None
            names_list = []
            for oof in tqdm(
                oof_list, desc=f"Building oof data for {(action, lab)} ..."
            ):
                name = oof.name
                names_list.append(name)
                folds = oof.metrics.get_folds()
                ratios = []
                for f in folds:
                    config = get_train_config_dl(name=oof.name, cv=f)
                    assert config is not None
                    step = oof.step_per_fold[f]
                    ratio = step_to_ratio(config=config, step=step)
                    ratios.append(ratio)
                oof_data = get_oof_data_from_folds_ratios(
                    name=oof.name,
                    action=action,
                    lab=lab,
                    folds=folds,
                    ratios=ratios,
                    verbose=False,
                )
                for f, step in oof_data.step_per_fold.items():
                    assert (
                        oof.step_per_fold.get(f) == step
                    ), f"name: {name}, initial step_per_fold: {oof.step_per_fold}, reproduced step_per_fold: {oof_data.step_per_fold}, action: {action}, lab: {lab}"
                if fold_id is None:
                    fold_id = oof_data.fold_id
                else:
                    assert (fold_id == oof_data.fold_id).all()
                if y_true is None:
                    y_true = oof_data.y_true
                else:
                    assert (y_true == oof_data.y_true).all()
                y_pred = oof_data.y_pred
                y_pred_list.append(y_pred)

            assert y_true is not None
            assert y_pred_list
            assert fold_id is not None

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
            submission.thresholds.append(
                SubmissionThreshold(action=action, lab=lab, th=th)
            )
            for idx, w in zip(e.model_idx, e.weights):
                submission.models.append(
                    SubmissionModel(
                        action=action, lab=lab, coef=w, oof_full=oof_list[idx]
                    )
                )
            print()
        submission.dataset_score = DatasetScore.from_oof_metrics(oof_metrics_map)
        return submission

    def write_dir(self, dst: Path | str, overwrite: bool = False):
        dst = Path(dst)
        if dst.exists():
            if not overwrite:
                raise RuntimeError(f"{dst} should not exist")
            shutil.rmtree(dst)
        dst.mkdir(exist_ok=True, parents=True)
        for model in tqdm(self.models, desc=f"Writing models to {dst}"):
            src_model = MODELS_ROOT / model.name()
            dst_model = dst / model.name()
            dst_model.mkdir(exist_ok=True)
            for fold in model.folds():
                src_dir = src_model / f"cv{fold}"
                dst_dir = dst_model / f"cv{fold}"
                step = model.get_step(fold)
                rel_paths = [
                    "train_config.json",
                    "stats.npz",
                    "info/split.json",
                    f"checkpoint-{step}/model.safetensors",
                ]
                for rel_path in rel_paths:
                    rel_path = Path(rel_path)
                    src_file = src_dir / rel_path
                    dst_file = dst_dir / rel_path
                    dst_file.parent.mkdir(exist_ok=True, parents=True)
                    shutil.copy2(src_file, dst_file)

        base_model_to_file(self, dst / "submission.json")
