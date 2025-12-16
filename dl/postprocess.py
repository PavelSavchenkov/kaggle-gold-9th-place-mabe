from __future__ import annotations

import json
import math
import pickle
import shutil
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from atomicwrites import atomic_write
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader  # type: ignore
from tqdm import tqdm
from transformers import TrainerState  # type: ignore

from common.config_utils import (
    base_model_from_file,
    base_model_to_file,
    base_model_to_hash,
)
from common.constants import (
    ACTION_NAMES_IN_TEST,
    LABS_IN_TEST_PER_ACTION,
    VIDEO_CATEGORICAL_FEATURES,
)
from common.folds_split_utils import train_test_split
from common.helpers import get_model_cv_path, get_train_meta, str_uint32_hash
from common.metrics_common import OOF_Metrics, calc_pr_auc
from common.paths import MODELS_ROOT
from dl.configs import AugmentationsConfig, BalanceConfig, DL_TrainConfig
from dl.data_balancer import DataBalancerForLabeledTest, DataBalancerForLabeledTrain
from dl.dataset import MiceDataset
from dl.features_lookup_gpu import FeaturesLookupGPU
from dl.metrics import MetricsComputer
from dl.model_helpers import InferenceModel


@dataclass
class Ckpt:
    path: Path
    step: int

    @staticmethod
    def from_config_and_step(config: DL_TrainConfig, step: int) -> Ckpt:
        path = config.save_dir() / f"checkpoint-{step}"
        assert path.exists()
        return Ckpt(path=path, step=step)

    def get_trainer_state(self) -> TrainerState | None:
        state_path = self.path / "trainer_state.json"
        if not state_path.exists():
            return None
        return TrainerState.load_from_json(state_path)

    def has_optimizer(self) -> bool:
        return (self.path / "optimizer.pt").exists()

    def get_train_config(self) -> DL_TrainConfig:
        return base_model_from_file(
            DL_TrainConfig, self.path.parent / "train_config.json"
        )

    def metrics_path(self):
        return self.path / "metrics.json"

    def read_metrics(self):
        return json.load(self.metrics_path().open())

    def all_action_lab_from_metrics(self) -> list[tuple[str, str]]:
        res = []
        m = self.read_metrics()
        for action in ACTION_NAMES_IN_TEST:
            for lab in LABS_IN_TEST_PER_ACTION[action]:
                key = f"test-each-pr-auc/{action}+{lab}"
                if key in m:
                    res.append((action, lab))
        return res

    def all_actions_from_metrics(self) -> list[str]:
        res = []
        m = self.read_metrics()
        for action in ACTION_NAMES_IN_TEST:
            key = f"test-per-action-pr-auc/{action}"
            if key in m:
                res.append(action)
        return res

    def preds_dir(self, suff: str = ""):
        return self.path / f"preds{suff}"

    def gt_dir(self, suff: str = ""):
        return self.path.parent / f"gt{suff}"

    def npy_fname_to_save(self, action, lab):
        return f"{action}-{lab}.npy"

    def preds_npy_path(self, action, lab, suff: str = ""):
        return self.preds_dir(suff=suff) / self.npy_fname_to_save(
            action=action, lab=lab
        )

    def get_pred_for_action(self, action: str, suff: str = "") -> np.ndarray:
        action_lab_set = set(self.all_action_lab_from_metrics())
        pred_list = []
        for lab in LABS_IN_TEST_PER_ACTION[action]:
            if (action, lab) not in action_lab_set:
                continue
            p = self.preds_npy_path(action=action, lab=lab, suff=suff)
            pred = np.load(p).astype(np.float32)
            pred_list.append(pred)
        return np.concatenate(pred_list, axis=0)

    def gt_npy_path(self, action, lab, suff: str = ""):
        return self.gt_dir(suff=suff) / self.npy_fname_to_save(action=action, lab=lab)

    def get_gt_for_action(self, action: str, suff: str = "") -> np.ndarray:
        return self.get_gt_for_action_with_lab_idx_mask(action=action, suff=suff)[0]

    def get_gt_for_action_with_lab_idx_mask(
        self, action: str, suff: str = ""
    ) -> tuple[np.ndarray, np.ndarray]:
        action_lab_set = set(self.all_action_lab_from_metrics())
        gt_list = []
        lab_idx_list = []
        for lab in LABS_IN_TEST_PER_ACTION[action]:
            if (action, lab) not in action_lab_set:
                continue
            p = self.gt_npy_path(action=action, lab=lab, suff=suff)
            gt = np.load(p).astype(np.int8)
            gt_list.append(gt)
            lab_idx = VIDEO_CATEGORICAL_FEATURES["lab_id"].index(lab)
            lab_idx = np.ones_like(gt) * lab_idx
            lab_idx_list.append(lab_idx)
        return np.concatenate(gt_list, axis=0), np.concatenate(lab_idx_list, axis=0)

    def sanity_check_preds(self):
        preds_dir = self.preds_dir()
        assert preds_dir.exists()
        gt_dir = self.gt_dir()
        assert gt_dir.exists()
        metrics = self.read_metrics()
        for action in ACTION_NAMES_IN_TEST:
            for lab in LABS_IN_TEST_PER_ACTION[action]:
                key = f"test-each-pr-auc/{action}+{lab}"
                pr_auc = metrics.get(key, None)
                if pr_auc is None:
                    continue
                preds = np.load(self.preds_npy_path(action=action, lab=lab)).astype(
                    np.float16
                )
                gt = np.load(self.gt_npy_path(action=action, lab=lab))
                print(f"preds: {preds.shape}, {preds.dtype}")
                print(f"gt: {gt.shape}, {gt.dtype}, sum: {np.sum(gt)}")

                pr_auc_recomputed = calc_pr_auc(y_true=gt, y_pred=preds)
                diff = np.abs(pr_auc_recomputed - pr_auc)
                msg = f"ckpt: {self.path}, action: {action}, lab: {lab}, pr_auc: {pr_auc:.7f}, pr_auc_recomputed: {pr_auc_recomputed:.7f}"
                print(msg)

    def lazy_calc_and_save_eval_data(
        self,
        feats_lookup: FeaturesLookupGPU,
        batch_size: int = 16000,
        num_workers: int = 14,
        action_lab_list: list[tuple[str, str]] | None = None,
        test_meta: pd.DataFrame | None = None,
        suff: str = "",
    ):
        is_outside_data = test_meta is not None
        assert is_outside_data == (suff != "")

        need_to_recalc_metrics = not self.metrics_path().exists()
        if is_outside_data:
            assert not need_to_recalc_metrics
            assert action_lab_list is not None

        if not need_to_recalc_metrics:
            if action_lab_list is None:
                action_lab_list = self.all_action_lab_from_metrics()
            already_have_all = True
            for action, lab in action_lab_list:
                if not self.preds_npy_path(action=action, lab=lab, suff=suff).exists():
                    already_have_all = False
                    break
                if not self.gt_npy_path(action=action, lab=lab, suff=suff).exists():
                    already_have_all = False
                    break
            if already_have_all:
                print(f"Already have all eval data in {self.path}. Skipping...")
                return
        else:
            action_lab_list = None

        train_config = self.get_train_config()
        feats_lookup.reset_configs(train_config=train_config)
        model = InferenceModel(
            ckpt_dir=self.path, feats_lookup=feats_lookup, train_config=train_config
        )

        limit_to_labs = None
        if action_lab_list is not None:
            limit_to_labs = set()
            for action, lab in action_lab_list:
                limit_to_labs.add(lab)
            limit_to_labs = list(sorted(limit_to_labs))
        if test_meta is not None:
            test_loader = test_loader_from_test_meta(
                test_meta=test_meta,
                batch_size=batch_size,
                num_workers=num_workers,
                limit_to_labs=limit_to_labs,
            )
        else:
            test_loader = test_loader_from_train_config_dl(
                train_config=train_config,
                batch_size=batch_size,
                num_workers=num_workers,
                limit_to_labs=limit_to_labs,
            )

        if need_to_recalc_metrics:
            probs_list = []
            labels_list = []
            labels_known_list = []
            lab_id_list = []

        probs_map = defaultdict(list)
        gt_map = defaultdict(list)
        for batch in tqdm(test_loader, f"Inferring for {self.path}"):
            out = model.predict_batch(batch)
            lab_id_idx = batch["lab_idx"].cpu().numpy()
            labels_known = batch["labels_known"].cpu().numpy()
            probs = out["probs"].float().cpu().numpy()
            gt = batch["labels"].float().cpu().numpy()

            if need_to_recalc_metrics:
                probs_list.append(probs)
                labels_list.append(batch["labels"].float().cpu().numpy())
                labels_known_list.append(labels_known)
                lab_id_list.append(lab_id_idx)

            lab_idx_list = list(np.unique(lab_id_idx))
            for lab_idx in lab_idx_list:
                lab_name = VIDEO_CATEGORICAL_FEATURES["lab_id"][lab_idx]
                mask_rows_lab = lab_id_idx == lab_idx
                probs_lab = probs[mask_rows_lab]
                gt_lab = gt[mask_rows_lab]
                labels_known_lab = labels_known[mask_rows_lab]
                action_idx_candidates_list = np.flatnonzero(
                    labels_known_lab.any(axis=0)
                )
                for action_idx in action_idx_candidates_list:
                    mask_rows_action = labels_known_lab[:, action_idx]
                    probs_action_lab = probs_lab[mask_rows_action, action_idx]
                    action_name = ACTION_NAMES_IN_TEST[action_idx]
                    key = (action_name, lab_name)
                    probs_map[key].append(probs_action_lab)
                    gt_action_lab = gt_lab[mask_rows_action, action_idx] > 0.5
                    gt_map[key].append(gt_action_lab)

        probs_map = {k: np.concatenate(v, axis=0) for k, v in probs_map.items()}
        gt_map = {k: np.concatenate(v, axis=0) for k, v in gt_map.items()}

        if need_to_recalc_metrics:
            assert action_lab_list is None
            probs = np.concatenate(probs_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            labels_known = np.concatenate(labels_known_list, axis=0)
            lab_ids = np.concatenate(lab_id_list, axis=0)

            computer = MetricsComputer(
                labels=labels, labels_known=labels_known, lab_ids=lab_ids
            )
            raw_metrics = computer.compute_eval_data(probs=probs).metrics
            renamed = {f"test-{key}": value for key, value in raw_metrics.items()}
            with atomic_write(self.metrics_path(), overwrite=True) as f:
                json.dump(renamed, f, indent=2)

        self.preds_dir(suff=suff).mkdir(exist_ok=True, parents=True)
        for (action, lab), probs in probs_map.items():
            dst = self.preds_npy_path(action=action, lab=lab, suff=suff)
            if not dst.exists():
                np.save(
                    dst,
                    probs.astype(np.float16),
                )

        self.gt_dir(suff=suff).mkdir(exist_ok=True, parents=True)
        for (action, lab), gt in gt_map.items():
            assert gt.dtype == bool
            dst = self.gt_npy_path(action=action, lab=lab, suff=suff)
            if not dst.exists():
                np.save(dst, gt)


def test_loader_from_test_meta(
    test_meta: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    limit_to_labs: list[str] | None = None,
):
    test_balancer = DataBalancerForLabeledTest(
        meta=test_meta, limit_to_labs=limit_to_labs
    )
    test_dataset = MiceDataset(
        data_balancer_for_labeled=test_balancer,
    )
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def test_loader_from_train_config_dl(
    train_config: DL_TrainConfig,
    batch_size: int,
    num_workers: int,
    limit_to_labs: list[str] | None = None,
):
    _, test_meta = train_test_meta_from_config_dl(train_config)
    return test_loader_from_test_meta(
        test_meta=test_meta,
        batch_size=batch_size,
        num_workers=num_workers,
        limit_to_labs=limit_to_labs,
    )


def get_train_config_dl(name: str, cv: str | int) -> DL_TrainConfig | None:
    p = get_model_cv_path(name, cv) / "train_config.json"
    if not p.exists():
        return None
    return base_model_from_file(DL_TrainConfig, p)


def get_data_size_from_config_dl(
    train_config: DL_TrainConfig, cache_dir: Path | str = "cache/data_size"
) -> tuple[int, int]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_key = base_model_to_hash(train_config.data_split_config)
    cache_key += f"remove_25fps_adaptable_snail_from_train: {train_config.remove_25fps_adaptable_snail_from_train}"
    cache_key = str(str_uint32_hash(cache_key))
    cache_path = cache_dir / f"{cache_key}.pkl"
    if cache_path.exists():
        return pickle.load(cache_path.open("rb"))

    print(
        f"===== DATA SIZE CACHE CALCULATION (will be ran once). For {train_config.name} | cv{train_config.cv()} ====="
    )
    train_meta, test_meta = train_test_meta_from_config_dl(train_config)
    if train_config.remove_25fps_adaptable_snail_from_train:
        train_meta = remove_25fps_adaptable_snail_from_meta(
            meta=train_meta, verbose=True
        )
    train_balancer = DataBalancerForLabeledTrain(
        meta=train_meta,
        balance_config=BalanceConfig(),
        aug_config=AugmentationsConfig(),
    )
    test_balancer = DataBalancerForLabeledTest(
        meta=test_meta,
    )
    res = (train_balancer.total_samples(), test_balancer.total_samples())
    pickle.dump(res, cache_path.open("wb"))
    return res


def train_test_meta_from_config_dl(
    train_config: DL_TrainConfig, meta: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_path = train_config.save_dir() / "info" / "split.json"
    if meta is None:
        meta = get_train_meta()
    if split_path.exists():
        split = json.load(open(split_path))
        train_videos = set(split["train_videos"])
        test_videos = set(split["test_videos"])
    else:
        return train_test_split(meta, train_config.data_split_config)
    train_meta = meta[meta["video_id"].isin(train_videos)]
    test_meta = meta[meta["video_id"].isin(test_videos)]
    return (train_meta, test_meta)


def is_fully_trained_according_to_saved_checkpoints(save_dir: Path | str) -> bool:
    ckpt = get_latest_saved_checkpoint(save_dir=save_dir)
    if not ckpt:
        return False
    state = ckpt.get_trainer_state()
    if not state:
        return False
    return state.global_step >= state.max_steps


def full_training_steps_from_config_dl(train_config: DL_TrainConfig) -> int:
    if train_config.max_steps > 0:
        return train_config.max_steps
    train_size, _ = get_data_size_from_config_dl(train_config)

    per_device_bs = train_config.train_bs
    ga = max(1, train_config.gradient_accumulation_steps)
    epochs = train_config.epochs

    # batches per epoch (len(train_dataloader))
    batches_per_epoch = math.ceil(train_size / per_device_bs)

    # optimizer steps per epoch
    steps_per_epoch = math.ceil(batches_per_epoch / ga)

    # total optimizer steps over all epochs
    total_steps = steps_per_epoch * epochs

    return total_steps

    # total_samples = train_size * train_config.epochs
    # bs = train_config.train_bs * train_config.gradient_accumulation_steps
    # total_steps = (total_samples + bs - 1) // bs
    # return total_steps


def total_actual_training_steps_from_config_dl(train_config: DL_TrainConfig) -> int:
    full_train_steps = full_training_steps_from_config_dl(train_config)
    cutoff = train_config.max_steps_to_run
    if cutoff is None:
        return full_train_steps
    return min(full_train_steps, cutoff)


def iter_all_dl_model_names(take_filter=None):
    for model_path in MODELS_ROOT.rglob("dl*"):
        name = model_path.name
        if take_filter is not None and not take_filter(name):
            continue
        yield name


def iter_all_configs_for_model_name_dl(name: str):
    for cv_dir in (MODELS_ROOT / name).iterdir():
        config = get_train_config_dl(name, cv_dir.name)
        if config is None:
            continue
        yield config


def iter_all_checkpoints_for_model_cv(save_dir):
    for p in save_dir.rglob("checkpoint*"):
        step = int(p.name.split("-")[-1])
        yield Ckpt(path=p, step=step)


def get_latest_saved_checkpoint(save_dir) -> Ckpt | None:
    max_ckpt = None
    for ckpt in iter_all_checkpoints_for_model_cv(save_dir):
        if max_ckpt is None or ckpt.step > max_ckpt.step:
            max_ckpt = ckpt
    return max_ckpt


def get_latest_saved_checkpoint_with_optim(save_dir) -> Ckpt | None:
    max_ckpt = None
    for ckpt in iter_all_checkpoints_for_model_cv(save_dir):
        if ckpt.has_optimizer() and (max_ckpt is None or ckpt.step > max_ckpt.step):
            max_ckpt = ckpt
    return max_ckpt


def get_best_pr_auc_ckpt(save_dir: Path | str) -> tuple[float, Ckpt]:
    save_dir = Path(save_dir)
    best = (-1.0, None)
    key = "test-prod/pr-auc"
    for ckpt in iter_all_checkpoints_for_model_cv(save_dir):
        metrics = ckpt.read_metrics()
        value = metrics[key]
        if value > best[0]:
            best = (value, ckpt)
    assert best[1] is not None
    return best


def get_best_pr_auc_ckpt_for_each_action(
    save_dir: Path | str,
) -> dict[str, tuple[float, Ckpt]]:
    save_dir = Path(save_dir)
    res = {}
    for ckpt in iter_all_checkpoints_for_model_cv(save_dir):
        try:
            metrics = ckpt.read_metrics()
        except:
            print(f"Missing metrics at {ckpt.path}")
            continue
        for action in ACTION_NAMES_IN_TEST:
            metrics_key = f"test-per-action-pr-auc/{action}"
            value = metrics.get(metrics_key)
            if value is None:
                continue
            key = action
            if key not in res or res[key][0] < value:
                res[key] = (value, ckpt)
    return res


@lru_cache
def get_best_pr_auc_ckpt_for_each_action_lab(
    save_dir: Path | str,
) -> dict[tuple[str, str], tuple[float, Ckpt]]:
    save_dir = Path(save_dir)
    res = {}
    for ckpt in iter_all_checkpoints_for_model_cv(save_dir):
        try:
            metrics = ckpt.read_metrics()
        except:
            print(f"Missing metrics at {ckpt.path}")
            continue
        for action in ACTION_NAMES_IN_TEST:
            for lab in LABS_IN_TEST_PER_ACTION[action]:
                metrics_key = f"test-each-pr-auc/{action}+{lab}"
                value = metrics.get(metrics_key)
                if value is None:
                    continue
                key = (action, lab)
                if key not in res or res[key][0] < value:
                    res[key] = (value, ckpt)
    return res


def gt_sanity_check():
    gt_per_action_lab_cv = {}
    cnt_good = 0
    for name in iter_all_dl_model_names():
        for config in iter_all_configs_for_model_name_dl(name):
            gt_dir = config.save_dir() / "gt"
            if not gt_dir.exists():
                continue
            for npy in gt_dir.iterdir():
                if not npy.name.endswith("npy"):
                    continue
                action, lab = npy.stem.split("-")
                key = (action, lab, config.cv())
                gt = np.load(npy)
                if key not in gt_per_action_lab_cv:
                    gt_per_action_lab_cv[key] = gt
                else:
                    if (gt_per_action_lab_cv[key] == gt).all():
                        cnt_good += 1
                    else:
                        raise RuntimeError(f"[FAIL] {npy}")

    print(f"cnt_good: {cnt_good}")


def step_to_ratio(config: DL_TrainConfig, step: int) -> float:
    full = full_training_steps_from_config_dl(config)
    return step / full


def ratio_to_step(config, ratio: float) -> int:
    full = full_training_steps_from_config_dl(config)
    desired_step = full * ratio
    all_ckpts = list(iter_all_checkpoints_for_model_cv(config.save_dir()))
    all_steps = [ckpt.step for ckpt in all_ckpts]
    closest_step = min(all_steps, key=lambda step: abs(step - desired_step))
    return closest_step


def get_action_lab_set_from_config_dl(config: DL_TrainConfig) -> set[tuple[str, str]]:
    ckpt = list(iter_all_checkpoints_for_model_cv(config.save_dir()))[0]
    return set(ckpt.all_action_lab_from_metrics())


def get_actions_set_from_config_dl(config: DL_TrainConfig) -> set[str]:
    ckpt = list(iter_all_checkpoints_for_model_cv(config.save_dir()))[0]
    return set(ckpt.all_actions_from_metrics())


def remove_25fps_adaptable_snail_from_meta(
    meta: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    old_train_len = len(meta)
    mask_remove = meta.apply(
        lambda video_meta: video_meta["lab_id"] == "AdaptableSnail"
        and abs(video_meta["frames_per_second"] - 25.0) < 1e-2,
        axis=1,
    )
    cnt_to_remove = sum(mask_remove)
    if verbose:
        print(f"[DATA REMOVE] {cnt_to_remove} AdaptableSnail videos with 25 fps")
    meta = meta[~mask_remove]
    new_train_len = len(meta)
    assert old_train_len - new_train_len == cnt_to_remove
    return meta
