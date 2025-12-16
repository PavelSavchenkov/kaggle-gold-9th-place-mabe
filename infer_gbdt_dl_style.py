import gc
import json
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common.config_utils import base_model_from_file
from common.constants import ACTION_NAMES_IN_TEST
from common.folds_split_utils import train_test_split
from common.helpers import get_train_meta
from common.metrics_common import calc_pr_auc
from common.paths import MODELS_ROOT
from dl.data_balancer import DataBalancerForLabeledTest
from dl.postprocess import get_train_config_dl, train_test_meta_from_config_dl
from gbdt.configs import FeaturesConfig
from gbdt.features_utils import calc_features_video
from gbdt.helpers import get_train_config
from infer_gbdt_submission import ModelCache
from postprocess.submission_utils import Submission

FOLDS = 5
model_cache = ModelCache(root=MODELS_ROOT, max_size=10000)
ref_dl_model_name = "dl-tcn-base8-5s-45"

import hashlib
import json
from pathlib import Path
from typing import Tuple

import numpy as np


class CachedFeatures:
    def __init__(self):
        self.base_dir = Path("cache/gbdt_feats")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: Tuple[Tuple[int, int, int], str]) -> Path:
        (video_id, agent, target), feats_hash = key
        digest = hashlib.sha1(feats_hash.encode("utf-8")).hexdigest()
        filename = f"{video_id}_{agent}_{target}_{digest}.npy"
        return self.base_dir / filename

    def get_features(
        self, video_meta: dict, feats_config: "FeaturesConfig", agent: int, target: int
    ) -> np.ndarray:
        video_id = int(video_meta["video_id"])
        key_inner = (video_id, agent, target)
        feats_hash = json.dumps(feats_config.model_dump(), sort_keys=True)
        key = (key_inner, feats_hash)

        path = self._key_to_path(key)

        # 1) Try to load from disk
        if path.exists():
            data = np.load(path, allow_pickle=False)
            # Ensure float32
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            return data

        # 2) Compute and save to disk
        feats = calc_features_video(
            video_meta=video_meta,
            feats_config=feats_config,
            mice_pairs=[(agent, target)],
            threads=1,
        )[(agent, target)]

        data = np.asarray(feats.X.data, dtype=np.float32)

        np.save(path, data)
        return data


cached_features = CachedFeatures()


@lru_cache
def get_gt(action: str, lab: str, cv: int) -> np.ndarray:
    gt_path = MODELS_ROOT / ref_dl_model_name / f"cv{cv}" / "gt" / f"{action}-{lab}.npy"
    return np.load(gt_path).astype(np.int8)


def calc_all_gbdt_preds():
    meta_all = get_train_meta()
    video_meta_by_video_id = {}
    for video_meta in meta_all.to_dict(orient="records"):
        video_meta_by_video_id[int(video_meta["video_id"])] = video_meta

    def get_probs_for_pair(
        video_id: int, agent: int, target: int, name: str, fold: int, step: int
    ):
        config = get_train_config(name=name, cv=fold)
        features_config = config.features_config
        features_config.target_fps = -1.0
        features_config.hflip = False
        data = cached_features.get_features(
            video_meta=video_meta_by_video_id[video_id],
            feats_config=features_config,
            agent=agent,
            target=target,
        )
        model = model_cache.get_model(name=name, cv=fold)
        probs = model.predict_prod(
            X=data, num_trees=step, verbose=False, threads_hint=20
        )
        assert probs.dtype == np.float32
        return probs

    @dataclass
    class TestData:
        video_pairs_list: list[tuple[int, int, int]] = field(default_factory=list)
        actions_set_per_video_id_pair: defaultdict[tuple[int, int, int], set[str]] = (
            field(default_factory=lambda: defaultdict(set))
        )

    dl_test_data_per_fold = {}
    labs_set_by_action_cv = defaultdict(set)
    for cv in tqdm(range(FOLDS), desc="Loading dl test data..."):
        ref_dl_config = get_train_config_dl(name=ref_dl_model_name, cv=cv)
        assert ref_dl_config is not None
        _, test_meta = train_test_meta_from_config_dl(ref_dl_config, meta=meta_all)
        test_balancer = DataBalancerForLabeledTest(
            meta=test_meta, limit_to_labs=None, verbose=False
        )
        test_data = TestData()
        for i in range(test_balancer.total_samples()):
            frame = test_balancer.raw_samples_list.frame[i]
            if frame == 0:
                video_id = int(test_balancer.raw_samples_list.video_id[i])
                agent = int(test_balancer.raw_samples_list.agent[i])
                target = int(test_balancer.raw_samples_list.target[i])
                test_data.video_pairs_list.append((video_id, agent, target))
                action_idx_list = test_balancer.per_pair_index[
                    (video_id, agent, target)
                ].get_known_actions_idx()
                action_names_list = [
                    ACTION_NAMES_IN_TEST[idx] for idx in action_idx_list
                ]
                test_data.actions_set_per_video_id_pair[(video_id, agent, target)] = (
                    set(action_names_list)
                )
                for action in action_names_list:
                    labs_set_by_action_cv[(action, cv)].add(
                        video_meta_by_video_id[video_id]["lab_id"]
                    )
        dl_test_data_per_fold[cv] = test_data

    submission = base_model_from_file(
        Submission, "submissions/e-nested-f1-all/submission.json"
    )
    steps_by_name = defaultdict(set)
    for model in tqdm(
        sorted(submission.models, key=lambda m: m.name),
        desc="Preparing steps_by_name for each gbdt model...",
    ):
        num_trees = 1_000_000
        for f in range(5):
            try:
                model_inference = model_cache.get_model(name=model.name, cv=f)
            except:
                continue
            cur = model_inference.get_num_trees()
            num_trees = min(num_trees, cur)

        needed_steps = set()
        for valid_fold in [*model.folds, -1]:
            steps = []
            for f, step in zip(model.folds, model.steps):
                if f != valid_fold:
                    steps.append(step)
            step = int(np.median(steps))
            step = min(step, num_trees)
            needed_steps.add(step)
        steps_by_name[model.name].update(needed_steps)

    prev_action = None
    for model in tqdm(
        sorted(submission.models, key=lambda m: m.action),
        desc="Processing gbdt models...",
    ):
        if model.action != prev_action:
            cached_features = CachedFeatures()
            gc.collect()
        prev_action = model.action
        banned_videos_per_fold = defaultdict(set)
        for f in model.folds:
            config = get_train_config(name=model.name, cv=f)
            meta_train, _ = train_test_split(
                meta=meta_all, config=config.data_split_config
            )
            videos_train = list(sorted(set(meta_train.video_id.unique())))
            banned_videos_per_fold[f] = set(videos_train)

        dst_name = f"gbdt-{model.name}"
        dst_path = MODELS_ROOT / dst_name
        dst_path.mkdir(exist_ok=True, parents=True)
        for cv in tqdm(range(FOLDS), desc="Iterate over dl folds..."):
            missing_labs_steps = set()
            missing_labs = set()
            for step in steps_by_name[model.name]:
                for lab in labs_set_by_action_cv[(model.action, cv)]:
                    preds_dir = dst_path / f"cv{cv}" / f"checkpoint-{step}" / "preds"
                    preds_npy = preds_dir / f"{model.action}-{lab}.npy"
                    if not preds_npy.exists():
                        missing_labs_steps.add((lab, step))
                        missing_labs.add(lab)
            if not missing_labs_steps:
                print(
                    f"\nInfer for {model.name} on dl fold #{cv} is ready. Continue..."
                )
                continue
            test_data: TestData = dl_test_data_per_fold[cv]
            probs_per_lab_ckpt = defaultdict(list)
            failed = False
            for video_id, agent, target in test_data.video_pairs_list:
                if failed:
                    break
                if (
                    model.action
                    not in test_data.actions_set_per_video_id_pair[
                        (video_id, agent, target)
                    ]
                ):
                    continue
                lab = video_meta_by_video_id[video_id]["lab_id"]
                if lab not in missing_labs:
                    continue
                done_infer = False
                for f in model.folds:
                    if video_id in banned_videos_per_fold[f]:
                        continue
                    assert not done_infer
                    for step in steps_by_name[model.name]:
                        if (lab, step) not in missing_labs_steps:
                            continue
                        probs = get_probs_for_pair(
                            video_id=video_id,
                            agent=agent,
                            target=target,
                            name=model.name,
                            fold=f,
                            step=step,
                        )
                        probs_per_lab_ckpt[(lab, step)].append(probs)
                    done_infer = True
                if not done_infer:
                    failed_names_path = Path("failed_names.json")
                    js = []
                    if failed_names_path.exists():
                        js = json.load(failed_names_path.open())
                    js.append(model.name)
                    js = list(sorted(set(js)))
                    json.dump(js, failed_names_path.open("w"))
                    print(f"[FAIL TO FIND OOF FOLD] name: {model.name}")
                    failed = True

            if failed:
                continue
            for (lab, step), probs_list in probs_per_lab_ckpt.items():
                preds_dir = dst_path / f"cv{cv}" / f"checkpoint-{step}" / "preds"
                preds_dir.mkdir(exist_ok=True, parents=True)
                probs = np.concatenate(probs_list, axis=0)
                preds_npy = preds_dir / f"{model.action}-{lab}.npy"
                gt = get_gt(action=model.action, lab=lab, cv=cv)
                assert (
                    gt.shape == probs.shape
                ), f"name: {model.name}, cv: {cv}, lab: {lab}, step: {step}, gt.shape: {gt.shape}, probs.shape: {probs.shape}"
                print(f"Writing to {preds_npy} ...")
                np.save(preds_npy, probs.astype(np.float16))


# calc_all_gbdt_preds()
# exit(0)

# import shutil
# failed = json.load(open("failed_names.json"))
# for name in failed:
#     print(name)
#     shutil.rmtree(MODELS_ROOT / f"gbdt-{name}")
# exit(0)

for model_path in tqdm(
    list(MODELS_ROOT.rglob("gbdt-*")),
    desc="Creating best_step_per_lab.json for each gbdt-* model...",
):
    for cv_dir in model_path.iterdir():
        assert cv_dir.is_dir()
        dst_best_step_per_lab_path = cv_dir / "best_step_per_lab.json"
        # if dst_best_step_per_lab_path.exists():
        #     print(f"{dst_best_step_per_lab_path} already exists. Continue...")
        #     continue
        cv = int(cv_dir.name[2:])
        best_step_per_lab = {}
        best_step = None
        global_action = None
        steps = set()
        for ckpt_dir in cv_dir.iterdir():
            if not ckpt_dir.is_dir():
                continue
            step = int(ckpt_dir.name[len("checkpoint-") :])
            steps.add(step)
            pred_list = []
            gt_list = []
            for npy_path in ckpt_dir.rglob("*.npy"):
                fname = npy_path.stem
                action, lab = fname.split("-")
                assert global_action is None or global_action == action
                global_action = action
                pred = np.load(npy_path).astype(np.float32)
                gt = get_gt(action=action, lab=lab, cv=cv).astype(np.int8)
                assert (
                    pred.shape == gt.shape
                ), f"pred.shape: {pred.shape}, gt.shape: {gt.shape}, name: {model_path.name}, cv: {cv}, step: {step}"
                pred_list.append(pred)
                gt_list.append(gt)
                pr_auc = calc_pr_auc(y_true=gt, y_pred=pred)
                if lab not in best_step_per_lab or best_step_per_lab[lab][1] < pr_auc:
                    best_step_per_lab[lab] = [step, pr_auc]
            pred = np.concatenate(pred_list, axis=0)
            gt = np.concatenate(gt_list, axis=0)
            pr_auc = calc_pr_auc(y_true=gt, y_pred=pred)
            if best_step is None or best_step[1] < pr_auc:
                best_step = (step, pr_auc)
        assert global_action is not None
        assert best_step is not None
        best_step_per_lab["best_step"] = best_step[0]
        best_step_per_lab["action"] = global_action
        best_step_per_lab["steps"] = list(sorted(steps))
        json.dump(best_step_per_lab, open(dst_best_step_per_lab_path, "w"))
