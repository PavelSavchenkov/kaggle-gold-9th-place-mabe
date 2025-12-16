from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import time
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.config_utils import base_model_from_file, base_model_to_file
from common.constants import ACTION_NAMES_IN_TEST, LABS_IN_TEST_PER_ACTION
from common.folds_split_utils import train_test_split
from common.helpers import get_tracking_df, get_train_meta
from common.parse_utils import parse_behaviors_labeled_from_row
from common.submission_common import (
    PredictedProbsForBehavior,
    moving_average,
    predicted_probs_to_segments,
    prepare_meta_for_inference,
)
from gbdt.configs import GBDT_TrainConfig
from gbdt.features_utils import calc_features_video
from gbdt.model import GBDT_Model
from postprocess.segments import analyze_segments
from postprocess.submission_utils import Stats, Submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_dir",
        required=True,
    )
    parser.add_argument(
        "--meta_csv",
        required=True,
    )
    parser.add_argument("--out_csv", type=str, required=False, default=None)
    parser.add_argument(
        "--model_cache_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--preds_file",
        type=str,
        default=None,
        help=".pkl file to save list of predictions",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Take first max_rows from meta csv for debugging",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--features_threads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--limit_to",
        type=str,
        default=None,
        help="path to .json file with [action, lab] pairs to limit inference to",
    )
    parser.add_argument(
        "--lab_debug", type=str, default=None, help="Infer only on this lab"
    )
    parser.add_argument("--verbose", type=int, default=0, help="0, 1, 2, 3")
    parser.add_argument("--oof", action="store_true", default=False)
    parser.add_argument("--enable_cache", action="store_true", default=False)
    parser.add_argument("--overwrite_cache", action="store_true", default=False)
    parser.add_argument("--check_inference", action="store_true", default=False)
    return parser.parse_args()


STATS = Stats()
CACHE_DIR = Path("cache/infer_ensemble")
BANNED_VIDEOS_BY_MODEL = {}

VERBOSE = 0
OOF = False
THREADS = -1
FEATURES_THREADS = -1

LAB_DEBUG = None
# LAB_DEBUG = "CautiousGiraffe"
# LAB_DEBUG = "ElegantMink"
# LAB_DEBUG = "LyricalHare"
# LAB_DEBUG = "NiftyGoldfinch"

ACTION_DEBUG = None
# ACTION_DEBUG = "approach"

LIMIT_TO = None

PER_ACTION_MODELS_FOR_SET = set()


class BaseStrideManager:
    stride: int
    cur_base: dict[tuple[int, int, int, str], int]

    def __init__(self, stride: int):
        self.stride = stride
        self.cur_base = defaultdict(int)

    def get_and_update_base(
        self, video_id: int, agent: int, target: int, action: str
    ) -> int:
        key = (video_id, agent, target, action)
        base = self.cur_base[key]
        self.cur_base[key] = (self.cur_base[key] + 1) % self.stride
        return base


BASE_STRIDE_MANAGER: BaseStrideManager | None = None


def _load_features_config_for_model(models_dir: Path, name: str, cv: int):
    cfg_path = models_dir / name / f"cv{cv}" / "train_config.json"
    train_cfg = base_model_from_file(GBDT_TrainConfig, cfg_path)
    feats_cfg = train_cfg.features_config.model_copy(deep=True)
    feats_cfg.target_fps = -1.0
    feats_cfg.hflip = False
    return feats_cfg


def _calc_features_config_hash(config) -> str:
    return json.dumps(config.model_dump(), sort_keys=True)


class ModelCache:
    def __init__(self, root: Path, max_size: int) -> None:
        self.root = root
        self.max_size = max(1, int(max_size))
        self._cache: OrderedDict[Tuple[str, int], GBDT_Model] = OrderedDict()
        self._lock = Lock()

    def get_model(self, name: str, cv: int) -> GBDT_Model:
        key = (name, cv)

        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached

        model_path = self.root / name / f"cv{cv}"
        assert model_path.exists()
        model = GBDT_Model.from_path(model_path)

        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = model

        return model


def _read_submission(models_dir: Path, submission_meta_path: Path) -> Submission:
    submission = base_model_from_file(Submission, submission_meta_path)

    for i in range(len(submission.models)):
        model = submission.models[i]
        assert len(model.folds) == len(model.steps)
        model.features_config = _load_features_config_for_model(
            models_dir=models_dir, name=model.name, cv=model.folds[0]
        )
        model.features_config_hash = _calc_features_config_hash(
            config=model.features_config
        )
        submission.models[i] = model

    if OOF:
        meta_all = get_train_meta()
        for model in tqdm(
            submission.models, desc="Filling banned videos per model for OOF"
        ):
            if LAB_DEBUG is not None and model.lab != LAB_DEBUG:
                continue

            for fold in model.folds:
                config_path = (
                    models_dir / model.name / f"cv{fold}" / "train_config.json"
                )
                config = base_model_from_file(GBDT_TrainConfig, config_path)
                if ACTION_DEBUG is not None and config.action != ACTION_DEBUG:
                    continue
                meta_train, _ = train_test_split(
                    meta=meta_all, config=config.data_split_config
                )
                videos_train = list(sorted(set(meta_train.video_id.unique())))
                # videos_train_hash = str_uint32_hash(", ".join(map(str, videos_train)))
                # print(f"HASH: {model.name}/cv{fold}: {videos_train_hash}")
                BANNED_VIDEOS_BY_MODEL[(model.name, fold)] = set(videos_train)

    return submission


def _predict_interp(
    predict_func, data: np.ndarray, base: int, stride: int
) -> np.ndarray:
    pred = predict_func(data[base::stride])
    n = len(data)
    dtype = pred.dtype
    x = np.arange(n, dtype=dtype)
    xp = x[base::stride]
    fp = pred
    pred = np.interp(x, xp, fp, left=fp[0], right=fp[-1])
    pred = pred.astype(np.float32, copy=False)
    return pred


def _predict_one_model(
    model_cache: ModelCache,
    model_name: str,
    folds: list[int],
    steps: list[int],
    X: np.ndarray,
    video_id: int,
    agent: int,
    target: int,
    action: str,
) -> tuple[np.ndarray, list[int]]:  # (probs, folds_used)
    assert len(folds) == len(steps)
    assert X.dtype == np.float32

    time_start = time.time()

    acc = np.zeros(X.shape[0], dtype=np.float32)
    folds_used = []
    sum_trees = 0
    for cv, step in zip(folds, steps):
        if OOF:
            assert video_id > 0
            if video_id in BANNED_VIDEOS_BY_MODEL[(model_name, cv)]:
                continue

        model = model_cache.get_model(model_name, cv)

        def predict(data):
            return model.predict_prod(
                X=data, num_trees=step, verbose=VERBOSE >= 3, threads_hint=THREADS
            )

        base = cv
        stride = len(folds)
        if BASE_STRIDE_MANAGER is not None:
            base = BASE_STRIDE_MANAGER.get_and_update_base(
                video_id=video_id, agent=agent, target=target, action=action
            )
            stride = BASE_STRIDE_MANAGER.stride
        pred = _predict_interp(predict, X, base=base, stride=stride)

        assert pred.dtype == np.float32
        acc += pred
        folds_used.append(cv)
        sum_trees += step

    assert folds_used

    elapsed = time.time() - time_start
    global STATS
    STATS.submit_predict(elapsed=elapsed, rows_x_trees=len(X) * sum_trees)
    if VERBOSE >= 3:
        print(
            # f"Predict on {X.shape}, {X.dtype}, folds={folds_used}, num_trees={step}, model_name={model_name}: {time.time() - time_start:.4f}"
            f"Predict on {X.shape}, {X.dtype}, folds={folds_used}, num_trees={step}: {elapsed:.4f}"
        )

    acc /= len(folds_used)
    assert acc.dtype == np.float32
    return acc, folds_used


def process_video_end2end(
    video_meta: dict, model_cache: ModelCache, submission: Submission
) -> list[PredictedProbsForBehavior]:
    lab_id = video_meta["lab_id"]
    video_id = video_meta["video_id"]
    cnt_frames = video_meta["cnt_frames"]

    if LAB_DEBUG is not None and lab_id != LAB_DEBUG:
        return []

    # group by (features_config_hash, is_self)
    behs_by_group = defaultdict(set)
    features_config_by_hash = {}
    all_behaviors = []
    for beh in parse_behaviors_labeled_from_row(row=video_meta):
        if LIMIT_TO is not None:
            if (beh.action, lab_id) not in LIMIT_TO:
                continue
        if beh.action not in ACTION_NAMES_IN_TEST:
            continue
        # if lab_id not in LABS_IN_TEST_PER_ACTION[beh.action]:
        #     continue
        if ACTION_DEBUG is not None and beh.action != ACTION_DEBUG:
            continue
        at_least_one_model = False
        for model in submission.models:
            if model.action == beh.action and model.lab == lab_id:
                key = (model.features_config_hash, beh.agent == beh.target)
                behs_by_group[key].add(beh)
                features_config_by_hash[model.features_config_hash] = (
                    model.features_config
                )
                at_least_one_model = True
        if not at_least_one_model:
            for model in submission.models:
                if model.action == beh.action and model.lab is None:
                    key = (model.features_config_hash, beh.agent == beh.target)
                    behs_by_group[key].add(beh)
                    features_config_by_hash[model.features_config_hash] = (
                        model.features_config
                    )
                    at_least_one_model = True
        if not at_least_one_model:
            continue
        # assert (
        #     at_least_one_model
        # ), f"[FAIL] Missing model for lab={lab_id}, action={beh.action}. Probably because action hasn't actually happened in trainset"
        all_behaviors.append(beh)

    probs_by_beh = {}
    cnt_probs_by_beh = defaultdict(int)
    fold_by_beh = {}  # only for OOF
    for (feats_hash, _), behaviors in sorted(behs_by_group.items()):
        assert behaviors
        behaviors = list(sorted(behaviors))
        features_config = features_config_by_hash[feats_hash]
        pairs = list(sorted(set((b.agent, b.target) for b in behaviors)))

        if VERBOSE >= 2:
            cnt_frames = video_meta["cnt_frames"]
            print(
                f"[FEATURES PREPARE] frames={cnt_frames}, pairs={len(pairs)}, total={cnt_frames * len(pairs)}"
            )

        time_start = time.time()

        feats_map = calc_features_video(
            video_meta=video_meta,
            feats_config=features_config,
            mice_pairs=pairs,
            threads=FEATURES_THREADS,
        )

        total_feats_rows = 0
        for _, feats in feats_map.items():
            total_feats_rows += len(feats.X.data)
        elapsed = time.time() - time_start
        global STATS
        STATS.submit_features(elapsed=elapsed, rows=total_feats_rows)
        if VERBOSE >= 2:
            print(f"[FEATURES   READY] rows={total_feats_rows}: {elapsed:.4f} s")

        for beh in behaviors:
            X = feats_map[(beh.agent, beh.target)].X.data
            infer_models = []
            for model in submission.models:
                if (
                    model.lab == lab_id
                    and model.action == beh.action
                    and model.features_config_hash == feats_hash
                ):
                    infer_models.append(model)
            if not infer_models:
                print(
                    f"Searching for per-action models for action={beh.action}, lab={lab_id}, video_id={video_id} ..."
                )
                for model in submission.models:
                    if (
                        model.action == beh.action
                        and model.features_config_hash == feats_hash
                        and model.lab is None
                    ):
                        PER_ACTION_MODELS_FOR_SET.add((beh.action, lab_id))
                        infer_models.append(model)
            assert infer_models
            for model in infer_models:
                probs, folds_used = _predict_one_model(
                    model_cache=model_cache,
                    model_name=model.name,
                    folds=model.folds,
                    steps=model.steps,
                    X=X,
                    video_id=video_id,
                    agent=beh.agent,
                    target=beh.target,
                    action=beh.action,
                )
                if len(folds_used) == 1:
                    if beh in fold_by_beh:
                        assert fold_by_beh[beh] == folds_used[0]
                    else:
                        fold_by_beh[beh] = folds_used[0]
                else:
                    fold_by_beh[beh] = -1
                if beh not in probs_by_beh:
                    probs_by_beh[beh] = np.zeros_like(probs)
                probs_by_beh[beh] += probs * model.coef
                cnt_probs_by_beh[beh] += 1

        gc.collect()

    # group PredictedProbsForBehavior in case of several models per one action
    result = []
    for beh in all_behaviors:
        assert beh in probs_by_beh
        fold = fold_by_beh[beh]
        threshold_fold = -1.0
        if fold != -1:
            try:
                threshold_fold = submission.get_threshold_for_fold(
                    lab=lab_id, action=beh.action, fold=fold
                )
            except:
                pass
        pred = PredictedProbsForBehavior(
            video_id=video_id,
            behavior=beh,
            probs=probs_by_beh[beh],
            threshold=submission.get_threshold(lab=lab_id, action=beh.action),
            fold=fold_by_beh[beh],
            threshold_fold=threshold_fold,
            lab_name=lab_id,
            fps=float(video_meta["frames_per_second"]),
        )
        result.append(pred)
        # cnt_probs = cnt_probs_by_beh[beh]
        # if cnt_probs > 1:
        #     print(f"Ensemble with {cnt_probs} models: lab={lab_id}, action={beh.action}")
    return result


def main():
    args = parse_args()

    assert (
        args.preds_file is not None or args.out_csv is not None
    ), "At least one of --preds_file and --out_csv should be set"

    models_dir = Path(args.models_dir)
    meta_csv = Path(args.meta_csv)

    if args.limit_to is not None:
        assert args.limit_to.endswith(".json")
        limit_list = json.load(open(args.limit_to))
        global LIMIT_TO
        LIMIT_TO = set()
        for action, lab in limit_list:
            LIMIT_TO.add((action, lab))

    global VERBOSE
    VERBOSE = args.verbose
    global OOF
    OOF = args.oof
    global THREADS
    THREADS = args.threads
    global FEATURES_THREADS
    FEATURES_THREADS = args.features_threads
    global LAB_DEBUG
    LAB_DEBUG = args.lab_debug
    global BASE_STRIDE_MANAGER
    if args.stride is not None:
        BASE_STRIDE_MANAGER = BaseStrideManager(stride=args.stride)

    print(f"VERBOSE={VERBOSE}")
    print(f"OOF={OOF}")
    print(f"THREADS={THREADS}")
    print(f"FEATURES_THREADS={FEATURES_THREADS}")
    print(f"LAB_DEBUG={LAB_DEBUG}")
    print(f"ACTION_DEBUG={ACTION_DEBUG}")
    print(f"LIMIT_TO: {None if LIMIT_TO is None else len(LIMIT_TO)} items")
    print(
        f"STRIDE: {None if BASE_STRIDE_MANAGER is None else BASE_STRIDE_MANAGER.stride}"
    )

    meta = prepare_meta_for_inference(meta_csv)
    meta = meta.sample(frac=1, random_state=42).reset_index(drop=True)
    # meta.to_csv("data/test_prepared.csv")

    meta_rows = meta.to_dict(orient="records")
    if args.max_rows is not None:
        meta_rows = meta_rows[: args.max_rows]

    # stats
    num_frames_for_features = 0
    num_frames_for_inference = 0
    progress_weight = {}
    for video_meta in meta_rows:
        cnt_frames = video_meta["cnt_frames"]
        behs = parse_behaviors_labeled_from_row(row=video_meta)
        num_pairs = len(set((b.agent, b.target) for b in behs))
        num_frames_for_features += num_pairs * cnt_frames
        num_frames_for_inference += len(behs) * cnt_frames
        progress_weight[video_meta["video_id"]] = (num_pairs + len(behs)) * cnt_frames

    print(f"Frames for features: {num_frames_for_features/1e6:.2f}M")
    print(f"Inference runs on 1M frames: {num_frames_for_inference / 1e6:.1f}")

    submission_meta_path = models_dir / "submission.json"
    cache_payload = {
        "video_ids": list(sorted(set(map(lambda row: row["video_id"], meta_rows)))),
        "submission_meta": json.load(open(submission_meta_path)),
        "LAB_DEBUG": LAB_DEBUG,
        "ACTION_DEBUG": ACTION_DEBUG,
        "STRIDE": args.stride,
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache_path = CACHE_DIR / f"{cache_key}.pkl"
    if args.enable_cache and not args.overwrite_cache and cache_path.exists():
        print(f"Reading prediction probs from cache at {str(cache_path)} ...")
        all_predictions = pickle.load(cache_path.open("rb"))
    else:
        submission = _read_submission(
            models_dir, submission_meta_path=submission_meta_path
        )

        model_cache = ModelCache(
            root=models_dir,
            max_size=int(args.model_cache_size),
        )

        all_predictions = []
        process = partial(
            process_video_end2end, model_cache=model_cache, submission=submission
        )
        iterator = map(process, meta_rows)

        total_progress_weight = sum(value for value in progress_weight.values())
        with tqdm(
            total=total_progress_weight,
            unit_scale=1e-6,
            bar_format="{l_bar}{bar}| {n:.2f}M/{total:.2f}M {elapsed}/{remaining} ({percentage:2.2f}%)",
        ) as pbar:
            for video_meta, preds in zip(meta_rows, iterator):
                all_predictions += preds

                progress_update = progress_weight[video_meta["video_id"]]
                pbar.update(progress_update)
                if VERBOSE >= 1 and preds:
                    STATS.print()

                print(f"Per-action models for: {PER_ACTION_MODELS_FOR_SET}")

        if args.enable_cache or args.overwrite_cache:
            cache_path.parent.mkdir(exist_ok=True, parents=True)
            print(f"Caching prediction probs to {str(cache_path)} ...")
            pickle.dump(all_predictions, cache_path.open("wb"))

        STATS.print(final=True)

    # old_submission = base_model_from_file(Submission, "new_submission.json")
    # analyze_segments(all_predictions, submission=old_submission)
    # base_model_to_file(old_submission, "new_new_submission.json")

    lab_id_by_video_id = {}
    for row in meta_rows:
        lab_id_by_video_id[int(row["video_id"])] = row["lab_id"]

    # submission = base_model_from_file(Submission, "new_new_submission.json")
    submission = base_model_from_file(Submission, models_dir / "submission.json")
    for pred in all_predictions:
        lab = lab_id_by_video_id[pred.video_id]
        window = submission.get_smoothing_window(lab=lab, action=pred.behavior.action)
        if window is not None:
            pred.probs = moving_average(pred.probs, window=window)

    if args.preds_file is not None:
        assert args.preds_file.endswith(".pkl")
        dst = Path(args.preds_file)
        dst.parent.mkdir(exist_ok=True, parents=True)
        pickle.dump(all_predictions, dst.open("wb"))

    if args.out_csv is not None:
        df = predicted_probs_to_segments(all_predictions)
        df.to_csv(args.out_csv)


if __name__ == "__main__":
    main()
