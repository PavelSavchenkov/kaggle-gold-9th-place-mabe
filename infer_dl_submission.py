import argparse
import gc
import hashlib
import json
import os
import pickle
import secrets
import sys
import time
from collections import defaultdict
from pathlib import Path
from pkgutil import get_data

import numpy as np
import pandas as pd
import torch  # type: ignore
from tqdm import tqdm

from common.config_utils import base_model_from_file
from common.constants import (
    ACTION_NAMES_IN_TEST,
    LABS_IN_TEST_PER_ACTION,
    VIDEO_CATEGORICAL_FEATURES,
)
from common.helpers import get_visible_gpu_ids
from common.parse_utils import parse_behaviors_labeled_from_row
from common.spawn_runner import spawn
from common.submission_common import (
    PredictedProbsForBehavior,
    predicted_probs_to_segments,
    prepare_meta_for_inference,
)
from dl.choose_avg_morph import PostprocessingParams, evaluate_postprocessing_and_save
from dl.configs import AugmentationsConfig, FeaturesConfigDL
from dl.features_lookup_gpu import FeaturesLookupGPU
from dl.model_helpers import InferenceModel
from dl.pairs_postprocessor import maximise_f1_one_action_pairwise_oof
from dl.segments import analyse_segments_dl
from dl.submission import Submission, SubmissionModel

VERBOSITY = 0
OOF = False
CACHE_DIR = Path("cache/infer_ensemble")
BATCH_SIZE = -1
STRIDE = -1
HFLIP = False

LAB_DEBUG = None
# LAB_DEBUG = "CautiousGiraffe"

LIMIT_TO = None


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
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        required=False,
        help=".csv file to save postprocessed preds",
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
        "--tmp_dir",
        type=str,
    )
    parser.add_argument(
        "--lab_debug", type=str, default=None, help="Infer only on this lab"
    )
    parser.add_argument(
        "--hflip", action="store_true", help="hflip TTA for all samples"
    )
    parser.add_argument(
        "--limit_to",
        type=str,
        default=None,
        help="path to .json file with [action, lab] pairs to limit inference to",
    )
    parser.add_argument("--stride", type=int, required=False, default=1)
    parser.add_argument("--batch_size", type=int, required=False, default=1000)
    parser.add_argument("--oof", action="store_true", default=False)
    parser.add_argument("--verbose", type=int, default=0, help="0, 1, 2, 3")
    parser.add_argument("--enable_cache", action="store_true", default=False)
    parser.add_argument("--overwrite_cache", action="store_true", default=False)
    return parser.parse_args()


def memory_cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    # (Optional) Clean up inter-process memory handles, if you use multiprocessing
    torch.cuda.ipc_collect()


def _predict_interp(
    predict_func, dataset: dict[str, torch.Tensor], base: int, stride: int
) -> np.ndarray:
    assert base >= 0
    assert stride >= 1
    dataset_strided = {}
    N_strided = None
    N = None
    for k, v in dataset.items():
        N = v.size(0)
        v_strided = v[base::stride]
        N_strided = v_strided.size(0)
        dataset_strided[k] = v_strided

    for v in dataset_strided.values():
        assert v.size(0) == N_strided
    pred = predict_func(dataset_strided, n=N_strided)

    dtype = pred.dtype
    x = np.arange(N, dtype=dtype)
    xp = x[base::stride]
    fp = pred
    pred = np.interp(x, xp, fp, left=fp[0], right=fp[-1])
    pred = pred.astype(np.float32, copy=False)
    return pred


def generate_preds_single_process(
    models_dir: Path, meta: pd.DataFrame
) -> list[PredictedProbsForBehavior]:
    if meta.size == 0:
        return []

    submission = base_model_from_file(Submission, models_dir / "submission.json")

    meta_rows = meta.to_dict(orient="records")
    if LAB_DEBUG is not None:
        meta_rows = [row for row in meta_rows if row["lab_id"] == LAB_DEBUG]

    BANNED_VIDEOS_BY_MODEL_CV = defaultdict(set)
    if OOF:
        for model in submission.models:
            if LAB_DEBUG is not None and model.lab != LAB_DEBUG:
                continue
            for fold in model.folds():
                with open(
                    model.get_cv_dir(models_dir=models_dir, fold=fold)
                    / "info"
                    / "split.json"
                ) as f:
                    split = json.load(f)
                    BANNED_VIDEOS_BY_MODEL_CV[(model.name(), fold)] = set(
                        split["train_videos"]
                    )

    feats_lookup = FeaturesLookupGPU(
        meta=meta_rows,
        feats_config=FeaturesConfigDL(),
        aug_config=AugmentationsConfig(),
    )

    video_meta_by_video_id = {}
    preds_by_action_lab = defaultdict(list)
    total_samples_by_action_lab = defaultdict(int)
    total_samples = 0
    all_preds = []
    per_action_models_for_set = set()
    for video_meta in meta_rows:
        cnt_frames = video_meta["cnt_frames"]
        video_id = int(video_meta["video_id"])
        lab = str(video_meta["lab_id"])
        video_meta_by_video_id[video_id] = video_meta
        behs = parse_behaviors_labeled_from_row(row=video_meta)
        for beh in behs:
            if beh.action not in ACTION_NAMES_IN_TEST:
                continue
            # if lab not in LABS_IN_TEST_PER_ACTION[beh.action]:
            #     continue
            key = (beh.action, lab)
            if LIMIT_TO is not None:
                if key not in LIMIT_TO:
                    continue

            models = submission.get_all_models_for(action=beh.action, lab=lab)
            assert models
            th = submission.get_threshold_for(action=beh.action, lab=lab)
            empty_pred = PredictedProbsForBehavior(
                video_id=video_id,
                behavior=beh,
                threshold=th,
                probs=np.zeros(cnt_frames, dtype=np.float32),
                lab_name=lab,
                fps=float(video_meta["frames_per_second"]),
            )
            preds_by_action_lab[key].append(empty_pred)
            all_preds.append(empty_pred)

            samples = 0
            for model in models:
                assert model.name()
                if model.lab is None:
                    per_action_models_for_set.add((model.action, lab))
                if OOF:
                    for fold in model.folds():
                        if (
                            video_id
                            not in BANNED_VIDEOS_BY_MODEL_CV[(model.name(), fold)]
                        ):
                            samples += cnt_frames
                else:
                    samples += len(model.folds()) * cnt_frames
            if HFLIP:
                samples *= 2
            total_samples_by_action_lab[(beh.action, lab)] += samples
            total_samples += samples

    print(f"Per-action models for (action, lab): {per_action_models_for_set}")

    progress_bar = tqdm(
        total=total_samples,
        unit_scale=1e-6,
        bar_format=f"{{l_bar}}{{bar}}| {{n:.2f}}M/{total_samples/1e6:.2f}M {{elapsed}}/{{remaining}} ({{percentage:2.2f}}%)",
    )

    action_lab_list = list(preds_by_action_lab.keys())
    action_lab_list.sort(key=lambda k: total_samples_by_action_lab[k], reverse=True)

    for action, lab in action_lab_list:
        preds_list = preds_by_action_lab[(action, lab)]
        action_idx = ACTION_NAMES_IN_TEST.index(action)

        def batched_inference(
            inference_model: InferenceModel,
            dataset: dict[str, torch.Tensor],
            N: int,
        ) -> torch.Tensor:
            total_probs = torch.zeros(N, dtype=torch.float32)
            assert BATCH_SIZE >= 1
            for l in range(0, N, BATCH_SIZE):
                r = min(N, l + BATCH_SIZE)

                batch = {}
                for key, tensor in dataset.items():
                    batch[key] = tensor[l:r].to(inference_model.device)

                out = inference_model.predict_batch(batch=batch)

                batch_probs = out["probs"][:, action_idx].to(torch.float32).cpu()
                total_probs[l:r] += batch_probs

                progress_bar.update((r - l) * STRIDE)
            return total_probs

        models = submission.get_all_models_for(action=action, lab=lab)
        folds = models[0].folds()
        assert folds
        for model in models:
            assert folds == model.folds()
        dataset_per_pred = {}

        def get_dataset_for_pred(
            pred: PredictedProbsForBehavior,
        ) -> dict[str, torch.Tensor]:
            video_id = pred.video_id
            agent = pred.behavior.agent
            target = pred.behavior.target
            key = pred.key()
            if key in dataset_per_pred:
                return dataset_per_pred[key]
            assert pred.behavior.action == action
            cnt_frames = video_meta_by_video_id[video_id]["cnt_frames"]
            lab_idx = VIDEO_CATEGORICAL_FEATURES["lab_id"].index(lab)

            def arr_same(val: int) -> torch.Tensor:
                return torch.ones(cnt_frames, dtype=torch.long, device=device) * val

            dataset = {}
            dataset["video_id"] = arr_same(video_id)
            dataset["agent"] = arr_same(agent)
            dataset["target"] = arr_same(target)
            dataset["frame"] = torch.arange(cnt_frames, dtype=torch.long, device=device)
            dataset["lab_idx"] = arr_same(lab_idx)
            dataset_per_pred[key] = dataset
            return dataset

        base = 0
        for model in models:
            probs_per_pred = defaultdict(list)
            fold_per_pred = {}
            for fold in folds:
                train_config = model.get_config(models_dir=models_dir, fold=fold)
                feats_lookup.reset_configs(train_config=train_config)
                inference_model = InferenceModel(
                    ckpt_dir=model.get_ckpt_dir(models_dir=models_dir, fold=fold),
                    feats_lookup=feats_lookup,
                    train_config=train_config,
                    verbose=False,
                )
                device = inference_model.device
                for pred in preds_list:
                    if OOF:
                        if (
                            pred.video_id
                            in BANNED_VIDEOS_BY_MODEL_CV[(model.name(), fold)]
                        ):
                            continue
                    dataset = get_dataset_for_pred(pred)

                    hflips = [False]
                    if HFLIP:
                        hflips.append(True)
                    for hflip in hflips:

                        def predict_func(
                            data: dict[str, torch.Tensor], n: int
                        ) -> np.ndarray:
                            if hflip:
                                inference_model.model.landmarks_feature_extractor.force_hflip()
                            res = batched_inference(
                                inference_model=inference_model, dataset=data, N=n
                            ).numpy()
                            if hflip:
                                inference_model.model.landmarks_feature_extractor.remove_force_hflip()
                            return res

                        predicted_probs = _predict_interp(
                            predict_func=predict_func,
                            dataset=dataset,
                            base=base,
                            stride=STRIDE,
                        )

                        probs_per_pred[pred.key()].append(predicted_probs)
                    fold_per_pred[pred.key()] = fold
                base = (base + 1) % STRIDE
                memory_cleanup()
            for pred in preds_list:
                key = pred.key()
                if OOF:
                    if key not in probs_per_pred:
                        continue
                assert len(probs_per_pred[key]) > 0
                probs = np.average(probs_per_pred[key], axis=0)
                pred.probs += probs * model.coef
                pred.fold = fold_per_pred[pred.key()]

    progress_bar.close()

    return all_preds


def get_cache_path(args, meta: pd.DataFrame) -> Path:
    video_ids = list(sorted(int(video_id) for video_id in meta["video_id"].unique()))
    cache_payload = {
        "video_ids": video_ids,
        "submission_meta": json.load(open(Path(args.models_dir) / "submission.json")),
        "LAB_DEBUG": LAB_DEBUG,
        "OOF": OOF,
        "LIMIT_TO": (
            None if LIMIT_TO is None else list(sorted([list(x) for x in LIMIT_TO]))
        ),
        "STRIDE": STRIDE,
        "HFLIP": HFLIP,
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return CACHE_DIR / f"{cache_key}.pkl"


def main_get_predictions(args, meta: pd.DataFrame) -> list[PredictedProbsForBehavior]:
    if args.enable_cache:
        cache_path = get_cache_path(args=args, meta=meta)
        if not args.overwrite_cache and cache_path.exists():
            print(f"Reading prediction probs from cache at {str(cache_path)} ...")
            with cache_path.open("rb") as f:
                return pickle.load(f)

    models_dir = Path(args.models_dir)

    gpu_ids = get_visible_gpu_ids()
    assert gpu_ids

    if len(gpu_ids) == 1:
        all_predictions = generate_preds_single_process(models_dir, meta=meta)
    else:
        assert args.tmp_dir is not None, "--tmp_dir is needed for multi-gpu inference"

        print()
        print(f"===== RUNNING MULTIGPU ({len(gpu_ids)}) PROCESSING =====")
        print()

        tmp_dir = Path(args.tmp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)

        SCRIPT_PATH = os.path.abspath(__file__)

        pref_name = str(secrets.randbits(32))
        processes = []
        shard_preds_pkl_list = []

        def push_not_started_process_for_meta(shard_meta: pd.DataFrame, gpu_id: str):
            idx = len(processes)

            csv = tmp_dir / f"{pref_name}_shard_meta_{idx}.csv"
            shard_meta.to_csv(csv, index=False)
            pkl = tmp_dir / f"{pref_name}_shard_preds_{idx}.pkl"
            shard_preds_pkl_list.append(pkl)

            argv = [
                sys.executable,
                SCRIPT_PATH,
                "--models_dir",
                str(models_dir),
                "--meta_csv",
                str(csv),
                "--preds_file",
                str(pkl),
                "--batch_size",
                str(BATCH_SIZE),
                "--verbose",
                str(VERBOSITY),
                "--stride",
                str(STRIDE),
            ]
            if args.oof:
                argv += ["--oof"]
            if args.enable_cache:
                argv += ["--enable_cache"]
            if args.overwrite_cache:
                argv += ["--overwrite_cache"]
            if args.lab_debug:
                argv += ["--lab_debug", str(args.lab_debug)]
            if args.limit_to:
                argv += ["--limit_to", str(args.limit_to)]
            if args.hflip:
                argv += ["--hflip"]

            env_updates = {"CUDA_VISIBLE_DEVICES": gpu_id}
            p = spawn(argv=argv, env_updates=env_updates, start=False)
            processes.append(p)

        for idx in range(len(gpu_ids)):
            cur_meta = meta.iloc[idx :: len(gpu_ids)]
            if cur_meta.size == 0:
                continue
            push_not_started_process_for_meta(shard_meta=cur_meta, gpu_id=gpu_ids[idx])

        for p in processes:
            p.start()

        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Child process exited with code {p.exitcode}")

        all_predictions = []
        for pkl in shard_preds_pkl_list:
            assert pkl.exists(), str(pkl)
            with pkl.open("rb") as f:
                shard_predictions = pickle.load(f)
                all_predictions += shard_predictions

    if args.enable_cache:
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        with cache_path.open("wb") as f:
            pickle.dump(all_predictions, f)

    return all_predictions


def main():
    args = parse_args()

    assert (
        args.preds_file is not None or args.out_csv is not None
    ), "At least one of --preds_file and --out_csv should be set"

    if args.overwrite_cache and not args.enable_cache:
        raise RuntimeError(
            "You have to set enable_cache on in order to have overwrite_cache on"
        )

    if args.limit_to is not None:
        assert args.limit_to.endswith(".json")
        limit_list = json.load(open(args.limit_to))
        global LIMIT_TO
        LIMIT_TO = set()
        for action, lab in limit_list:
            LIMIT_TO.add((action, lab))

    global VERBOSITY
    VERBOSITY = args.verbose
    global OOF
    OOF = args.oof
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size
    assert BATCH_SIZE >= 1
    global STRIDE
    STRIDE = args.stride
    assert STRIDE >= 1
    global LAB_DEBUG
    LAB_DEBUG = args.lab_debug
    global HFLIP
    HFLIP = args.hflip

    print()
    print(f"VERBOSITY={VERBOSITY}")
    print(f"OOF={OOF}")
    print(f"BATCH_SIZE={BATCH_SIZE}")
    print(f"STRIDE={STRIDE}")
    print(f"LAB_DEBUG={LAB_DEBUG}")
    print(f"LIMIT_TO: {None if LIMIT_TO is None else len(LIMIT_TO)} items")
    print(f"HFLIP={HFLIP}")
    print()

    meta_csv = Path(args.meta_csv)
    meta = prepare_meta_for_inference(meta_csv)
    meta = meta.sample(frac=1, random_state=42).reset_index(drop=True)
    # meta.to_csv("data/test_prepared.csv")
    if args.max_rows is not None:
        meta = meta.iloc[: args.max_rows]

    all_predictions = main_get_predictions(args=args, meta=meta)

    # analyse_segments_dl(predictions=all_predictions)
    # video_meta_by_video_id = {}
    # for video_meta in meta.to_dict(orient="records"):
    #     video_meta_by_video_id[int(video_meta["video_id"])] = video_meta
    # for pred in all_predictions:
    #     if pred.fps is None:
    #         pred.fps = float(video_meta_by_video_id[pred.video_id]["frames_per_second"])
    # param_grid = []
    # for first in [False, True]:
    #     for window_avg in [3, 5, 7, 9]:
    #         for window_morph in [3, 5, 7, 9]:
    #             param_grid.append(
    #                 PostprocessingParams(
    #                     moving_average_window_30fps=window_avg,
    #                     morph_kernel_30fps=window_morph,
    #                     morph_first=first,
    #                 )
    #             )
    # evaluate_postprocessing_and_save(
    #     predictions=all_predictions,
    #     param_grid=param_grid,
    #     json_out_path="optimal_params.json",
    #     max_workers=8,
    # )

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
