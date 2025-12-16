import json
import os
import time
from pathlib import Path

import cupy as cp  # type: ignore
import numpy as np
import wandb  # type: ignore
from tqdm import tqdm

from common.constants import DEFAULT_DURATION_CAP
from common.folds_split_utils import train_test_split
from common.helpers import get_train_meta
from common.config_utils import base_model_from_file, base_model_to_dict, base_model_to_file
from gbdt.configs import GBDT_TrainConfig
from gbdt.feature_importances import save_feature_importances
from gbdt.features_utils import calc_features
from gbdt.helpers import (
    GBDT_Model_Type,
    get_model_cv_path,
    get_feats_test,
    get_feats_test_for_model,
    get_pr_auc_metric_key,
    get_pred_test_path,
    get_steps_to_save,
    get_test_folder_path,
    get_test_ground_truth_np,
    get_test_ground_truth_path,
    get_test_index_path,
    get_train_config,
    is_fully_trained,
    print_features_stats,
    read_metrics,
    save_downsample_csv,
)
from gbdt.model import GBDT_Model


def fill_test_folder(
    name: str, cv, feats: dict, rewrite: bool = False, verbose: bool = False
):
    print(f"{name}/{cv}")

    X = feats["X"].data
    y_true = feats["y"].data

    assert X.dtype == np.float32

    get_test_folder_path(name, cv).mkdir(exist_ok=True, parents=True)

    index_path = get_test_index_path(name, cv)
    if not index_path.exists() or rewrite:
        feats["index"].to_csv(index_path)

    test_gt_path = get_test_ground_truth_path(name, cv)
    if not test_gt_path.exists() or rewrite:
        if verbose:
            print(f"Writing y_true for cv={cv}: {y_true.shape}")
        np.save(test_gt_path, y_true)

    model = GBDT_Model.from_path(get_model_cv_path(name, cv))
    # if model.model_type == GBDT_Model_Type.xgboost:
    #     model.gpu_on()
    #     X = cp.asarray(X)

    metrics = read_metrics(name, cv)

    steps = get_steps_to_save(metrics)
    for step in steps:
        dst = get_pred_test_path(name, cv, step)
        if dst.exists() and not rewrite:
            continue
        start_time = time.time()
        pred = model.predict_prod(X, num_trees=step)
        if verbose:
            elapsed = time.time() - start_time
            rows = len(X)
            trees = step
            time_per_1M_per_300 = elapsed * (1_000_000 / rows) * (300 / trees)
            # pr_auc = metrics["all_metric_values"][get_pr_auc_metric_key()][str(step)]
            print(f"Predict step={step:4}, wall time: {elapsed:.5f}")
            # print(f"prod-avg/pr-auc={pr_auc:.5f}")
            print(f"    time per 1M samples and 300 trees: {time_per_1M_per_300:.5f}")
        np.save(dst, pred.astype(np.float16))


def try_fill_all_test_folders_for_model(
    name: str,
    rewrite: bool = False,
    recalc_features: bool = False,
    verbose: bool = False,
    duration_cap: float = -1,
):
    for cv_dir in sorted((Path("train_logs") / name).iterdir()):
        cv = cv_dir.name
        if "cv" not in cv:
            continue
        try_fill_test_fold_for_model_cv(
            name,
            cv,
            rewrite=rewrite,
            recalc_features=recalc_features,
            verbose=verbose,
            duration_cap=duration_cap,
        )


def try_fill_test_fold_for_model_cv(
    name: str,
    cv: int | str,
    rewrite: bool = False,
    recalc_features: bool = False,
    verbose: bool = False,
    duration_cap: float = -1,
):
    if not is_fully_trained(name, cv):
        return
    metrics = read_metrics(name, cv)
    if not rewrite:
        skip = True
        for step in get_steps_to_save(metrics):
            if not get_pred_test_path(name, cv, step).exists():
                skip = False
                break
        index_path = get_test_index_path(name, cv)
        if not index_path.exists():
            skip = False
        if not get_test_ground_truth_path(name, cv).exists():
            skip = False
        if skip:
            return

    if duration_cap >= 0:
        config = get_train_config(name, cv)
        if verbose:
            print(
                f"Current duration cap: {config.test_downsample_params.duration_cap:.6f}, new: {duration_cap:.6f}"
            )
        config.test_downsample_params.duration_cap = duration_cap
        base_model_to_file(config, get_model_cv_path(name, cv) / "train_config.json")

    get_test_folder_path(name, cv).mkdir(parents=True, exist_ok=True)
    feats = get_feats_test_for_model(
        name, cv, force_recompute=recalc_features, verbose=verbose
    )
    fill_test_folder(name, cv, feats, rewrite=rewrite, verbose=verbose)


def save_stats_before_train(
    config: GBDT_TrainConfig,
    feats_train: dict,
    feats_test: dict,
    name: str,
    cv,
):
    info_dir = get_model_cv_path(name, cv) / "info"
    info_dir.mkdir(exist_ok=True, parents=True)

    print_features_stats(
        feats_train,
        [config.action],
        title="train",
        out_path=info_dir / "train_data_stats.txt",
        sample_weight=feats_train["sample_weight"],
    )
    print_features_stats(
        feats_test,
        [config.action],
        title="test",
        out_path=info_dir / "test_data_stats.txt",
        sample_weight=None,
    )
    save_downsample_csv(
        cfg=feats_train["downsample_config"],
        actions=[config.action],
        out_path=info_dir / "train_downsample.csv",
    )
    save_downsample_csv(
        cfg=feats_test["downsample_config"],
        actions=[config.action],
        out_path=info_dir / "test_downsample.csv",
    )


def train(config: GBDT_TrainConfig):
    os.environ["OMP_NUM_THREADS"] = "20"

    assert config.test_downsample_params.duration_cap == DEFAULT_DURATION_CAP

    save_dir = Path(config.save_dir)
    model_name = save_dir.parent.name
    cv = save_dir.name

    meta = get_train_meta()
    train_meta, test_meta = train_test_split(meta, config.data_split_config)

    train_meta = train_meta
    test_meta = test_meta

    downsample_config_train = config.train_downsample_params.build_downsample_config(
        actions=[config.action],
        video_ids=list(train_meta["video_id"]),
    )
    feats_train = calc_features(
        train_meta,
        feats_config=config.features_config,
        downsample_config=downsample_config_train,
        action=config.action,
        enable_tqdm=True,
        threads=8,
        force_recompute=False,
    )
    feats_train["downsample_config"] = downsample_config_train

    feats_test = get_feats_test(config, test_meta)

    info_dir = save_dir / "info"
    info_dir.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "train_videos": list(sorted(set(train_meta.video_id))),
            "test_videos": list(sorted(set(test_meta.video_id))),
        },
        open(info_dir / "split.json", "w"),
    )

    print(f"X_train dims: {feats_train['X'].data.shape}")
    print(f"X_test dims: {feats_test['X'].data.shape}")

    sample_weight_train = None
    if config.sample_coefs_params_train is not None:
        sample_weight_train = config.sample_coefs_params_train.calc_sample_coefs(
            actions=[config.action],
            feats=feats_train,
        )
    feats_train["sample_weight"] = sample_weight_train

    save_stats_before_train(
        config=config,
        feats_train=feats_train,
        feats_test=feats_test,
        name=model_name,
        cv=cv,
    )

    if config.use_wandb:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT"),
            name=config.run_name,
            group=config.group,
            config=base_model_to_dict(config),
        )

    model = GBDT_Model.from_config(config=config)

    model.fit(
        train_feats=feats_train, test_feats=feats_test, use_wandb=config.use_wandb
    )
    model.save(save_dir / "final_model")
    model.print_post_train_stats()

    metrics = model.get_after_train_metrics()
    json.dump(metrics, (save_dir / "final_model" / "metrics.json").open("w"))

    if config.use_wandb:
        wandb_metrics = {
            "best_prod-avg-pr-auc": metrics["best_metric_values"][
                "test-prod-avg/pr-auc"
            ],
            "best_step_prod-avg-pr-auc": metrics["best_metric_steps"][
                "test-prod-avg/pr-auc"
            ],
        }
        wandb.log(wandb_metrics)
        wandb.run.summary.update(wandb_metrics)
        wandb.finish()

    fill_test_folder(name=model_name, cv=cv, feats=feats_test, verbose=True)

    # feature importances
    # lab_prefix = "test-per-lab-pr-auc/"
    # best_iteration_per_lab = {
    #     k.split("/", 1)[1]: int(step)
    #     for k, step in metrics["best_metric_steps"].items()
    #     if k.startswith(lab_prefix)
    # }

    # feat_imp_dir = final_model_dir / "feature_importances"
    # feature_names = list(feats_train["X"].columns)
    # save_feature_importances(
    #     booster=booster,
    #     clf=clf,
    #     feature_names=feature_names,
    #     overall_best_iter=metrics["best_metric_steps"]["test-prod-avg/pr-auc"],
    #     lab_best_iters=best_iteration_per_lab,
    #     out_dir=feat_imp_dir,
    #     importance_type="gain",
    # )


if __name__ == "__main__":
    import argparse

    # import cProfile
    # from pstats import Stats
    # profiler = cProfile.Profile()
    # profiler.enable()

    parser = argparse.ArgumentParser("GBDT Training")
    parser.add_argument("--config_path", required=True, type=str)
    args = parser.parse_args()

    config = base_model_from_file(GBDT_TrainConfig, args.config_path)
    train(config)

    # profiler.disable()
    # Stats(profiler).strip_dirs().sort_stats("cumtime").print_stats(50)
