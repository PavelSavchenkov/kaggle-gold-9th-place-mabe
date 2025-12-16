from __future__ import annotations

from functools import lru_cache
import json
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import cupy as cp  # type: ignore
import numpy as np
import pandas as pd
import xgboost as xgb  # type: ignore
from xgboost import XGBClassifier  # type: ignore

from common.folds_split_utils import train_test_split
from common.helpers import (
    ensure_1d_numpy,
    get_model_cv_path,
    get_config_path,
    get_train_meta,
)
from common.config_utils import base_model_from_file
from gbdt.configs import DataPreprocessConfig, GBDT_TrainConfig
from gbdt.features_utils import calc_features


class GBDT_Model_Type(str, Enum):
    xgboost = "xgboost"
    lightgbm = "lightgbm"
    catboost = "catboost"


def model_type_by_name(name: str) -> GBDT_Model_Type:
    if name.startswith("xgboost"):
        return GBDT_Model_Type.xgboost
    if name.startswith("lgbm"):
        return GBDT_Model_Type.lightgbm
    if name.startswith("catb"):
        return GBDT_Model_Type.catboost
    raise ValueError(f"Can not decide model type for {name}")


def get_pr_auc_metric_key(lab=None) -> str:
    return "test-prod-avg/pr-auc" if lab is None else f"test-per-lab-pr-auc/{lab}"


def get_logloss_metric_key(lab=None) -> str:
    return "test-prod-avg/log-loss" if lab is None else f"test-per-lab-log-loss/{lab}"


def get_f1_metric_key(lab=None) -> str:
    return (
        "test-prod-avg/f1-best-th" if lab is None else f"test-per-lab-f1-best-th/{lab}"
    )


def get_best_pr_auc_step(metrics: dict, lab=None) -> int:
    return metrics["best_metric_steps"][get_pr_auc_metric_key(lab)]


def get_best_logloss_step(metrics: dict, lab=None) -> int:
    return metrics["best_metric_steps"][get_logloss_metric_key(lab)]


def get_best_pr_auc_value(metrics: dict, lab=None) -> int:
    return metrics["best_metric_values"][get_pr_auc_metric_key(lab)]


def get_best_f1_value(metrics: dict, lab=None) -> int:
    return metrics["best_metric_values"][get_f1_metric_key(lab)]


def get_train_config(name, cv) -> GBDT_TrainConfig:
    return base_model_from_file(
        GBDT_TrainConfig, get_model_cv_path(name, cv) / "train_config.json"
    )


def get_data_preprocess_config_path_from_ckpt_path(ckpt: Path | str) -> Path:
    ckpt = Path(ckpt)
    return ckpt / "final_model" / "preprocess_config.json"


def get_data_preprocess_config_path(name, cv) -> Path:
    return get_data_preprocess_config_path_from_ckpt_path(get_model_cv_path(name, cv))


def read_data_preprocess_config(name, cv) -> DataPreprocessConfig:
    return base_model_from_file(
        DataPreprocessConfig, get_data_preprocess_config_path(name, cv)
    )


def get_any_train_config(name: str) -> GBDT_TrainConfig | None:
    for cv_dir in sorted((Path("train_logs") / name).iterdir()):
        cv = cv_dir.name
        if "cv" not in cv:
            continue
        return get_train_config(name, cv)
    return None


def get_gbdt_model_path(name, cv) -> Path:
    return get_model_cv_path(name, cv) / "final_model" / "model.json"


def get_test_folder_path(name: str, cv) -> Path:
    return get_model_cv_path(name, cv) / "test"


def get_test_index_path(name: str, cv) -> Path:
    return get_test_folder_path(name, cv) / "X_test_index.csv"


@lru_cache
def get_test_index_df(
    name: str, cv, usecols: tuple[str, ...] | None = None
) -> pd.DataFrame:
    if usecols is not None:
        usecols = list(usecols)  # type: ignore
    return pd.read_csv(get_test_index_path(name, cv), usecols=usecols)


def get_test_ground_truth_path(name: str, cv) -> Path:
    return get_test_folder_path(name, cv) / "y_test.npy"


@lru_cache
def get_test_ground_truth_np(name, cv, dtype=np.int8) -> np.ndarray:
    y = np.load(get_test_ground_truth_path(name, cv))
    y = ensure_1d_numpy(y)
    y = y.astype(dtype=dtype, copy=False)
    return y


def get_pred_test_path(name: str, cv, step: int) -> Path:
    return get_test_folder_path(name, cv) / f"pred_{step}.npy"


@lru_cache
def get_pred_test_np(name, cv, step, dtype=np.float32) -> np.ndarray:
    y = np.load(get_pred_test_path(name, cv, step))
    if y.ndim == 2 and y.shape[1] == 2:
        assert "catb" in name
        y = y[:, 1]
    y = ensure_1d_numpy(y)
    y = y.astype(dtype, copy=False)
    return y


def get_oof_metrics_path(name: str, folds: list[int] | None) -> Path:
    fname = "oof_metrics"
    if folds is not None:
        for f in sorted(folds):
            fname += f"_{f}"
    return Path("train_logs") / name / f"{fname}.json"


def get_metrics_path(name: str, cv) -> Path:
    return get_model_cv_path(name, cv) / "final_model" / "metrics.json"


@lru_cache
def read_metrics(name: str, cv) -> dict:
    return json.load(open(get_metrics_path(name, cv)))


def is_fully_trained(name: str, cv) -> bool:
    for fname in ["model.json", "model.txt", "model.cbm"]:
        if (get_model_cv_path(name, cv) / "final_model" / fname).exists():
            return True
    return False


def get_steps_to_save(metrics: dict) -> list[int]:
    steps = set()
    for who, step in metrics["best_metric_steps"].items():
        if "pr-auc" in who or "log-loss" in who:
            steps.add(step)
    return list(sorted(steps))


def get_feats_test(config: GBDT_TrainConfig, test_meta: pd.DataFrame, **kwargs) -> dict:
    test_feats_config = config.features_config.model_copy(deep=True)
    test_feats_config.target_fps = -1.0
    test_feats_config.hflip = False
    downsample_config_test = config.test_downsample_params.build_downsample_config(
        actions=[config.action],
        video_ids=list(test_meta["video_id"]),
    )
    call_kwargs = {
        "enable_tqdm": True,
        "threads": 8,
        "force_recompute": False,
        **kwargs,
    }
    feats = calc_features(
        test_meta,
        feats_config=test_feats_config,
        downsample_config=downsample_config_test,
        action=config.action,
        **call_kwargs,
    )
    feats["downsample_config"] = downsample_config_test
    return feats


def get_feats_test_for_config(
    config: GBDT_TrainConfig, force_recompute: bool = False, verbose: bool = False
) -> dict:
    meta = get_train_meta()
    _, test_meta = train_test_split(meta, config=config.data_split_config)
    feats = get_feats_test(
        config,
        test_meta,
        threads=8,
        enable_tqdm=verbose,
        force_recompute=force_recompute,
    )
    assert len(feats["index"]) == feats["y"].data.shape[0]
    return feats


def get_feats_test_for_model(
    name: str, cv: str | int, force_recompute: bool = False, verbose: bool = False
) -> dict:
    config = base_model_from_file(GBDT_TrainConfig, get_config_path(name, cv))
    return get_feats_test_for_config(
        config, force_recompute=force_recompute, verbose=verbose
    )


def save_downsample_csv(cfg: Any, actions: list[str], out_path: Path) -> None:
    labs = sorted(cfg.drop_rate.keys())
    cols = list(actions) + ["passive"]
    data = [
        [cfg.drop_rate.get(lab, {}).get(col, float("nan")) for col in cols]
        for lab in labs
    ]
    df = pd.DataFrame(data, index=labs, columns=cols)
    df.index.name = "lab_id"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)


def print_features_stats(
    feats: dict,
    actions: list[str],
    title: str,
    out_path: Path | str | None = None,
    sample_weight: Iterable[float] | None = None,
):
    labels: np.ndarray = feats["y"].data
    index_df: pd.DataFrame = feats["index"]

    labs = index_df["lab_id"].astype(str).to_numpy()
    unique_labs = np.unique(labs)

    n = labels.shape[0]
    weights = (
        np.asarray(list(sample_weight), dtype=np.float32)
        if sample_weight is not None
        else None
    )
    total_w = float(weights.sum()) if weights is not None else None

    # Precompute header strings and width with fixed inner alignment
    # Align: lab name, raw%, (optional) w%, and N in brackets
    lab_names = [str(l) for l in unique_labs.tolist()]
    lab_name_width = max((len(s) for s in lab_names), default=0)

    # Precompute lab sizes to get a common width for N
    lab_sizes: dict[str, int] = {}
    for lab in unique_labs:
        lab_mask = labs == lab
        lab_sizes[str(lab)] = int(lab_mask.sum())
    lab_n_width = max((len(str(v)) for v in lab_sizes.values()), default=1)

    header_strs: dict[str, str] = {}
    for lab in unique_labs:
        lab_str = str(lab)
        lab_mask = labs == lab
        lab_n = lab_sizes[lab_str]
        raw_lab_pct = (lab_n / n * 100.0) if n > 0 else 0.0
        if weights is not None:
            lab_w = float(weights[lab_mask].sum())
            lab_w_pct = (lab_w / total_w * 100.0) if total_w and total_w > 0 else 0.0
            header = (
                f"{lab_str:<{lab_name_width}} "
                f"({raw_lab_pct:6.2f}%|w={lab_w_pct:6.2f}%) "
                f"[{lab_n:>{lab_n_width}}]"
            )
        else:
            header = (
                f"{lab_str:<{lab_name_width}} "
                f"({raw_lab_pct:6.2f}%) "
                f"[{lab_n:>{lab_n_width}}]"
            )
        header_strs[lab_str] = header

    # All headers are now the same length by construction
    header_width = max((len(s) for s in header_strs.values()), default=0)

    # Assemble output
    lines: list[str] = []
    lines.append("")
    lines.append(f"*** Stats for {title} begin***")

    # Fixed, deterministic column order: passive first, then actions as provided
    columns: list[str] = ["passive", *actions]
    name_pad = max((len(c) for c in columns), default=0)

    # Precompute per-lab stats for each column and determine widths for alignment
    # stats_map[lab][col] = (count, raw_pct, w_pct_or_None, lab_n)
    stats_map: dict[str, dict[str, tuple[int, float, float | None, int]]] = {}

    # Initialize width trackers per column
    count_width: dict[str, int] = {col: 1 for col in columns}

    # Compute stats per lab and update widths
    for lab in sorted(unique_labs.tolist()):
        lab_mask = labs == lab
        lab_labels = labels[lab_mask]
        lab_n = int(lab_mask.sum())
        lab_w = float(weights[lab_mask].sum()) if weights is not None else None

        lab_stats: dict[str, tuple[int, float, float | None, int]] = {}

        # Passive: rows where all labels are 0 or -1
        passive_mask = np.all((lab_labels == 0) | (lab_labels == -1), axis=1)
        passive_count = int(passive_mask.sum())
        passive_raw_pct = (passive_count / lab_n * 100.0) if lab_n > 0 else 0.0
        if lab_w and lab_w > 0.0:
            lab_weights = weights[lab_mask]
            w_passive = float(lab_weights[passive_mask].sum())
            passive_w_pct: float | None = 100.0 * w_passive / lab_w
        else:
            passive_w_pct = None
        lab_stats["passive"] = (passive_count, passive_raw_pct, passive_w_pct, lab_n)
        count_width["passive"] = max(count_width["passive"], len(str(passive_count)))

        # Actions: positives only per column
        if lab_labels.ndim == 1:
            # Single-action case stored as 1D
            pos_counts = np.array([(lab_labels == 1).sum()], dtype=int)
        else:
            pos_counts = (lab_labels == 1).sum(axis=0).astype(int)

        for j, act in enumerate(actions):
            cnt = int(pos_counts[j])
            raw_pct = (cnt / lab_n * 100.0) if lab_n > 0 else 0.0
            if lab_w and lab_w > 0.0:
                lab_weights = weights[lab_mask]
                w_pos = float(lab_weights[(lab_labels[:, j] == 1)].sum())
                w_pct: float | None = 100.0 * w_pos / lab_w
            else:
                w_pct = None
            lab_stats[act] = (cnt, raw_pct, w_pct, lab_n)
            count_width[act] = max(count_width.get(act, 1), len(str(cnt)))

        stats_map[str(lab)] = lab_stats

    # Percentage field format is constant-width per line
    pct_fmt = "{raw:6.2f}%" if weights is None else "{raw:6.2f}%|w={w:6.2f}%"

    # Build lines per lab using fixed-width parts so columns line up
    for lab in sorted(unique_labs.tolist()):
        header = header_strs[str(lab)]
        parts: list[str] = []
        for col in columns:
            cnt, raw_pct, w_pct, lab_n = stats_map[str(lab)][col]
            # Always print all columns to preserve alignment
            if weights is None:
                pct_str = pct_fmt.format(raw=raw_pct)
            else:
                pct_str = pct_fmt.format(raw=raw_pct, w=(w_pct or 0.0))
            # name (count_aligned, pct_aligned)
            part = f"{col:<{name_pad}} ({cnt:>{count_width[col]}}, {pct_str})"
            parts.append(part)

        lines.append(f"{header:<{header_width}}: " + ", ".join(parts))

    lines.append(f"*** Stats for {title} end***")
    lines.append("")

    for line in lines:
        print(line)

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
