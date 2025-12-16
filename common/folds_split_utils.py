import copy
import hashlib
import json
import os
import random
from functools import lru_cache
from pathlib import Path
from random import shuffle
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo  # type: ignore
from matplotlib.pylab import rand
from skmultilearn.model_selection import IterativeStratification  # type: ignore

from common.config_utils import DataSplitConfig, SplitByStrategy
from common.constants import LAB_NAMES_IN_TEST
from common.helpers import get_annotation_by_video_meta, raise_if_not_local
from common.parse_utils import parse_behaviors_labeled


def split_folds_linear_programming(
    Y: np.ndarray,
    n_folds: int,
    balance_counts: bool,
    time_limit_s: float | None,
    threads: Optional[int] = None,
    mip_rel_gap: Optional[float] = None,
    random_state: Optional[int] = None,
    use_warm_start: bool = True,
    minimise_nan_cells_first: bool = False,
) -> np.ndarray:
    """
    Multi-feature k-way balancing via MILP (L∞ objective) with a secondary goal:
    *among solutions with equal L∞ deviation*, minimize the number of (feature, fold)
    pairs where the assigned sum is zero (i.e., spread each feature across folds when possible).

    Primary objective (scaled by n for integrality): minimize z such that
        -z <= n * sum_i Y[i,j] x[i,k] - totals[j] <= z   for all k, j
    Secondary objective: maximize count of (k, j) where sum_i Y[i,j] x[i,k] > 0.

    Set `minimise_nan_cells_first=True` to prioritize maximizing non-zero (feature, fold)
    pairs before minimizing the balance deviation.

    Returns
    -------
    assign : np.ndarray of shape (n_items,) with fold indices 0..n_folds-1
    """
    Y = np.asarray(Y, dtype=int)
    assert (Y >= 0).all(), "Y must be non-negative integers"
    n_items, n_feat = Y.shape
    n = int(n_folds)

    totals = Y.sum(axis=0).astype(int)
    # binary presence matrix per (item, feature)
    B = (Y > 0).astype(int)

    # targets only used in warm start
    targets = totals / n

    # capacities for balanced counts
    cap = np.full(n, n_items // n, dtype=int)
    cap[: (n_items % n)] += 1

    # ----- Build MILP (scaled by n to keep integer sums) -----
    m = pyo.ConcreteModel()
    I = range(n_items)
    K = range(n)
    J = range(n_feat)

    m.x = pyo.Var(I, K, domain=pyo.Binary)  # assign item i -> fold k
    m.s = pyo.Var(K, J, domain=pyo.Binary)  # 1 iff feature j appears in fold k
    m.z = pyo.Var(domain=pyo.NonNegativeReals)  # L∞ deviation * n

    # Each item in exactly one fold
    m.one_fold = pyo.Constraint(I, rule=lambda m, i: sum(m.x[i, k] for k in K) == 1)

    # Optional equal counts
    if balance_counts:

        def count_rule(m, k):
            return sum(m.x[i, k] for i in I) == int(cap[k])

        m.counts = pyo.Constraint(K, rule=count_rule)

    # L∞ constraints: -z <= n * sum_i Y[i,j] x[i,k] - totals[j] <= z
    def pos_rule(m, k, j):
        return n * sum(int(Y[i, j]) * m.x[i, k] for i in I) - int(totals[j]) <= m.z

    def neg_rule(m, k, j):
        return -m.z <= n * sum(int(Y[i, j]) * m.x[i, k] for i in I) - int(totals[j])

    m.pos = pyo.Constraint(K, J, rule=pos_rule)
    m.neg = pyo.Constraint(K, J, rule=neg_rule)

    # Link m.s[k,j] <-> "feature j appears in fold k"
    # 1) If s=0 then sum Y == 0:  sum_i Y[i,j] x[i,k] <= totals[j] * s[k,j]
    def appear_upper(m, k, j):
        return sum(int(Y[i, j]) * m.x[i, k] for i in I) <= int(totals[j]) * m.s[k, j]

    # 2) If sum of any positive-Y item assigned then s must be 1:
    #    s[k,j] <= sum_i 1{Y[i,j]>0} x[i,k]
    def appear_lower(m, k, j):
        return m.s[k, j] <= sum(int(B[i, j]) * m.x[i, k] for i in I)

    m.appear_upper = pyo.Constraint(K, J, rule=appear_upper)
    m.appear_lower = pyo.Constraint(K, J, rule=appear_lower)

    # ----- Objective: lexicographic -----
    # Depending on minimise_nan_cells_first, prioritize balancing (m.z) or maximizing
    # non-zero (feature, fold) assignments first; the larger weight enforces the priority.
    if minimise_nan_cells_first:
        weight_z = 1
        weight_secondary = int(n * n_feat + 1)
    else:
        weight_z = int(n * n_feat + 1)
        weight_secondary = 1
    m.obj = pyo.Objective(
        expr=weight_z * m.z - weight_secondary * sum(m.s[k, j] for k in K for j in J),
        sense=pyo.minimize,
    )

    # ----- Greedy warm start (optional, deterministic with random_state) -----
    if use_warm_start:
        rng = np.random.default_rng(random_state)
        order = np.argsort(-np.linalg.norm(Y, ord=1, axis=1))
        if random_state is not None:
            # Small shuffle among equal norms to remove ties deterministically
            equal_blocks = {}
            norms = np.linalg.norm(Y, ord=1, axis=1)
            for idx in order:
                equal_blocks.setdefault(norms[idx], []).append(idx)
            order = []
            for _, block in sorted(equal_blocks.items(), key=lambda kv: -kv[0]):
                rng.shuffle(block)
                order.extend(block)
            order = np.array(order, dtype=int)

        sums = np.zeros((n, n_feat), dtype=float)
        counts = np.zeros(n, dtype=int)
        assign0 = np.full(n_items, -1, dtype=int)

        for i in order:
            y = Y[i]
            best_k, best_s, best_t = None, None, None
            for k in K:
                if balance_counts and counts[k] >= cap[k]:
                    continue
                residual = (sums[k] + y) - targets
                s = np.abs(residual).max()  # L∞
                t = np.abs(residual).sum()  # L1 tie-breaker
                if (best_k is None) or (s < best_s) or (s == best_s and t < best_t):
                    best_k, best_s, best_t = k, s, t
            if best_k is None:
                best_k = int(
                    np.argmin([np.abs((sums[k] + y) - targets).max() for k in K])
                )
            assign0[i] = best_k
            sums[best_k] += y
            counts[best_k] += 1

        # Load as MIP start
        for i in I:
            for k in K:
                m.x[i, k].value = 1.0 if assign0[i] == k else 0.0

        # Warm start for s: 1 iff any item with Y[:,j]>0 is in fold k
        for k in K:
            mask_k = assign0 == k
            for j in J:
                has_pos = bool(np.any(B[mask_k, j])) if mask_k.any() else False
                m.s[k, j].value = 1.0 if has_pos else 0.0

    # ----- Solve with HiGHS -----
    opt = pyo.SolverFactory("highs")
    if opt is None or not opt.available():
        raise RuntimeError(
            "HiGHS not available. Install it with: mamba install -c conda-forge highspy"
        )

    if time_limit_s is not None:
        opt.options["time_limit"] = float(time_limit_s)
    if threads is not None:
        opt.options["threads"] = int(threads)
    if mip_rel_gap is not None:
        opt.options["mip_rel_gap"] = float(mip_rel_gap)
    if random_state is not None:
        # HiGHS supports a random seed option
        opt.options["random_seed"] = int(random_state)

    res = opt.solve(m, tee=False)

    # ----- Extract solution -----
    assign = np.empty(n_items, dtype=int)
    for i in I:
        for k in K:
            if pyo.value(m.x[i, k]) >= 0.5:
                assign[i] = k
                break

    # max_abs_dev = float(pyo.value(m.z)) / n  # undo scaling by n (if you want it)
    return assign


def split_zero_one(y, num_folds: int, seed: int):
    """
    return (N,) np array of integers 0..num_folds-1
    """

    np.random.seed(seed)

    N, _ = y.shape
    assert y.dtype == int
    assert 0 <= y.min() and y.max() <= 1

    splitter = IterativeStratification(
        n_splits=num_folds,
        order=1,  # random_state=config.seed
    )

    X = np.zeros((N,))
    fold_id = -np.ones(N, dtype=int)
    for k, (_, test_idx) in enumerate(splitter.split(X, y)):
        fold_id[test_idx] = k
    assert fold_id.min() >= 0
    return fold_id


def create_y(meta: pd.DataFrame, config: DataSplitConfig):
    N = len(meta)
    labs = list(sorted(meta.lab_id.unique()))
    y = np.zeros((N, len(labs) + len(config.actions)), dtype=np.float32)
    for i, row in enumerate(meta.to_dict(orient="records")):
        lab_id = row["lab_id"]
        annot = get_annotation_by_video_meta(video_meta=row)
        match config.split_strategy:
            case SplitByStrategy.cnt_videos:
                y[i, labs.index(lab_id)] = 1
                actions = set(
                    it.action
                    for it in parse_behaviors_labeled(row["behaviors_labeled"])
                )
                for action in actions:
                    if action not in config.actions:
                        continue
                    col = len(labs) + config.actions.index(action)
                    y[i, col] = 1
                assert np.sum(y[i, len(labs) :]) > 0
            case SplitByStrategy.cnt_segments:
                cnt_segments = 0
                for annot_row in annot.to_dict(orient="records"):
                    action = annot_row["action"]
                    if action not in config.actions:
                        continue
                    col = len(labs) + config.actions.index(action)
                    y[i, col] += 1
                    cnt_segments += 1
                y[i, labs.index(lab_id)] = cnt_segments
            case SplitByStrategy.total_duration:
                total_duration = 0.0
                fps = row["frames_per_second"]
                for annot_row in annot.to_dict(orient="records"):
                    action = annot_row["action"]
                    if action not in config.actions:
                        continue
                    duration = (
                        annot_row["stop_frame"] - annot_row["start_frame"]
                    ) / fps
                    total_duration += duration
                    col = len(labs) + config.actions.index(action)
                    y[i, col] += duration
                    total_duration += duration
                y[i, labs.index(lab_id)] = total_duration
    return y.astype(int)


def _fold_cache_key(meta: pd.DataFrame, config: DataSplitConfig) -> str:
    """
    Build a deterministic jhash from the set of video ids and the split config.
    """
    config = copy.deepcopy(config)
    config.test_fold = 0
    config.train_folds = None
    payload = {
        "video_ids": sorted(set(map(str, meta["video_id"].tolist()))),
        "config": config.model_dump(mode="json"),
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def calc_fold_id(
    meta: pd.DataFrame,
    config: DataSplitConfig,
    cache_dir: Path | str,
    force_recalc: bool = False,
    must_exist: bool = True,
):
    cache_path: Optional[Path] = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = _fold_cache_key(meta=meta, config=config)
        cache_path = cache_dir / f"fold_id_{cache_key}.npy"
        if cache_path.exists() and not force_recalc:
            cached = np.load(cache_path, allow_pickle=False)
            return cached.astype(int)
    assert not must_exist
    # try:
    #     raise_if_not_local()
    # except Exception as ex:
    #     print(ex)
    #     print(f"Current working dir: {os.getcwd()}")
    #     print(f"Its content: {os.listdir()}")
    #     print(f"cache_dir = {cache_dir}")
    #     print(f"cache_key = {cache_key}")
    #     raise ex

    y = create_y(meta, config)

    # fold_id = split_zero_one(y, config.num_folds, seed=config.seed)
    fold_params = config.folds_integer_programming_params
    fold_id = split_folds_linear_programming(
        Y=y,
        n_folds=config.num_folds,
        random_state=config.seed,
        balance_counts=fold_params.balance_counts,
        time_limit_s=fold_params.time_limit_s,
        mip_rel_gap=fold_params.mip_rel_gap,
        use_warm_start=fold_params.use_warm_start,
        minimise_nan_cells_first=fold_params.minimise_nan_cells_first,
    )

    fold_id = fold_id.astype(int)

    if cache_path is not None:
        np.save(cache_path, fold_id)

    return fold_id


def train_test_split(
    meta: pd.DataFrame,
    config: DataSplitConfig,
    cache_dir: Path | str = "split_cache",
    force_recalc: bool = False,
    must_exist: bool = True,
    remove_25fps_adaptable_snail_from_train: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    returns (train_videos_meta, test_videos_meta) data frames
    """
    assert config.test_fold < config.num_folds

    should_keep_video = meta["behaviors_labeled"].apply(
        lambda beh: any(
            item.action in config.actions for item in parse_behaviors_labeled(beh)
        )
    )

    meta = meta[should_keep_video]

    always_train_mask = ~meta["lab_id"].isin(LAB_NAMES_IN_TEST)
    always_train_rows = meta[always_train_mask].copy()
    meta = meta[~always_train_mask]

    fold_id = calc_fold_id(
        meta,
        config,
        cache_dir=cache_dir,
        force_recalc=force_recalc,
        must_exist=must_exist,
    )

    mask_test = fold_id == config.test_fold
    if config.train_folds is None:
        mask_train = ~mask_test
    else:
        mask_train = np.isin(fold_id, config.train_folds)

    return (
        pd.concat([meta[mask_train], always_train_rows], axis=0, copy=True),
        meta[mask_test].copy(),
    )
