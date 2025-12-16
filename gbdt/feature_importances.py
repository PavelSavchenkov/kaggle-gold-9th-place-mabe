from __future__ import annotations

import re
import pandas as pd  # type: ignore
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


def sanitize_filename(name: str) -> str:
    """Return a filesystem-friendly filename derived from a label/name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._") or "lab"


def trees_for_iteration(iteration: int, clf) -> int:
    """Convert boosting iterations to number of trees.

    - Binary objectives: 1 tree per iteration
    - Multi-class objectives: n_classes trees per iteration
    """
    n_classes = int(clf.n_classes_)
    if n_classes and n_classes > 2:
        return int(iteration) * n_classes
    return int(iteration)


def feature_importances_upto_iter(
    booster,
    n_trees: int | None,
    feature_names: Iterable[str],
    *,
    importance_type: str = "gain",
) -> np.ndarray:
    """Aggregate feature importance from trees up to ``n_trees``.

    Uses ``booster.trees_to_dataframe()`` and sums per-split statistics
    (Gain/Cover/Count) for non-leaf nodes across selected trees.
    """
    feature_names = list(feature_names)
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    df = booster.trees_to_dataframe()

    mask = df["Feature"] != "Leaf"
    if n_trees is not None and int(n_trees) > 0:
        mask &= df["Tree"].astype(int) < int(n_trees)
    df = df.loc[mask]

    if importance_type == "gain":
        agg = df.groupby("Feature")["Gain"].sum()
    elif importance_type == "cover":
        agg = df.groupby("Feature")["Cover"].sum()
    else:  # weight
        agg = df.groupby("Feature")["Feature"].count()

    # Map XGBoost's internal feature names (e.g., "f0") to our provided
    # feature_names when necessary. If the booster already carries the
    # DataFrame column names, they will match directly.
    out = np.zeros(len(feature_names), dtype=float)
    for feat_key, value in agg.items():
        # Exact name match first
        idx = name_to_idx.get(str(feat_key))
        if idx is None:
            # Try to parse keys like 'f12' produced when training on numpy arrays
            try:
                key = str(feat_key)
                if key.startswith("f"):
                    k = int(key[1:])
                    if 0 <= k < len(feature_names):
                        idx = k
            except Exception:
                idx = None
        if idx is not None:
            out[idx] = float(value)
    return out


def write_feature_importances(
    importances: np.ndarray, feature_names: Iterable[str], dst: Path
) -> None:
    feature_names = list(feature_names)
    order = np.argsort(importances)[::-1]
    with dst.open("w") as fh:
        for idx in order:
            fh.write(f"{feature_names[idx]}, {float(importances[idx])}\n")


def save_feature_importances(
    *,
    booster,
    clf,
    feature_names: Iterable[str],
    overall_best_iter: int,
    lab_best_iters: Dict[str, int] | None,
    out_dir: Path,
    importance_type: str = "gain",
) -> None:
    """Save feature importances at global and per-lab best steps.

    Writes to:
      - out_dir / "overall.txt"
      - out_dir / f"{lab}.txt" for each lab (sanitized lab names)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_names = list(feature_names)

    # Overall
    n_trees_overall = (
        trees_for_iteration(int(overall_best_iter), clf)
        if int(overall_best_iter) > 0
        else None
    )
    overall_imps = feature_importances_upto_iter(
        booster=booster,
        n_trees=n_trees_overall,
        feature_names=feature_names,
        importance_type=importance_type,
    )
    write_feature_importances(overall_imps, feature_names, out_dir / "overall.txt")

    # Per-lab
    lab_best_iters = lab_best_iters or {}
    for lab, step in lab_best_iters.items():
        n_trees_lab = trees_for_iteration(int(step), clf) if int(step) > 0 else None
        per_lab_imps = feature_importances_upto_iter(
            booster=booster,
            n_trees=n_trees_lab,
            feature_names=feature_names,
            importance_type=importance_type,
        )
        lab_fname = sanitize_filename(str(lab)) + ".txt"
        write_feature_importances(per_lab_imps, feature_names, out_dir / lab_fname)
