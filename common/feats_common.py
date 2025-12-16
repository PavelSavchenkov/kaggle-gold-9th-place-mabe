from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.constants import MOUSE_CATEGORICAL_FEATURES, VIDEO_CATEGORICAL_FEATURES


@dataclass
class NamedCoords:
    columns: list[str]
    x: np.ndarray
    y: np.ndarray

    def __init__(self, columns, x, y):
        assert len(x.shape) == 2
        n, m = x.shape
        assert y.shape == (n, m)
        assert len(columns) == m
        self.x = x
        self.y = y
        self.columns = list(columns)

    def is_aligned(self, other: NamedCoords) -> bool:
        n, m = self.x.shape
        return other.x.shape == (n, m)

    def ensure_dtype(self, dtype):
        assert self.x.dtype == dtype
        assert self.y.dtype == dtype


@dataclass
class NamedFeatures:
    columns: list[str]
    data: np.ndarray

    @staticmethod
    def hstack(feats: list["NamedFeatures"]):
        columns = []
        datas = []
        for f in feats:
            columns += f.columns
            datas.append(f.data)

        return NamedFeatures(columns=columns, data=np.concatenate(datas, axis=1))

    @staticmethod
    def vstack(feats: list["NamedFeatures"]):
        assert feats, "feats list is empty - cannot do vstack"
        return NamedFeatures(
            columns=list(feats[0].columns),
            data=np.concatenate([f.data for f in feats], axis=0),
        )

    def add_prefix(self, prefix):
        self.columns = [f"{prefix}_{col}" for col in self.columns]


def tracking_for_mouse_to_named_coords(
    tracking: pd.DataFrame,
    mouse_id: int,
    cnt_frames: int,
    bodyparts: list[str],
) -> NamedCoords:
    sub = tracking.loc[
        tracking.mouse_id == mouse_id, ["video_frame", "bodypart", "x", "y"]
    ]
    sub = sub[sub.bodypart.isin(bodyparts)]

    bp_to_col = {bp: i for i, bp in enumerate(bodyparts)}
    n = cnt_frames
    m = len(bodyparts)
    x = np.full((n, m), np.nan, dtype=np.float32)
    y = np.full((n, m), np.nan, dtype=np.float32)

    frames = sub["video_frame"].to_numpy(dtype=int)
    cols = sub["bodypart"].map(bp_to_col).to_numpy()
    xvals = sub["x"].to_numpy(dtype=np.float32)
    yvals = sub["y"].to_numpy(dtype=np.float32)

    x[frames, cols] = xvals
    y[frames, cols] = yvals
    return NamedCoords(columns=list(bodyparts), x=x, y=y)


def interpolate_rows_in_2d(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    assert X.ndim == 2
    assert X.dtype == np.float32
    assert t.ndim == 1

    i0 = np.floor(t).astype(int)
    i1 = i0 + 1
    coef = (t - i0)[:, None]

    i0 = np.clip(i0, 0, X.shape[0] - 1)
    i1 = np.clip(i1, 0, X.shape[0] - 1)

    left = X[i0]
    right = X[i1]

    res = (1 - coef) * left + coef * right
    return res.astype(np.float32, copy=False)


def interpolate_nans_per_column(arr: np.ndarray) -> np.ndarray:
    """Interpolate NaNs in each column; zero-fill outside observed spans."""
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")

    dtype = arr.dtype
    filled = np.array(arr, dtype=dtype, copy=True)

    t = np.arange(filled.shape[0], dtype=dtype)
    for col_idx in range(filled.shape[1]):
        column = filled[:, col_idx]
        mask_nan = np.isnan(column)
        if mask_nan.all():
            continue
        xp = t[~mask_nan]
        fp = column[~mask_nan]
        interp = np.interp(t, xp, fp, left=0.0, right=0.0)
        interp[~mask_nan] = fp
        filled[:, col_idx] = interp
    return filled


def interpolate_all_nans_per_column(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")

    dtype = arr.dtype
    filled = np.array(arr, dtype=dtype, copy=True)

    t = np.arange(filled.shape[0], dtype=dtype)
    for col_idx in range(filled.shape[1]):
        column = filled[:, col_idx]
        mask_nan = np.isnan(column)
        if mask_nan.all():
            filled[:, col_idx] = 0.0
            continue
        xp = t[~mask_nan]
        fp = column[~mask_nan]
        interp = np.interp(t, xp, fp)
        interp[~mask_nan] = fp
        filled[:, col_idx] = interp
    return filled


def safe_index_lookup(container, value) -> int:
    if value not in container:
        if None in container:
            return container.index(None)
        return 0
    return container.index(value)


def get_categorical_map_for_mouse(mouse_id: int, video_meta: dict) -> dict[str, int]:
    def normalise_value(attr: str, value: str) -> str:
        if attr == "condition":
            if "lights on" in value:
                return "lights on"
            if "lights off" in value:
                return "lights off"
        return value

    res = {}
    for attr, options in MOUSE_CATEGORICAL_FEATURES.items():
        key = f"mouse{mouse_id}_{attr}"
        value = None
        if key in video_meta and not pd.isna(video_meta[key]):
            value = normalise_value(attr, video_meta[key])
            if not isinstance(value, str):
                value = None
        res[attr] = safe_index_lookup(options, value)

    return res


def get_categorical_map_for_video(video_meta: dict) -> dict[str, int]:
    res = {}
    for attr, options in VIDEO_CATEGORICAL_FEATURES.items():
        options = VIDEO_CATEGORICAL_FEATURES[attr]
        value = None
        if attr in video_meta and not pd.isna(video_meta[attr]):
            value = video_meta[attr]
            if not isinstance(value, str):
                value = None
        if value == "split rectangular":
            value = "split rectangluar"
        res[attr] = safe_index_lookup(options, value)
    return res
