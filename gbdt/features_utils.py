from __future__ import annotations

import enum
import hashlib
import json
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.constants import (
    ACTIONS_TO_REMOVE,
    LAB_NAMES_IN_TEST,
    LAB_NAMES_WITH_ANNOTATION,
    MOUSE_CATEGORICAL_FEATURES,
    SYMMETRIC_BODYPART_PAIRS,
)
from common.feats_common import (
    NamedCoords,
    NamedFeatures,
    interpolate_nans_per_column,
    tracking_for_mouse_to_named_coords,
)
from common.helpers import (
    get_annotation_by_video_meta,
    get_tracking_by_video_meta,
    has_annotation_df,
)
from common.parse_utils import parse_behaviors_labeled_from_row
from gbdt.configs import FeaturesConfig
from gbdt.numeric_features_utils import (
    calc_numeric_mice_pair,
    calc_numeric_single_mouse,
    calc_O_H,
    interpolate_nans_per_column,
)
from gbdt.rebalance_utils import DownsampleConfig


@dataclass
class PairFeatures:
    actions: list[str]
    X: NamedFeatures
    y: Optional[np.ndarray]

    def rows(self):
        return self.X.data.shape[0]

    def resample_fps(self, orig_fps: float, target_fps: float) -> PairFeatures:
        assert orig_fps > 0

        n = self.X.data.shape[0]
        if target_fps <= 0 or np.isclose(orig_fps, target_fps) or n <= 1:
            return self

        frame_span = n - 1
        target_frames = int(np.round(frame_span * target_fps / orig_fps)) + 1
        target_frames = max(1, target_frames)
        if target_frames == n:
            return self

        positions = np.linspace(0.0, frame_span, target_frames, dtype=np.float32)
        base_idx = np.floor(positions).astype(int)
        next_idx = np.clip(base_idx + 1, 0, n - 1)
        alpha = (positions - base_idx)[:, None]

        X0 = self.X.data[base_idx]
        X1 = self.X.data[next_idx]
        with np.errstate(invalid="ignore"):
            X_interp = ((1.0 - alpha) * X0) + (alpha * X1)
        X_new = NamedFeatures(columns=list(self.X.columns), data=X_interp)

        nearest_idx = np.clip(np.rint(positions).astype(int), 0, n - 1)
        y_new = None if self.y is None else self.y[nearest_idx]
        return PairFeatures(actions=list(self.actions), X=X_new, y=y_new)

    def apply_downsample(
        self, downsample_config: DownsampleConfig, lab_id: str, video_id: int
    ):
        assert self.y is not None

        rng = np.random.default_rng(downsample_config.seed ^ video_id)
        drop_rates = downsample_config.drop_rate[lab_id]

        rows_to_remove = np.zeros(self.rows(), dtype=bool)

        y_cols_mask = np.zeros(self.y.shape[1], dtype=bool)
        actions = list(set(drop_rates.keys()).difference(["passive"]))
        for action in actions:
            y_cols_mask[self.actions.index(action)] = True
        y_for_drop = self.y[:, y_cols_mask]
        for action_idx, action_name in enumerate(actions):
            drop_rate = drop_rates[action_name]
            assert drop_rate >= 0
            action_indices = np.flatnonzero(y_for_drop[:, action_idx] == 1)
            if action_indices.size == 0:
                continue
            drop_flags = rng.random(action_indices.size) < drop_rate
            rows_to_remove[action_indices] |= drop_flags

        passive_drop_rate = drop_rates["passive"]
        if passive_drop_rate > 0:
            passive_mask = np.all((y_for_drop == 0) | (y_for_drop == -1), axis=1)
            passive_indices = np.flatnonzero(passive_mask)
            if passive_indices.size:
                drop_flags = rng.random(passive_indices.size) < passive_drop_rate
                rows_to_remove[passive_indices] |= drop_flags

        keep_mask = ~rows_to_remove
        self.X.data = self.X.data[keep_mask]
        self.y = self.y[keep_mask]


def calc_categorical_single_mouse(
    mouse_id: int, video_meta: dict, config: FeaturesConfig
) -> NamedFeatures:
    def normalise_value(attr: str, value: str) -> str:
        if pd.isna(value):
            return value
        if attr == "condition":
            if "lights on" in value:
                return "lights on"
            if "lights off" in value:
                return "lights off"
        return value

    def category_suffix(token: str) -> str:
        token = str(token).strip().lower()
        token = re.sub(r"[^0-9a-z]+", "_", token)
        return token.strip("_")

    columns = []
    data = []
    for attr in sorted(MOUSE_CATEGORICAL_FEATURES.keys()):
        key = f"mouse{mouse_id}_{attr}"
        value = normalise_value(attr, video_meta.get(key, pd.NA))
        categories = MOUSE_CATEGORICAL_FEATURES[attr]
        assert pd.isna(value) or value in categories
        if config.one_hot_metadata:
            for category in categories:
                columns.append(f"{attr}_is_{category_suffix(category)}")
                data.append(np.float32(value == category))
        else:
            columns.append(f"{attr}_encoded")
            data.append(
                np.float32(value if pd.isna(value) else categories.index(value))
            )

    return NamedFeatures(columns=columns, data=np.array(data, dtype=np.float32))


def calc_features_given_mouse_pair(
    agent_id: int,
    agent_p: NamedCoords,
    target_id: int,
    target_p: NamedCoords,
    video_meta: dict,
    config: FeaturesConfig,
    hflip: bool = False,
) -> PairFeatures:
    assert agent_p.is_aligned(target_p)

    cnt_frames = agent_p.x.shape[0]

    X_parts: list[NamedFeatures] = []

    def do_interpolation(p: NamedCoords, prefix: str) -> NamedCoords:
        if hflip:
            p.x = -p.x
            if config.flip_sides_if_hflip:
                pair_name = {}
                for n1, n2 in SYMMETRIC_BODYPART_PAIRS:
                    assert n1 != n2
                    pair_name[n1] = n2
                    pair_name[n2] = n1
                col_to_idx = {}
                for idx, col_name in enumerate(p.columns):
                    col_to_idx[col_name] = idx
                new_p = NamedCoords(columns=list(p.columns), x=p.x.copy(), y=p.y.copy())
                for idx, col_name in enumerate(p.columns):
                    if col_name not in pair_name:
                        continue
                    col_name_other = pair_name[col_name]
                    idx_from = col_to_idx[col_name_other]
                    new_p.x[:, idx] = p.x[:, idx_from]
                    new_p.y[:, idx] = p.y[:, idx_from]
                    new_p.columns[idx] = p.columns[idx_from]
                p = new_p

        if config.single_mouse_config.include_is_missing_keypoint:
            missing_mask = NamedFeatures(
                data=np.isnan(p.x).astype(np.float32),
                columns=[f"{prefix}_is_missing_{name}" for name in p.columns],
            )
            X_parts.append(missing_mask)

        x_interp = interpolate_nans_per_column(p.x)
        y_interp = interpolate_nans_per_column(p.y)

        return NamedCoords(columns=p.columns, x=x_interp, y=y_interp)

    p = {
        "agent": do_interpolation(agent_p, prefix="agent"),
        "target": do_interpolation(target_p, prefix="target"),
    }

    O, H = calc_O_H(p["agent"], config)

    if config.lab_id_feature:
        lab_list = LAB_NAMES_WITH_ANNOTATION
        row = np.zeros((len(lab_list)), dtype=np.float32)
        lab_id = video_meta["lab_id"]
        idx = lab_list.index(lab_id)
        row[idx] = 1.0
        row = np.broadcast_to(row, (cnt_frames, row.shape[0]))
        row = NamedFeatures(
            columns=[f"lab_id_{i}" for i in range(len(lab_list))], data=row
        )
        X_parts.append(row)

    for name, id in [("agent", agent_id), ("target", target_id)]:
        if config.single_mouse_config.include_categorical:
            cat = calc_categorical_single_mouse(id, video_meta, config)
            cat.data = np.broadcast_to(cat.data, (cnt_frames, cat.data.shape[0]))
            cat.add_prefix(name)
            X_parts.append(cat)

        with np.errstate(invalid="ignore"):
            numeric = calc_numeric_single_mouse(
                p[name],
                O=O,
                H=H,
                config=config.single_mouse_config,
                fps=video_meta["frames_per_second"],
            )
        numeric.add_prefix(name)
        X_parts.append(numeric)

        if agent_id == target_id:
            break

    if agent_id != target_id:
        with np.errstate(invalid="ignore"):
            pair_numeric = calc_numeric_mice_pair(
                agent=p["agent"],
                target=p["target"],
                config=config.mice_pair_config,
                fps=video_meta["frames_per_second"],
            )
        X_parts.append(pair_numeric)

    behaviors_labeled = parse_behaviors_labeled_from_row(video_meta)
    actions = list(
        sorted(
            set(
                beh.action
                for beh in behaviors_labeled
                if beh.agent == agent_id and beh.target == target_id
            )
        )
    )

    y: Optional[np.ndarray] = None
    if has_annotation_df(
        lab_id=str(video_meta["lab_id"]), video_id=int(video_meta["video_id"])
    ):
        y = np.zeros((cnt_frames, len(actions)), dtype=np.int8)

        annotations = get_annotation_by_video_meta(video_meta=video_meta)
        annotations = annotations[
            (annotations["agent_id"] == agent_id)
            & (annotations["target_id"] == target_id)
        ]
        for annot_row in annotations.to_dict(orient="records"):
            action = annot_row["action"]
            if action in ACTIONS_TO_REMOVE:
                continue
            s, e = annot_row["start_frame"], annot_row["stop_frame"]
            assert s <= e, annot_row
            j = actions.index(action)
            y[s:e, j] = 1

    X_all = NamedFeatures.hstack(X_parts)

    feats = PairFeatures(
        actions=list(actions),
        X=X_all,
        y=y,
    )
    feats = feats.resample_fps(
        orig_fps=video_meta["frames_per_second"],
        target_fps=config.target_fps,
    )

    X_data = feats.X.data
    if np.isinf(X_data).any():
        cnt = int(np.isinf(X_data).sum())
        print(f"NOTE: Replacing {cnt} inf values with NaN in X")
        X_data[np.isinf(X_data)] = np.nan

    return feats


def calc_features_video(
    video_meta: dict,
    feats_config: FeaturesConfig,
    mice_pairs: list[tuple[int, int]],
    downsample_config: DownsampleConfig | None = None,
    threads: int = 1,
) -> Dict[Tuple[int, int], PairFeatures]:
    assert len(mice_pairs) == len(set(mice_pairs))

    lab_id = video_meta["lab_id"]
    video_id = video_meta["video_id"]

    mice_ids = set()
    for agent, target in mice_pairs:
        mice_ids.add(agent)
        mice_ids.add(target)

    tracking = get_tracking_by_video_meta(video_meta=video_meta)
    mice_coords: dict[int, NamedCoords] = {}
    for mouse in mice_ids:
        mice_coords[mouse] = tracking_for_mouse_to_named_coords(
            tracking=tracking,
            mouse_id=mouse,
            cnt_frames=video_meta["cnt_frames"],
            bodyparts=feats_config.bodyparts,
        )

    def process(pair):
        agent, target = pair
        feats = None
        for hflip in [False, True]:
            feats_mice_pair = calc_features_given_mouse_pair(
                agent_id=agent,
                agent_p=mice_coords[agent],
                target_id=target,
                target_p=mice_coords[target],
                video_meta=video_meta,
                config=feats_config,
                hflip=hflip,
            )
            if downsample_config is not None:
                feats_mice_pair.apply_downsample(
                    downsample_config=downsample_config,
                    lab_id=lab_id,
                    video_id=video_id,
                )
            if feats is None:
                feats = feats_mice_pair
            else:
                assert feats_config.hflip
                feats.X.data = np.concatenate(
                    [feats.X.data, feats_mice_pair.X.data], axis=0
                )
                if feats.y is not None:
                    assert feats_mice_pair.y is not None
                    feats.y = np.concatenate([feats.y, feats_mice_pair.y], axis=0)
            if not feats_config.hflip:
                break
        return feats

    result: Dict[Tuple[int, int], PairFeatures] = {}
    threads = min(threads, len(mice_pairs))
    if threads == 1:
        for pair in mice_pairs:
            result[pair] = process(pair)
    else:
        assert threads > 1
        with ThreadPoolExecutor(max_workers=threads) as ex:
            for pair, feats in zip(mice_pairs, ex.map(process, mice_pairs)):
                result[pair] = feats

    return result


def calc_features(
    meta: pd.DataFrame,
    feats_config: FeaturesConfig,
    downsample_config: DownsampleConfig | None,
    action: str,
    enable_tqdm=False,
    threads: int = 1,
    cache_dir: str = "cache/calc_features",
    force_recompute: bool = False,
) -> dict:
    cache_payload = {
        "video_ids": list(meta["video_id"].astype(int)),
        "feats_config": feats_config.model_dump(),
        "downsample_config": (
            None if downsample_config is None else downsample_config.model_dump()
        ),
        "action": action,
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache_path = Path(cache_dir) / f"{cache_key}.pkl"
    try:
        if cache_path.exists() and not force_recompute:
            with cache_path.open("rb") as fh:
                ret = pickle.load(fh)
                ret["hash"] = cache_key
                return ret
    except:
        print(
            f"[WARNING] Failed to read cache from {cache_path}. Recomputing features..."
        )

    rows = meta.to_dict(orient="records")
    total = len(rows)
    assert total > 0

    def _compute(row: dict) -> tuple[str, int, Dict[Tuple[int, int], PairFeatures]]:
        beh = parse_behaviors_labeled_from_row(row)
        pairs = list(
            sorted(set((b.agent, b.target) for b in beh if b.action == action))
        )
        feats_map = calc_features_video(
            row,
            feats_config,
            downsample_config=downsample_config,
            mice_pairs=pairs,
        )
        return str(row["lab_id"]), int(row["video_id"]), feats_map

    X_parts: list[NamedFeatures] = []
    y_parts: list[NamedFeatures] = []
    index_rows: list[dict] = []

    outputs = []
    if threads == 1:
        for row in tqdm(rows, desc=f"calc_features[{action}]", disable=not enable_tqdm):
            output = _compute(row)
            outputs.append(output)
    else:
        assert threads > 1
        with ThreadPoolExecutor(max_workers=threads) as ex:
            iterator = ex.map(_compute, rows)
            if enable_tqdm:
                iterator = tqdm(iterator, total=total, desc=f"calc_features[{action}]")
            try:
                for output in iterator:
                    outputs.append(output)
            except Exception:
                ex.shutdown(wait=False, cancel_futures=True)
                raise

    for lab_id, video_id, feats_map in outputs:
        assert feats_map
        for _, pair_features in sorted(feats_map.items()):
            index_rows.extend(
                {"lab_id": lab_id, "video_id": video_id}
                for _ in range(pair_features.rows())
            )
            X_parts.append(pair_features.X)
            assert pair_features.y is not None
            j = pair_features.actions.index(action)
            ycol = pair_features.y[:, j : j + 1]
            y_parts.append(NamedFeatures(columns=[action], data=ycol))

    assert X_parts
    X_all = NamedFeatures.vstack(X_parts)

    index_df = pd.DataFrame(index_rows, columns=["lab_id", "video_id"])  # type: ignore[arg-type]

    feats: dict = {"index": index_df, "X": X_all}
    y_all = NamedFeatures.vstack(y_parts)
    feats["y"] = y_all
    feats["hash"] = cache_key

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(feats, cache_path.open("wb"))

    return feats
