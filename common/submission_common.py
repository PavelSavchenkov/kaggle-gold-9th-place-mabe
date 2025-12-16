from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.config_utils import base_model_from_file
from common.constants import LABS_IN_TEST_PER_ACTION
from common.helpers import get_tracking_df
from common.parse_utils import BehaviorLabeled
from dl.pairs_postprocessor import PairwiseResolverParams, _select_pairwise_tournament


@dataclass
class PredictedProbsForBehavior:
    video_id: int
    behavior: BehaviorLabeled
    threshold: float
    probs: np.ndarray
    lab_name: str
    fps: float
    fold: int | None = None
    threshold_fold: float | None = None

    def key(self):
        return (
            self.video_id,
            self.behavior.agent,
            self.behavior.target,
            self.behavior.action,
        )


def prepare_meta_for_inference(meta_path: Path) -> pd.DataFrame:
    meta = pd.read_csv(meta_path, low_memory=False)
    assert not meta.empty

    if "cnt_frames" in meta.columns:
        return meta

    cnt_frames = []
    for lab_id, video_id in tqdm(
        meta[["lab_id", "video_id"]].itertuples(index=False, name=None),
        total=len(meta),
        desc="Meta csv preparation",
    ):
        track = get_tracking_df(lab_id, video_id)
        max_frame = track["video_frame"].max()
        cnt_frames.append(max_frame + 1)

    meta["cnt_frames"] = cnt_frames
    return meta


@dataclass
class Segment:
    video_id: int
    agent_id: str
    target_id: str
    action: str
    start_frame: int
    stop_frame: int


def pair_probs_to_segments(
    predictions: list[PredictedProbsForBehavior],
    params: PairwiseResolverParams,
) -> list[Segment]:
    n = len(predictions)
    assert n > 0

    assert all(
        pred.behavior.agent == predictions[0].behavior.agent
        and pred.behavior.target == predictions[0].behavior.target
        and pred.video_id == predictions[0].video_id
        for pred in predictions
    )
    video_id = predictions[0].video_id
    lab_name = predictions[0].lab_name

    # predictions = [
    #     pred
    #     for pred in predictions
    #     if lab_name in LABS_IN_TEST_PER_ACTION[pred.behavior.action]
    # ]
    # if not predictions:
    #     return []

    for pred in predictions:
        if lab_name not in LABS_IN_TEST_PER_ACTION[pred.behavior.action]:
            eps = 1e-2
            pred.probs = np.where(pred.probs >= pred.threshold, eps, 0.0)
            pred.threshold = eps / 2

    actions = [pred.behavior.action for pred in predictions]
    assert len(actions) == len(
        set(actions)
    ), "Duplicate actions within a (video,agent,target) group."

    probs = np.concatenate(
        [pred.probs[:, None].copy() for pred in predictions], axis=1
    )  # (T,K)
    thresholds = np.asarray(
        [float(pred.threshold) for pred in predictions], dtype=np.float32
    )
    lab = str(predictions[0].lab_name)

    min_prob = float(np.min(probs))
    max_prob = float(np.max(probs))
    assert (
        -1e-1 <= min_prob <= max_prob <= 1 + 1e-1
    ), f"min_prob={min_prob:.7f}, max_prob={max_prob:.7f}"

    # --- Pairwise tournament selection (returns (T,) with -1 meaning "emit nothing") ---
    chosen = _select_pairwise_tournament(
        probs=probs,
        thresholds=thresholds,
        actions=actions,
        lab=lab,
        params=params,
    )

    # --- Run-length encoding to segments ---
    segs: list[Segment] = []
    T = probs.shape[0]
    l = 0
    while l < T:
        r = l + 1
        while r < T and chosen[r] == chosen[l]:
            r += 1

        idx = int(chosen[l])
        if idx != -1:
            agent = predictions[idx].behavior.agent_str()
            target = predictions[idx].behavior.orig_target
            action = predictions[idx].behavior.action
            segs.append(
                Segment(
                    video_id=video_id,
                    agent_id=agent,
                    target_id=target,
                    action=action,
                    start_frame=int(l),
                    stop_frame=int(r),
                )
            )

        l = r

    return segs


def predicted_probs_to_segments(
    predictions: list[PredictedProbsForBehavior],
    pairs_params_json_path: Path | str | None = None,
) -> pd.DataFrame:
    if pairs_params_json_path is not None:
        print(f"[PAIRS JSON] Using pairs json from {pairs_params_json_path}")
        params = PairwiseResolverParams.from_json_file(pairs_params_json_path)
    else:
        params = PairwiseResolverParams()  # fallback to argmax

    groups = defaultdict(list)
    for pred in predictions:
        groups[(pred.video_id, pred.behavior.agent, pred.behavior.target)].append(pred)

    all_segs: list[Segment] = []
    for _, preds_list in tqdm(
        groups.items(), desc="Postprocessing probs into segments ..."
    ):
        all_segs.extend(
            pair_probs_to_segments(
                predictions=preds_list,
                params=params,
            )
        )

    print(f"Segments in submission: {len(all_segs)}")

    df = pd.DataFrame.from_records([asdict(seg) for seg in all_segs])
    df.index.name = "row_id"
    return df


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    x_padded = np.pad(
        x, pad_width=pad, mode="constant"
    )  # 'edge', 'reflect', 'constant', etc.
    kernel = np.ones(window) / window
    return np.convolve(x_padded, kernel, mode="valid")
