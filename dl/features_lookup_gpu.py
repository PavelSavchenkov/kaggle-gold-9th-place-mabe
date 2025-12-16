from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch  # type: ignore
from tqdm import tqdm

from common.feats_common import (
    get_categorical_map_for_mouse,
    get_categorical_map_for_video,
    interpolate_all_nans_per_column,
    tracking_for_mouse_to_named_coords,
)
from common.helpers import get_default_cuda_float_dtype, get_tracking_by_video_meta
from common.parse_utils import parse_behaviors_labeled_from_row
from dl.configs import AugmentationsConfig, DL_TrainConfig, FeaturesConfigDL


@dataclass
class TrackMeta:
    video_id: int
    mouse_id: int
    offset: int
    length: int
    fps: float


class FeaturesLookupGPU:
    """
    Converts batch of (video_id, agent, target, frame) and labels into windowed landmarks
    """

    big_arrays: dict[str, torch.Tensor]
    video_categorical: dict[int, dict[str, int]]
    mouse_categorical: dict[tuple[int, int], dict[str, int]]
    track_lookup: dict[tuple[int, int], int]
    offset_arr: torch.Tensor
    length_arr: torch.Tensor
    fps_arr: torch.Tensor

    def __init__(
        self,
        meta: pd.DataFrame | list[dict],
        feats_config: FeaturesConfigDL,
        aug_config: AugmentationsConfig,
    ):
        if isinstance(meta, pd.DataFrame):
            meta = meta.to_dict(orient="records")

        self.feats_config = feats_config.model_copy(deep=True)
        self.aug_config = aug_config.model_copy(deep=True)
        self.device = torch.device("cuda")

        self.float_dtype = get_default_cuda_float_dtype()
        # print(f"[FeaturesLookupGPU] float_dtype is {self.float_dtype}\n")

        self.K = len(self.feats_config.bodyparts)

        arrays_lists = defaultdict(list)
        tracks = []
        offset = 0
        self.track_lookup = {}
        self.video_categorical = {}
        self.mouse_categorical = {}
        for video_meta in tqdm(meta, desc="[FeaturesLookupGPU] Init"):
            video_id = int(video_meta["video_id"])
            fps = float(video_meta["frames_per_second"])
            cnt_frames = int(video_meta["cnt_frames"])

            self.video_categorical[video_id] = get_categorical_map_for_video(video_meta)

            tracking = get_tracking_by_video_meta(video_meta=video_meta)
            mouse_ids = set(tracking.mouse_id.unique())
            for beh in parse_behaviors_labeled_from_row(row=video_meta):
                mouse_ids.add(beh.agent)
                mouse_ids.add(beh.target)
            for mouse_id in sorted(mouse_ids):
                coords = tracking_for_mouse_to_named_coords(
                    tracking=tracking,
                    mouse_id=mouse_id,
                    cnt_frames=cnt_frames,
                    bodyparts=self.feats_config.bodyparts,
                )
                x_mask = np.isnan(coords.x)
                y_mask = np.isnan(coords.y)
                x = interpolate_all_nans_per_column(coords.x).astype(
                    np.float32, copy=False
                )
                y = interpolate_all_nans_per_column(coords.y).astype(
                    np.float32, copy=False
                )
                arrays_lists["x"].append(x)
                arrays_lists["y"].append(y)
                arrays_lists["x_mask"].append(x_mask)
                arrays_lists["y_mask"].append(y_mask)

                self.track_lookup[(video_id, mouse_id)] = len(tracks)
                tracks.append(
                    TrackMeta(
                        video_id=video_id,
                        mouse_id=mouse_id,
                        offset=offset,
                        length=cnt_frames,
                        fps=fps,
                    )
                )
                offset += cnt_frames

                self.mouse_categorical[(video_id, mouse_id)] = (
                    get_categorical_map_for_mouse(mouse_id, video_meta)
                )

        self.big_arrays = {}
        total_bytes = 0
        for key, arr_list in arrays_lists.items():
            arr_np = np.concatenate(arr_list, axis=0)
            dtype = self.float_dtype if arr_np.dtype == np.float32 else torch.bool
            arr_t = torch.from_numpy(arr_np).to(self.device, dtype=dtype)
            self.big_arrays[key] = arr_t
            cnt_bytes = arr_t.nelement() * arr_t.element_size()
            print(
                f"[GPU Storage] {key:>6}: {cnt_bytes/2**30:.2f}GB,  dtype: {str(arr_t.dtype):15},  device: {str(arr_t.device)}"
            )
            total_bytes += cnt_bytes
        print(f"[GPU Storage]  TOTAL: {total_bytes/2**30:.2f}GB\n")

        for attr, dtype in [
            ("offset", torch.long),
            ("length", torch.long),
            ("fps", torch.float32),
        ]:
            arr = np.array([getattr(tr, attr) for tr in tracks])
            setattr(
                self, f"{attr}_arr", torch.from_numpy(arr).to(self.device, dtype=dtype)
            )

    def reset_configs(self, train_config: DL_TrainConfig):
        assert self.feats_config.bodyparts == train_config.features_config.bodyparts
        self.feats_config = train_config.features_config.model_copy(deep=True)
        self.aug_config = train_config.aug.model_copy(deep=True)

    def _gather_coords_batched(
        self,
        track_idx: torch.Tensor,
        frames: torch.Tensor,
        fps_scale: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        B = track_idx.size(0)
        T = self.feats_config.frames_in_window

        offsets = self.offset_arr[track_idx]  # [B]
        lengths = self.length_arr[track_idx]  # [B]
        fps = self.fps_arr[track_idx]  # [B]

        if fps_scale is not None:
            fps = fps * fps_scale

        center = frames.to(dtype=torch.float32)  # [B]
        half = 0.5 * self.feats_config.window_len_seconds * fps  # [B]

        start = center - half  # [B]
        end = center + half  # [B]

        grid = torch.linspace(
            0.0, 1.0, T, device=self.device, dtype=torch.float32
        )  # [T]
        t = start[:, None] + (end - start)[:, None] * grid[None, :]  # [B, T]

        t = t.clamp_min(0.0)
        max_idx = (lengths - 1).to(t.dtype).view(B, 1)  # [B, 1]
        t = torch.minimum(t, max_idx)

        t0 = torch.floor(t)
        t1 = t0 + 1
        t1 = torch.minimum(t1, max_idx)
        w = (t - t0).view(B * T, 1)

        t0_long = t0.view(-1).long()  # [B * T]
        t1_long = t1.view(-1).long()  # [B * T]

        base = offsets.view(B, 1).long()
        idx0 = (base + t0_long.view(B, T)).view(-1)
        idx1 = (base + t1_long.view(B, T)).view(-1)

        ret = {}
        for c in ["x", "y"]:
            for suff in ["", "_mask"]:
                key = f"{c}{suff}"
                big = self.big_arrays[key]
                v0 = big[idx0].to(torch.float32)
                v1 = big[idx1].to(torch.float32)
                v = v0 + (v1 - v0) * w
                ret[key] = v.view(B, T, self.K).to(self.float_dtype)
        return ret

    @torch.no_grad()
    def preprocess_batch(
        self, batch: dict[str, torch.Tensor], is_train: bool
    ) -> dict[str, torch.Tensor]:
        video_ids_np = batch["video_id"].detach().cpu().numpy().astype(int)

        frames = batch["frame"].to(self.device)
        B = frames.size(0)

        # --- Time stretch factors: per-sample in batch ---
        fps_scale = None
        if is_train and self.aug_config.time_stretch_config is not None:
            cfg = self.aug_config.time_stretch_config
            fps_scale = torch.empty(B, device=self.device).uniform_(cfg.L, cfg.R)
            mask_apply = torch.rand(B, device=self.device) < cfg.apply_prob
            fps_scale[~mask_apply] = 1.0

        # agent/target
        # x, y, x_mask, y_mask
        ret = {}
        ids_np = {}
        for who in ["agent", "target"]:
            ids_np[who] = batch[who].detach().cpu().numpy().astype(int)
            track_idx = np.empty(B, dtype=np.int64)
            for i in range(B):
                key = (int(video_ids_np[i]), int(ids_np[who][i]))
                idx = 0
                if key in self.track_lookup:
                    idx = self.track_lookup[key]
                track_idx[i] = idx
            track_idx_t = torch.from_numpy(track_idx).to(self.device, dtype=torch.long)
            coords = self._gather_coords_batched(
                track_idx_t, frames, fps_scale=fps_scale
            )
            for c in ["x", "y"]:
                for suff in ["", "_mask"]:
                    ret[f"{who}_{c}{suff}"] = coords[f"{c}{suff}"]

        video_cat = defaultdict(list)  # dict[str, int[B]]
        for i in range(B):
            video_id = int(video_ids_np[i])
            cat = self.video_categorical[video_id]
            for name, idx in cat.items():
                video_cat[name].append(idx)
        for name, idx_list in video_cat.items():
            ret[f"{name}_idx"] = torch.tensor(
                idx_list, dtype=torch.long, device=self.device
            )

        for who in ["agent", "target"]:
            mouse_cat = defaultdict(list)
            for i in range(B):
                video_id = int(video_ids_np[i])
                mouse_id = int(ids_np[who][i])
                cat = self.mouse_categorical[(video_id, mouse_id)]
                for name, idx in cat.items():
                    mouse_cat[name].append(idx)
            for name, idx_list in mouse_cat.items():
                ret[f"{who}_{name}_idx"] = torch.tensor(
                    idx_list, dtype=torch.long, device=self.device
                )

        batch.update(ret)
        return batch
