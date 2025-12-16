from collections import defaultdict
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from common.constants import LAB_NAMES_IN_TEST
from common.helpers import get_annotation_by_video_meta, get_train_meta
from common.parse_utils import parse_behaviors_labeled


def get_labs_in_test_with_action(action: str) -> list[str]:
    train_meta = get_train_meta()
    train_meta = train_meta[train_meta.has_annotation]
    labs = set()
    for row in train_meta.to_dict(orient="records"):
        annot = get_annotation_by_video_meta(video_meta=row)
        if not annot[annot.action == action].empty:
            labs.add(row["lab_id"])
        # behs = parse_behaviors_labeled(row["behaviors_labeled"])
        # if any(b.action for b in behs if b.action == action):
        #     labs.add(row["lab_id"])
    return list(sorted(labs.intersection(LAB_NAMES_IN_TEST)))


class DurationStats:
    duration_by_lab_id: dict[
        str, dict[str, float]
    ]  # duration[lab_id] = { action_0: duration_0, ..., "passive": duration_passive } for given actions

    def __init__(
        self, actions: list[str], video_ids: list[int] | set[int] | None = None
    ):
        train_meta = get_train_meta()
        train_meta = train_meta[train_meta.has_annotation]
        self.duration_by_lab_id = {}

        if video_ids is not None:
            video_ids = set(video_ids)

        for row in train_meta.to_dict(orient="records"):
            if video_ids is not None and row["video_id"] not in video_ids:
                continue
            lab_id = row["lab_id"]
            fps = row["frames_per_second"]
            cnt_frames = row["cnt_frames"]
            annot = get_annotation_by_video_meta(row)
            active_frames_by_pair = defaultdict(set)
            active_frames_by_triple = defaultdict(set)
            beh = parse_behaviors_labeled(row["behaviors_labeled"])
            pairs_to_consider = set(
                (b.agent, b.target) for b in beh if b.action in actions
            )
            if len(pairs_to_consider) == 0:
                continue
            for b in beh:
                if b.action in actions:
                    p = (b.agent, b.target)
                    active_frames_by_pair[p] = set()
                    active_frames_by_triple[(p, b.action)] = set()
            for annot_row in annot.to_dict(orient="records"):
                pair = (annot_row["agent_id"], annot_row["target_id"])
                assert isinstance(pair[0], int)
                assert isinstance(pair[1], int)
                if pair not in pairs_to_consider:
                    continue
                action = annot_row["action"]
                if action not in actions:
                    continue
                triple = (pair, action)
                s = set(range(annot_row["start_frame"], annot_row["stop_frame"]))
                assert len(active_frames_by_pair[pair].intersection(s)) == 0
                assert len(active_frames_by_triple[triple].intersection(s)) == 0
                active_frames_by_pair[pair] |= s
                active_frames_by_triple[triple] |= s
            if lab_id not in self.duration_by_lab_id:
                self.duration_by_lab_id[lab_id] = defaultdict(float)
            for pair, s in active_frames_by_pair.items():
                cnt_passive_frames = cnt_frames - len(s)
                assert cnt_passive_frames >= 0
                self.duration_by_lab_id[lab_id]["passive"] += cnt_passive_frames / fps
            for (pair, action), s in active_frames_by_triple.items():
                self.duration_by_lab_id[lab_id][action] += len(s) / fps


class DownsampleConfig(BaseModel):
    drop_rate: dict = Field(
        default_factory=lambda: {}
    )  # lab_id -> dict; dict: action | "passive" -> float;
    seed: int = 0


class TestDownsampleParams(BaseModel):
    duration_cap: float = 1e6 / 30
    seed: int = 0

    def build_downsample_config(
        self, actions: list[str], video_ids: list[int]
    ) -> DownsampleConfig:
        """
        * maximise min over non-passive actions across all labs
        * keep original class proportions within each lab
        """
        stats = DurationStats(actions=actions, video_ids=video_ids)

        def duration_need_for(min_duration: float) -> tuple[float, dict[str, float]]:
            """
            returns:
                (duration spent, drop_rate per lab)
            """
            duration_spent = 0.0
            drop_rate = {}
            for lab_id, actions_duration in stats.duration_by_lab_id.items():
                lower_bound = 0.0
                for _, dur in actions_duration.items():
                    lower_bound = max(lower_bound, min_duration / dur)
                lower_bound = min(lower_bound, 1.0)
                drop_rate[lab_id] = 1 - lower_bound
                for _, dur in actions_duration.items():
                    duration_spent += dur * lower_bound
            return (duration_spent, drop_rate)

        L = 0.0
        R = self.duration_cap + 1
        while L + 0.5 < R:
            M = (L + R) / 2
            if duration_need_for(M)[0] < self.duration_cap:
                L = M
            else:
                R = M
        _, drop_rate = duration_need_for(R)

        final_drop_rate = {}
        for lab_id, lab_dict in stats.duration_by_lab_id.items():
            final_drop_rate[lab_id] = {}
            for action in lab_dict.keys():
                final_drop_rate[lab_id][action] = drop_rate[lab_id]

        return DownsampleConfig(drop_rate=final_drop_rate, seed=self.seed)


class DownsampleParams(BaseModel):
    """
    Controls downsampling across labs with
    - labs_baseline: baseline lab proportions before multipliers
    - labs_coefs_mult: multiplicative adjustment per lab
    - passive_target_percentage: target passive share per lab [0..1], or -1 to keep original
    - total_duration_cap: total seconds to keep across selected videos
    """

    labs_baseline: Literal["original", "uniform"] = Field(default="original")
    labs_coefs_mult: dict[str, float] = Field(default_factory=dict)
    passive_percentage: float = Field(default=-1.0)
    total_duration_cap: float = Field(default=1e6 / 30)
    seed: int = 0

    def build_downsample_config(
        self, actions: list[str], video_ids: list[int]
    ) -> DownsampleConfig:
        stats = DurationStats(actions=actions, video_ids=video_ids)

        # 1) Compute baseline weights per lab
        lab_ids = sorted(stats.duration_by_lab_id.keys())
        assert lab_ids

        baseline_weights: dict[str, float] = {}
        for lab_id in lab_ids:
            durations = stats.duration_by_lab_id[lab_id]
            total_lab = sum(float(v) for v in durations.values())
            if self.labs_baseline == "uniform":
                baseline = 1.0 if total_lab > 0 else 0.0
            else:
                assert self.labs_baseline == "original"
                baseline = total_lab
            baseline_weights[lab_id] = max(0.0, float(baseline))

        # 2) Apply multipliers and normalise
        weighted: dict[str, float] = {}
        for lab_id in lab_ids:
            mult = float(self.labs_coefs_mult.get(lab_id, 1.0))
            assert mult >= 0.0
            weighted[lab_id] = baseline_weights[lab_id] * mult

        sum_w = sum(weighted.values())
        assert sum_w > 0.0

        target_capacity_by_lab = {
            lab_id: (self.total_duration_cap * (weighted[lab_id] / sum_w))
            for lab_id in lab_ids
        }

        # Helpers to fetch durations safely
        def get_passive_and_active(
            lab_durations: dict[str, float],
        ) -> tuple[float, float]:
            passive = lab_durations["passive"]
            active_total = float(sum(lab_durations.values())) - passive
            return passive, active_total

        passive_target = float(self.passive_percentage)

        final_drop_rate: dict[str, dict[str, float]] = {}
        for lab_id in lab_ids:
            lab_actions = stats.duration_by_lab_id[lab_id]
            cap = float(target_capacity_by_lab[lab_id])

            P_avail, A_avail = get_passive_and_active(lab_actions)
            total_avail = P_avail + A_avail

            if total_avail <= 0.0 or cap <= 0.0:
                final_drop_rate[lab_id] = {"passive": 0.0}
                for a in actions:
                    final_drop_rate[lab_id][a] = 0.0
                continue

            assert cap > 0.0

            if passive_target < 0.0:
                # Keep original passive ratio within lab
                drop_rate = max(0.0, 1.0 - cap / total_avail)
                final_drop_rate[lab_id] = {}
                for a in actions:
                    final_drop_rate[lab_id][a] = drop_rate
                final_drop_rate[lab_id]["passive"] = drop_rate
                continue

            # Fixed passive percentage p_target
            # Ideal targets under the capacity cap
            P_target = passive_target * cap
            A_target = (1.0 - passive_target) * cap

            if P_target > P_avail:
                A_target *= P_avail / P_target
                P_target = P_avail
            if A_target > A_avail:
                P_target *= A_avail / A_target
                A_target = A_avail

            # Keep rates with clamping; do not rebalance remainder to preserve the ratio priority
            keep_passive = (P_target / P_avail) if P_avail > 0 else 0.0
            keep_active = (A_target / A_avail) if A_avail > 0 else 0.0
            assert keep_passive <= 1.0
            assert keep_active <= 1.0

            drop_passive = 1.0 - keep_passive
            drop_active = 1.0 - keep_active

            final_drop_rate[lab_id] = {}
            for a in actions:
                final_drop_rate[lab_id][a] = drop_active
            final_drop_rate[lab_id]["passive"] = drop_passive

        return DownsampleConfig(drop_rate=final_drop_rate, seed=self.seed)


class SampleCoefsParams(BaseModel):
    labs_baseline: Literal["original", "uniform"] = Field(default="original")
    labs_coefs_mult: dict[str, float] = Field(default_factory=dict)
    passive_percentage: float = Field(default=-1.0)

    def calc_sample_coefs(self, actions: list[str], feats: dict) -> np.ndarray:
        assert "y" in feats
        assert "index" in feats

        y: np.ndarray = feats["y"].data
        index_df = feats["index"]

        assert y.ndim == 2
        assert y.shape[1] == len(actions)

        labs = index_df["lab_id"].to_numpy()
        unique_labs = np.unique(labs)
        assert unique_labs.size > 0

        n = y.shape[0]
        coefs = np.ones(n, dtype=np.float32)

        # Precompute masks for passive/active per row
        passive_rows_mask = np.all((y == 0) | (y == -1), axis=1)
        active_rows_mask = np.any(y == 1, axis=1)

        # 1) Within-lab passive/active ratio adjustment
        p_target = float(self.passive_percentage)
        for lab_id in unique_labs:
            lab_mask = labs == lab_id
            assert np.any(lab_mask)

            lab_passive_mask = lab_mask & passive_rows_mask
            lab_active_mask = lab_mask & active_rows_mask

            cP = int(np.count_nonzero(lab_passive_mask))
            cA = int(np.count_nonzero(lab_active_mask))

            assert cP > 0
            assert cA > 0

            if p_target < 0.0:
                # Keep original within-lab ratio: uniform weights inside lab
                sP = 1.0
                sA = 1.0
            else:
                # Target passive share per lab
                p = min(max(p_target, 0.0), 1.0)
                # Choose weights so that weighted share of passive equals p
                # Any common scaling cancels later, so set sP and sA proportional to p/cP and (1-p)/cA
                sP = p / cP
                sA = (1.0 - p) / cA

            coefs[lab_passive_mask] *= sP
            coefs[lab_active_mask] *= sA

        # 2) Lab-level weighting according to baseline + multipliers

        # Compute target shares per lab
        baseline_num: dict[str, float] = {}
        for lab_id in unique_labs:
            lab_mask = labs == lab_id
            cnt = int(np.count_nonzero(lab_mask))
            assert cnt > 0
            if self.labs_baseline == "uniform":
                base = 1.0
            else:
                assert self.labs_baseline == "original"
                base = float(cnt)
            mult = float(self.labs_coefs_mult.get(str(lab_id), 1.0))
            assert mult >= 0.0
            baseline_num[str(lab_id)] = base * mult

        total_num = float(sum(baseline_num.values()))
        assert total_num > 0.0, "Sum of lab weights must be positive"

        target_share: dict[str, float] = {
            lab_id: (baseline_num[str(lab_id)] / total_num) for lab_id in unique_labs
        }

        # Current weighted share per lab (after step 1)
        sum_w_by_lab: dict[str, float] = {}
        total_w = 0.0
        for lab_id in unique_labs:
            lab_mask = labs == lab_id
            s = float(coefs[lab_mask].sum())
            sum_w_by_lab[str(lab_id)] = s
            total_w += s

        # We scale by p_lab / q_lab where q_lab = current share (s / total_w)
        for lab_id in unique_labs:
            s = sum_w_by_lab[str(lab_id)]
            assert s > 0.0
            assert total_w > 0.0
            q_lab = s / total_w
            p_lab = target_share[lab_id]
            scale = p_lab / q_lab
            coefs[labs == lab_id] *= float(scale)

        # 3) Normalise for convenience: make average weight = 1
        s_total = float(coefs.sum())
        if s_total > 0:
            coefs *= n / s_total

        return coefs
