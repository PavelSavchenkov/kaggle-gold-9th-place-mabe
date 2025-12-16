import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from common.helpers import get_annotation_by_video_meta, get_train_meta
from common.parse_utils import parse_behaviors_labeled_from_row
from common.submission_common import PredictedProbsForBehavior
from dl.metrics import ProdMetricCompHelperOnline


class PerVideoActionTimelineVisualizer:
    def __init__(self, out_dir: str = "action_timelines"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def plot_group(
        self,
        video_id: int,
        agent: int,
        target: int,
        group_preds: list[PredictedProbsForBehavior],
        gt_per_pred: dict[tuple[int, int, int, str], np.ndarray],
    ):
        if not group_preds:
            return

        # sort actions for deterministic ordering
        group_preds = sorted(group_preds, key=lambda p: p.behavior.action)

        actions = [p.behavior.action for p in group_preds]
        thresholds = np.array([p.threshold for p in group_preds], dtype=np.float32)

        # stack GT and probs: shape (T, K)
        gt_mat = np.stack([gt_per_pred[p.key()] for p in group_preds], axis=1)  # (T, K)
        probs_mat = np.stack([p.probs for p in group_preds], axis=1)  # (T, K)

        T, K = probs_mat.shape
        if T == 0 or K == 0:
            return

        t = np.arange(T)

        # ----- colors: one per action -----
        base_cmap = plt.cm.get_cmap("tab20", K)
        colors = base_cmap(np.arange(K))  # RGBA
        action_to_color = {actions[i]: colors[i] for i in range(K)}

        # ----- figure & axes: 1 (GT) + K (probs) -----
        fig, axes = plt.subplots(K + 1, 1, sharex=True, figsize=(12, 2 * (K + 1)))
        if K + 1 == 1:
            axes = np.array([axes])

        # ----- GT strip (top row) -----
        ax_gt = axes[0]

        # encode GT as indices: 0 = none, 1..K = each action
        gt_idx = np.zeros(T, dtype=np.int32)
        for j in range(K):
            gt_idx[gt_mat[:, j] > 0.5] = j + 1  # at most one GT per frame by design

        # colormap: 0->white, 1..K -> action colors
        color_list = np.vstack([[1, 1, 1, 1], colors])  # prepend white
        cmap = ListedColormap(color_list)

        ax_gt.imshow(
            gt_idx[np.newaxis, :],
            aspect="auto",
            cmap=cmap,
            vmin=-0.5,
            vmax=K + 0.5,
        )
        ax_gt.set_yticks([])
        ax_gt.set_ylabel("GT")
        ax_gt.set_title(f"Video {video_id}, agent {agent}, target {target}")

        # add legend for actions
        for j, action in enumerate(actions):
            ax_gt.plot([], [], color=colors[j], label=action)
        ax_gt.legend(loc="upper right", fontsize="small", ncol=2)

        # ----- per-action probability rows -----
        for j, action in enumerate(actions):
            ax = axes[j + 1]
            c = colors[j]
            ax.plot(t, probs_mat[:, j], color=c)
            ax.axhline(thresholds[j], color=c, linestyle="--", alpha=0.7)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel(action)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Frame")

        fig.tight_layout()

        fname = f"video_{video_id}_agent_{agent}_target_{target}.png"
        safe = re.sub(r"[^0-9a-zA-Z]+", "_", fname)
        fig.savefig(self.out_dir / safe, dpi=150)
        plt.close(fig)


class ActionPairProbsVisualizer:
    def __init__(self):
        # key: (action1, action2) with action1 < action2
        self.data = defaultdict(
            lambda: {
                "p1": [],  # prob(action1)
                "p2": [],  # prob(action2)
                "first_correct": [],  # True if action1 is GT, else action2
                "is_correct_selection": [],  # True if model selected GT, False otherwise
            }
        )

    def add_batch(self, probs, gt, actions):
        """
        probs:   (n_frames, n_actions)
        gt:      same shape, binary, at most one 1 per row
        actions: list[str], len == n_actions
        """
        probs = np.asarray(probs)
        gt = np.asarray(gt)
        assert probs.shape == gt.shape
        assert probs.shape[1] == len(actions)

        n_frames, n_actions = probs.shape
        if n_frames == 0 or n_actions < 2:
            return

        idx_of = {a: i for i, a in enumerate(actions)}

        gt_sum = gt.sum(axis=1)
        has_gt = gt_sum > 0
        if not np.any(has_gt):
            return

        correct_idx = gt.argmax(axis=1)
        selected_idx = probs.argmax(axis=1)
        max_prob = probs.max(axis=1)
        selected_valid = np.isfinite(max_prob)

        # --- 1) Wrong selections (GT != selected) ---
        mask_wrong = has_gt & selected_valid & (correct_idx != selected_idx)
        for t in np.where(mask_wrong)[0]:
            ci, si = int(correct_idx[t]), int(selected_idx[t])
            correct_name = actions[ci]
            selected_name = actions[si]
            if correct_name == selected_name:
                continue

            a1, a2 = sorted((correct_name, selected_name))
            p1 = probs[t, idx_of[a1]]
            p2 = probs[t, idx_of[a2]]
            if not (np.isfinite(p1) and np.isfinite(p2)):
                continue  # skip -inf etc.

            d = self.data[(a1, a2)]
            d["p1"].append(float(p1))
            d["p2"].append(float(p2))
            d["first_correct"].append(a1 == correct_name)
            d["is_correct_selection"].append(False)

        # --- 2) Correct selections (GT == selected) vs top-2 competitor ---
        mask_correct = has_gt & selected_valid & (correct_idx == selected_idx)
        for t in np.where(mask_correct)[0]:
            ci = int(correct_idx[t])
            correct_name = actions[ci]

            row = probs[t].copy()
            row[ci] = -np.inf  # remove correct action, find best competitor
            if not np.isfinite(row).any():
                continue
            comp_idx = int(row.argmax())
            comp_name = actions[comp_idx]

            a1, a2 = sorted((correct_name, comp_name))
            p1 = probs[t, idx_of[a1]]
            p2 = probs[t, idx_of[a2]]
            if not (np.isfinite(p1) and np.isfinite(p2)):
                continue

            d = self.data[(a1, a2)]
            d["p1"].append(float(p1))
            d["p2"].append(float(p2))
            d["first_correct"].append(a1 == correct_name)
            d["is_correct_selection"].append(True)

    def finalize(self, out_dir, plot_correct=False):
        """
        out_dir: directory path
        plot_correct: if True, plot correct and wrong selections;
                      if False, plot only wrong selections.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # JSON dump
        json_data = {}
        for (a1, a2), d in self.data.items():
            key = f"{a1}__{a2}"
            json_data[key] = {
                "action1": a1,
                "action2": a2,
                "p1": d["p1"],
                "p2": d["p2"],
                "first_correct": d["first_correct"],
                "is_correct_selection": d["is_correct_selection"],
            }

        with (out_dir / "action_pairs.json").open("w") as f:
            json.dump(json_data, f)

        # Plots
        for (a1, a2), d in self.data.items():
            if not d["p1"]:
                continue

            p1 = np.array(d["p1"])
            p2 = np.array(d["p2"])
            first_correct = np.array(d["first_correct"], bool)
            is_corr = np.array(d["is_correct_selection"], bool)

            # choose which points to show
            if plot_correct:
                mask = np.ones_like(is_corr, dtype=bool)
            else:
                mask = ~is_corr  # only wrong selections

            if not np.any(mask):
                continue

            m_red = mask & first_correct  # first action correct -> red
            m_blue = mask & ~first_correct  # second action correct -> blue

            plt.figure()
            if np.any(m_red):
                plt.scatter(p1[m_red], p2[m_red], s=5, c="red", label=f"{a1} correct")
            if np.any(m_blue):
                plt.scatter(
                    p1[m_blue], p2[m_blue], s=5, c="blue", label=f"{a2} correct"
                )

            plt.xlabel(a1)
            plt.ylabel(a2)
            plt.title(f"{a1} vs {a2}")
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.legend()
            plt.grid(True, alpha=0.3)

            safe = re.sub(r"[^0-9a-zA-Z]+", "_", f"{a1}__{a2}")
            plt.savefig(out_dir / f"{safe}.png", bbox_inches="tight")
            plt.close()


def analyse_segments_dl(predictions: list[PredictedProbsForBehavior]):
    meta = get_train_meta()
    gt_per_pred = {}
    for video_meta in tqdm(
        meta.to_dict(orient="records"), desc="Building gt from meta.csv..."
    ):
        video_id = int(video_meta["video_id"])
        cnt_frames = int(video_meta["cnt_frames"])
        behs = parse_behaviors_labeled_from_row(row=video_meta)
        for beh in behs:
            key = (video_id, beh.agent, beh.target, beh.action)
            if key not in gt_per_pred:
                gt_per_pred[key] = np.zeros(cnt_frames, dtype=np.int8)
        annot = get_annotation_by_video_meta(video_meta=video_meta)
        for annot_row in annot.to_dict(orient="records"):
            key = (
                video_id,
                int(annot_row["agent_id"]),
                int(annot_row["target_id"]),
                annot_row["action"],
            )
            assert key in gt_per_pred
            l = int(annot_row["start_frame"])
            r = int(annot_row["stop_frame"])
            gt_per_pred[key][l:r] = 1

    prediction_groups = defaultdict(list)
    for pred in predictions:
        prediction_groups[
            (pred.video_id, pred.behavior.agent, pred.behavior.target)
        ].append(pred)

    f1_helper = ProdMetricCompHelperOnline()
    for pred in tqdm(predictions, desc="Recomputing f1 over all predictions..."):
        key = pred.key()
        assert key in gt_per_pred
        f1_helper.update(
            pred=pred.probs >= pred.threshold,
            gt=gt_per_pred[key],
            lab=pred.lab_name,
            action=pred.behavior.action,
        )
    print(f"Recomputed f1 (actions are independent): {f1_helper.finalise():.5f}")

    # actions_pair_visualizer = ActionPairProbsVisualizer()
    for group in tqdm(prediction_groups.values(), desc="Analysing action pairs..."):
        group.sort(key=lambda pred: pred.behavior.action)
        assert group
        gt_list = []
        probs_list = []
        for pred in group:
            gt_list.append(gt_per_pred[pred.key()][:, None])
            probs = np.where(pred.probs < pred.threshold, -np.inf, pred.probs)
            probs_list.append(probs[:, None])
        actions = [pred.behavior.action for pred in group]

        gt = np.concatenate(gt_list, axis=1)
        probs = np.concatenate(probs_list, axis=1)
        assert gt.shape == probs.shape

    #     actions_pair_visualizer.add_batch(probs=probs, gt=gt, actions=actions)
    # actions_pair_visualizer.finalize("action_pairs", plot_correct=True)

    # ----- per-video / agent / target timelines -----
    # timeline_viz = PerVideoActionTimelineVisualizer("action_timelines")
    # for (video_id, agent, target), group in tqdm(
    #     prediction_groups.items(),
    #     desc="Plotting per-video timelines...",
    #     total=len(prediction_groups),
    # ):
    #     timeline_viz.plot_group(video_id, agent, target, group, gt_per_pred)
