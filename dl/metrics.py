from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score

from common.constants import (
    ACTION_NAMES_IN_TEST,
    ALL_LAB_NAMES,
    ALL_SELF_ACTIONS,
    VIDEO_CATEGORICAL_FEATURES,
)

LAB_FEATURE_NAME = "lab_id_idx"


def _compute_raw_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.int32).ravel()
    y_prob = y_prob.astype(np.float32).ravel()

    if y_true.size == 0:
        return {}

    if np.all(y_true == 0) or np.all(y_true == 1):
        pr_auc = 0.0
    else:
        pr_auc = float(average_precision_score(y_true, y_prob))

    eps = 1e-7
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    log_loss = float(
        -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1.0 - y_prob)).mean()
    )

    return {"pr-auc": pr_auc, "log-loss": log_loss}


@dataclass
class EvalData:
    metrics: dict[str, float]
    preds_map: dict[
        tuple[str, str], np.ndarray
    ]  # (action, lab) -> predicted probs, float32 or float16
    gt_map: dict[tuple[str, str], np.ndarray]  # (action, lab) -> gt, bool


class MetricsComputer:
    @dataclass
    class _Group:
        lab_id: int
        lab_name: str
        action_idx: int
        action_name: str
        sample_indices: np.ndarray  # 1D indices into rows [0..N)
        y_true: np.ndarray  # 1D, int (0/1)

    def __init__(
        self,
        labels: np.ndarray,
        labels_known: np.ndarray,
        lab_ids: np.ndarray,
    ):
        """
        labels       : [N, C] (bool)
        labels_known : [N, C] (bool)  (True for known, False for unknown)
        lab_ids      : [N] (int)
        """
        N, C = labels.shape
        assert labels_known.dtype == bool
        assert labels_known.shape == (N, C)

        self._N = N
        self._C = C
        self._labels_known = labels_known

        self._lab_name_lookup = VIDEO_CATEGORICAL_FEATURES["lab_id"]
        self._unique_labs = np.unique(lab_ids)

        lab_to_indices: dict[int, np.ndarray] = {}
        for lab in self._unique_labs:
            lab_mask = lab_ids == lab
            lab_to_indices[int(lab)] = np.nonzero(lab_mask)[0]

        self._groups: list[MetricsComputer._Group] = []

        # Precompute (lab, action) groups
        for action_idx, action_name in enumerate(ACTION_NAMES_IN_TEST):
            known_mask_for_action = labels_known[:, action_idx]
            if not known_mask_for_action.any():
                continue

            for lab in self._unique_labs:
                lab_indices = lab_to_indices[lab]
                assert lab_indices.size > 0

                known_in_lab = known_mask_for_action[lab_indices]
                if not known_in_lab.any():
                    continue

                sample_indices = lab_indices[known_in_lab]
                y_true = labels[sample_indices, action_idx].ravel()

                if (y_true == 1).all() or (y_true == 0).all():
                    continue

                self._groups.append(
                    MetricsComputer._Group(
                        lab_id=lab,
                        lab_name=self._lab_name_lookup[lab],
                        action_idx=action_idx,
                        action_name=action_name,
                        sample_indices=sample_indices,
                        y_true=y_true,
                    )
                )

    def compute_eval_data(self, probs: np.ndarray) -> EvalData:
        """
        probs: [N, C]

        Metrics: per-lab, per-action, per-lab-action, prod/*
        Preds
        GT
        """
        assert probs.shape == (self._N, self._C)
        assert -1e-6 <= probs.min() and probs.max() <= 1 + 1e-6

        metrics: dict[str, float] = {}
        preds_map: dict[tuple[str, str], np.ndarray] = {}
        gt_map: dict[tuple[str, str], np.ndarray] = {}

        # per_lab_action[metric_name][lab_id][action_name] = value
        per_lab_action: dict[str, dict[int, dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        metric_names: set[str] = set()

        # ---------- per-(lab, action) metrics ----------
        for g in self._groups:
            y_prob = probs[g.sample_indices, g.action_idx]
            y_true = g.y_true

            m = _compute_raw_metrics(y_true, y_prob)

            key = (g.action_name, g.lab_name)
            preds_map[key] = y_prob
            gt_map[key] = y_true

            key_suffix = f"{g.action_name}+{g.lab_name}"

            for metric_name, val in m.items():
                metric_names.add(metric_name)
                metrics[f"each-{metric_name}/{key_suffix}"] = val
                per_lab_action[metric_name][g.lab_id][g.action_name] = val

        # ---------- per-lab metrics (avg over actions) ----------
        lab_means: dict[str, list[float]] = {mn: [] for mn in metric_names}

        for metric_name in metric_names:
            per_lab_for_metric = per_lab_action[metric_name]

            for lab in self._unique_labs:
                action_dict = per_lab_for_metric[lab]

                vals = list(action_dict.values())
                mean_val = float(np.mean(vals))

                lab_name = self._lab_name_lookup[lab]
                metrics[f"per-lab-{metric_name}/{lab_name}"] = mean_val
                lab_means[metric_name].append(mean_val)

        # ---------- per-action metrics (avg over labs) ----------
        for metric_name in metric_names:
            per_lab_for_metric = per_lab_action[metric_name]

            per_action_vals: dict[str, list[float]] = defaultdict(list)
            for lab, action_dict in per_lab_for_metric.items():
                for action_name, val in action_dict.items():
                    per_action_vals[action_name].append(val)

            for action_name in ACTION_NAMES_IN_TEST:
                vals = per_action_vals.get(action_name)
                if not vals:
                    continue
                metrics[f"per-action-{metric_name}/{action_name}"] = float(
                    np.mean(vals)
                )

        # ---------- prod metrics (avg over labs) ----------
        for metric_name, vals in lab_means.items():
            metrics[f"prod/{metric_name}"] = float(np.mean(vals))

        MetricsComputer.add_self_pair(metrics=metrics)

        # ---------- prod-argmax metrics (one action per frame) ----------
        probs_argmax = self._apply_argmax_mask(probs)
        prod_argmax_metrics = self._compute_prod_metrics_from_probs(probs_argmax)
        for metric_name, val in prod_argmax_metrics.items():
            metrics[f"prod/argmax-{metric_name}"] = val
        # -----------------------------------------------------------------

        return EvalData(metrics=metrics, gt_map=gt_map, preds_map=preds_map)

    # keep only the argmax (over known actions) per frame
    def _apply_argmax_mask(self, probs: np.ndarray) -> np.ndarray:
        """
        Returns a copy of probs where, for each frame i, among actions with
        labels_known[i, j] == True we keep only the argmax prob and set the rest
        of those known actions to 0. Unknown actions are left untouched (they
        are not used in metric computation anyway).
        """
        assert probs.shape == (self._N, self._C)

        known = self._labels_known
        probs_argmax = probs.copy()

        # Compute argmax over known actions per frame
        masked = probs.copy()
        masked[~known] = -np.inf  # exclude unknowns from argmax
        argmax_idx = np.argmax(masked, axis=1)  # shape [N]

        # Build mask of entries to zero: known & not the argmax
        N, C = probs.shape
        argmax_mask = np.zeros_like(known, dtype=bool)
        argmax_mask[np.arange(N), argmax_idx] = True

        to_zero = known & ~argmax_mask
        probs_argmax[to_zero] = 0.0

        return probs_argmax

    # recompute prod/* metrics using arbitrary probs (e.g., argmaxed)
    def _compute_prod_metrics_from_probs(self, probs: np.ndarray) -> dict[str, float]:
        """
        Compute prod/{metric} in the same way as in compute_eval_data, but
        only return the prod-level metrics (no each-/per-lab/per-action entries).
        """
        # per_lab_action[metric_name][lab_id][action_name] = value
        per_lab_action: dict[str, dict[int, dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        metric_names: set[str] = set()

        # per-(lab, action) metrics for these probs
        for g in self._groups:
            y_prob = probs[g.sample_indices, g.action_idx]
            y_true = g.y_true

            m = _compute_raw_metrics(y_true, y_prob)
            for metric_name, val in m.items():
                metric_names.add(metric_name)
                per_lab_action[metric_name][g.lab_id][g.action_name] = val

        # per-lab metrics (avg over actions) -> lab_means
        lab_means: dict[str, list[float]] = {mn: [] for mn in metric_names}
        for metric_name in metric_names:
            per_lab_for_metric = per_lab_action[metric_name]
            for lab in self._unique_labs:
                action_dict = per_lab_for_metric[lab]
                vals = list(action_dict.values())
                mean_val = float(np.mean(vals))
                lab_means[metric_name].append(mean_val)

        # prod metrics (avg over labs)
        prod_metrics: dict[str, float] = {}
        for metric_name, vals in lab_means.items():
            prod_metrics[metric_name] = float(np.mean(vals))

        return prod_metrics

    @staticmethod
    def add_self_pair(metrics: dict[str, float]):
        vals_per_type_per_metric_name_per_lab = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for k, v in metrics.items():
            pref = "each-"
            if not k.startswith(pref):
                continue
            k = k[len(pref) :]
            metric_name, key_suffix = k.split("/")
            action_name, lab_name = key_suffix.split("+")
            if action_name in ALL_SELF_ACTIONS:
                vals_per_type_per_metric_name_per_lab["self"][metric_name][
                    lab_name
                ].append(v)
            else:
                vals_per_type_per_metric_name_per_lab["pair"][metric_name][
                    lab_name
                ].append(v)

        for t in ["self", "pair"]:
            for metric_name, per_lab in vals_per_type_per_metric_name_per_lab[
                t
            ].items():
                means = []
                for _, lst in vals_per_type_per_metric_name_per_lab[t][
                    metric_name
                ].items():
                    means.append(float(np.mean(lst)))
                metric = float(np.mean(means))
                metrics[f"prod/{t}-{metric_name}"] = metric


class ProdMetricCompHelper:
    def __init__(self):
        self.per_lab = defaultdict(list)

    def add(self, lab: str, val: float):
        self.per_lab[lab].append(val)

    def calc(self) -> float:
        val = []
        for lst in self.per_lab.values():
            val.append(float(np.mean(lst)))
        return float(np.mean(val))


class ProdMetricCompHelperOnline:
    def __init__(self):
        # stats[lab][action] = {"tp": int, "fp": int, "fn": int}
        self.stats = defaultdict(
            lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        )

    def update(self, pred, gt, lab: str, action: str):
        pred = np.asarray(pred).astype(bool)
        gt = np.asarray(gt).astype(bool)

        if pred.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch: pred.shape={pred.shape}, gt.shape={gt.shape}"
            )

        tp = np.logical_and(pred, gt).sum()
        fp = np.logical_and(pred, ~gt).sum()
        fn = np.logical_and(~pred, gt).sum()

        s = self.stats[lab][action]
        s["tp"] += int(tp)
        s["fp"] += int(fp)
        s["fn"] += int(fn)

    def finalise(self) -> float:
        lab_f1s = []

        for lab, actions in self.stats.items():
            action_f1s = []
            for action, s in actions.items():
                tp, fp, fn = s["tp"], s["fp"], s["fn"]
                denom = 2 * tp + fp + fn
                if denom == 0:
                    f1 = 0.0  # no positives in gt or pred
                else:
                    f1 = 2.0 * tp / denom
                action_f1s.append(f1)

            if action_f1s:
                lab_f1 = sum(action_f1s) / len(action_f1s)
                lab_f1s.append(lab_f1)

        if not lab_f1s:
            return 0.0

        return sum(lab_f1s) / len(lab_f1s)

    def reset(self):
        self.stats.clear()
