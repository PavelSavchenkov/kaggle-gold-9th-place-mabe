from collections import defaultdict

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from common.ensemble_building_primitives import (
    EnsembleObjective,
    build_ensembles_beam_search,
)
from common.helpers import logit_to_prob, prob_to_logit
from common.metrics_common import calc_best_f1_threshold


class EnsembleApproachDL(BaseModel):
    max_models: int = 3
    objective: EnsembleObjective = EnsembleObjective.nested_f1
    num_bins: int = 10
    pool_size: int = 1
    allow_neg: bool = False
    nested_cnt: bool = False
    logits: bool = False

    model_config = ConfigDict(frozen=True)

    def to_str(self) -> str:
        return f"models<={self.max_models}, objective={self.objective.value}, allow_neg={int(self.allow_neg)}"


class EnsembleDL(BaseModel):
    names: list[str] = Field(default_factory=list)
    weights: list[float] = Field(default_factory=list)
    th: float | None = None
    logits: bool = False

    def infer(self, names: list[str], preds: list[np.ndarray]) -> np.ndarray:
        assert self.names
        assert len(self.names) == len(self.weights)

        assert names
        assert len(names) == len(preds)

        res = np.zeros_like(preds[0])
        if self.logits:
            logits_preds = []
            for pred in preds:
                logits_preds.append(prob_to_logit(pred))
            preds = logits_preds
        for name, w in zip(self.names, self.weights):
            res += preds[names.index(name)] * w
        if self.logits:
            res = logit_to_prob(res)
        return res


def build_ensemble_map_per_approach(
    names: list[str],
    y_true: np.ndarray,
    y_pred_list: list[np.ndarray],
    fold_id: np.ndarray,
    approaches: list[EnsembleApproachDL],
) -> dict[EnsembleApproachDL, EnsembleDL]:
    max_models_per_app = defaultdict(int)
    for app in approaches:
        assert not app.nested_cnt
        app_base = app.model_copy(update={"max_models": 0})
        max_models_per_app[app_base] = max(max_models_per_app[app_base], app.max_models)

    res = {}
    for app, max_models in max_models_per_app.items():
        es = build_ensembles_beam_search(
            y_preds=y_pred_list,
            y_true=y_true,
            objective=app.objective,
            num_bins=app.num_bins,
            pool_size=app.pool_size,
            max_models=max_models,
            fold_id=fold_id,
            verbose=False,
            allow_neg_weights=app.allow_neg,
            use_logits=app.logits,
        )
        for m in range(
            1, max_models + 1
        ):  # can add new approaches not in the initial list
            cur_app = app.model_copy(update={"max_models": m})
            e = es[m]
            y_pred = e.infer(preds_list=y_pred_list)
            oof_th = calc_best_f1_threshold(y_true=y_true, y_pred=y_pred)
            e_dl = EnsembleDL(th=oof_th)
            for idx, w in zip(e.model_idx, e.weights):
                e_dl.names.append(names[idx])
                e_dl.weights.append(w)
            res[cur_app] = e_dl
    return res
