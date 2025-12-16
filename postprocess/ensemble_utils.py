import gc
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import joblib
import numpy as np
import torch  # type: ignore
from pydantic import BaseModel, ConfigDict
from sklearn import ensemble
from sklearn.metrics import f1_score
from tqdm import tqdm

from common.ensemble_building_primitives import (
    Ensemble,
    EnsembleObjective,
    build_ensembles_beam_search,
    build_ensembles_logreg,
)
from common.helpers import ensure_1d_numpy
from common.metrics_common import calc_best_f1, calc_best_f1_threshold, calc_nested_f1
from gbdt.helpers import (
    GBDT_Model_Type,
    get_pred_test_np,
    get_test_ground_truth_np,
    get_test_index_df,
    model_type_by_name,
)
from postprocess.postprocess_utils import OOF_Metrics  # type: ignore
from postprocess.postprocess_utils import OOF_Metrics_for_model


class EnsembleApproach(BaseModel):
    max_models: int = 3
    filter_same_model_type: bool = False
    objective: EnsembleObjective | None = None
    num_weight_bins: int | None = None
    pool_size: int | None = None
    refit_with_fold_avg: bool = False
    avg_k_best: bool = False  # select k <= max_models which maximizes nested-f1
    logreg: bool = False

    model_config = ConfigDict(frozen=True)

    def to_str(self) -> str:
        pref = f"<={self.max_models}, "
        if self.logreg:
            pref += "logreg"
            return pref
        pref += f"filter={int(self.filter_same_model_type)}, "
        if self.avg_k_best:
            pref += "avg_k_best"
            return pref
        assert self.objective is not None
        pref += f"->{self.objective.value}, bins={self.num_weight_bins}, pool={self.pool_size}"
        if self.refit_with_fold_avg:
            pref += f", refit_avg"
        return pref


def build_ensemble(
    y_pred_list: list[np.ndarray],
    y_true: np.ndarray,
    fold_id: np.ndarray,
    model_types: list[GBDT_Model_Type],
    e_approach: EnsembleApproach,
) -> tuple[OOF_Metrics, list[int], list[float]]:
    assert len(model_types) == len(y_pred_list)

    folds = list(sorted(set(np.unique(fold_id))))
    model_types_str = [t.value for t in model_types]

    def return_given_e(e: Ensemble):
        y_pred_ensemble = e.infer(y_pred_list)

        oof_metrics = OOF_Metrics.build(
            y_pred=y_pred_ensemble, y_true=y_true, fold_id=fold_id, folds=folds
        )

        gc.collect()
        torch.cuda.empty_cache()

        return oof_metrics, e.model_idx, e.weights

    @dataclass
    class Item:
        pred: np.ndarray
        model_type: str
        idx: int

    all_items = []
    for i, (pred, model_type) in enumerate(zip(y_pred_list, model_types_str)):
        all_items.append(Item(pred=pred, model_type=model_type, idx=i))

    if e_approach.avg_k_best:
        if e_approach.filter_same_model_type:
            items = []
            seen_type = set()
            for item in all_items:
                if item.model_type not in seen_type:
                    seen_type.add(item.model_type)
                    items.append(item)
        else:
            items = all_items

        best_e = None
        for k in range(1, min(e_approach.max_models, len(items)) + 1):
            e = Ensemble(
                model_idx=[it.idx for it in items[:k]], weights=[1 / k] * k, score=-1.0
            )
            y_pred_e = e.infer(y_pred_list)
            nested_f1 = calc_nested_f1(y_pred=y_pred_e, y_true=y_true, fold_id=fold_id)
            e.score = float(np.mean(nested_f1))
            # print(f"k={k}, score={e.score:.5f}")
            if best_e is None or e.score > best_e.score:
                best_e = e

        assert best_e is not None
        # print(f"best k is {len(best_e.model_idx)}")
        return return_given_e(e=best_e)

    # always returns global indexes
    def ensemble_build_helper(items, y_true, fold_id):
        y_preds = []
        types = []
        idx = []
        for it in items:
            y_preds.append(it.pred)
            types.append(it.model_type)
            idx.append(it.idx)
        if e_approach.logreg:
            assert e_approach.num_weight_bins is None
            assert e_approach.objective is None
            assert e_approach.pool_size is None
            es = build_ensembles_logreg(
                y_preds=y_preds,
                y_true=y_true,
                max_models=e_approach.max_models,
                verbose=False,
            )
        else:
            assert e_approach.num_weight_bins is not None
            assert e_approach.objective is not None
            assert e_approach.pool_size is not None
            es = build_ensembles_beam_search(
                y_preds=y_preds,
                y_true=y_true,
                num_bins=e_approach.num_weight_bins,  # type: ignore
                pool_size=e_approach.pool_size,  # type: ignore
                max_models=e_approach.max_models,
                objective=e_approach.objective,  # type: ignore
                fold_id=fold_id,
                model_types=types if e_approach.filter_same_model_type else None,
                verbose=False,
            )
        es_ret = {}
        for key, e in es.items():
            e.model_idx = [items[i].idx for i in e.model_idx]
            es_ret[key] = e
        return es_ret

    scores_by_cnt_models = defaultdict(list)

    for valid in folds:
        mask_train = fold_id != valid
        mask_valid = ~mask_train
        items_train = []
        y_pred_list_valid = []
        for item in all_items:
            items_train.append(replace(item, pred=item.pred[mask_train]))
            y_pred_list_valid.append(item.pred[mask_valid])
        y_true_train = y_true[mask_train]
        y_true_valid = y_true[mask_valid]

        es = ensemble_build_helper(
            items=items_train,
            y_true=y_true_train,
            fold_id=fold_id[mask_train],
        )

        for cnt_models in range(1, e_approach.max_models + 1):
            e = es[cnt_models]
            assert len(items_train) == len(all_items)
            y_pred_ensemble_train = e.infer([it.pred for it in items_train])
            assert len(y_pred_list_valid) == len(all_items)
            y_pred_ensemble_valid = e.infer(y_pred_list_valid)
            th = calc_best_f1_threshold(
                y_true=y_true_train, y_pred=y_pred_ensemble_train
            )
            f1_valid = f1_score(y_true=y_true_valid, y_pred=y_pred_ensemble_valid >= th)
            scores_by_cnt_models[cnt_models].append(f1_valid)

    best_score = -1.0
    best_cnt_models = -1
    for cnt_models in range(1, e_approach.max_models + 1):
        score = np.mean(scores_by_cnt_models[cnt_models])
        # print(f"k={cnt_models}, f1={score:.5f}")
        if score > best_score:
            best_score = score
            best_cnt_models = cnt_models

    global_ensemble = ensemble_build_helper(
        items=all_items,
        y_true=y_true,
        fold_id=fold_id,
    )[best_cnt_models]

    assert not e_approach.refit_with_fold_avg
    # if e_approach.refit_with_fold_avg:
    #     items = [all_items[i] for i in global_ensemble.model_idx]
    #     w_by_idx = defaultdict(float)
    #     for valid in folds:
    #         mask_train = fold_id != valid
    #         mask_valid = ~mask_train
    #         items_train = []
    #         y_pred_list_valid = []
    #         for item in items:
    #             items_train.append(replace(item, pred=item.pred[mask_train]))
    #             y_pred_list_valid.append(item.pred[mask_valid])
    #         y_true_train = y_true[mask_train]
    #         y_true_valid = y_true[mask_valid]

    #         es = beam_search_helper(
    #             items=items_train,
    #             y_true=y_true_train,
    #             fold_id=fold_id[mask_train],
    #         )
    #         e = es[e_approach.max_models]
    #         for idx, w in zip(e.model_idx, e.weights):
    #             w_by_idx[idx] += w
    #     w = [0.0] * len(items)
    #     for pos, item in enumerate(items):
    #         w[pos] = w_by_idx[item.idx] / len(folds)
    #     assert np.abs((np.sum(w) - 1)) < 1e-10
    #     global_ensemble.weights = w

    return return_given_e(e=global_ensemble)


def build_ensembles_from_models_per_approach(
    models: list[OOF_Metrics_for_model],
    lab: str | None,
    e_approaches: list[EnsembleApproach],
    verbose: bool = False,
) -> dict[EnsembleApproach, tuple[OOF_Metrics, list[int], list[float]]]:
    assert models
    folds = models[0].folds
    for model in models:
        assert model.folds == folds

    y_pred_list = []
    model_types = []
    y_true = None
    fold_id = None
    for model in tqdm(models, "Assemble oof data"):
        y_pred, y_true_model, fold_id_model = model.oof_data(lab=lab)
        y_pred_list.append(y_pred)
        if y_true is None:
            y_true = y_true_model
        else:
            assert (
                y_true.shape == y_true_model.shape
            ), f"y_true.shape={y_true.shape}, y_true_model.shape={y_true_model.shape}, model={model.name}"
            assert (
                y_true.dtype == y_true_model.dtype
            ), f"y_true.dtype={y_true.dtype}, y_true_model.dtype={y_true_model.dtype}, model={model.name}"
        if fold_id is None:
            fold_id = fold_id_model
        else:
            assert fold_id.shape == fold_id_model.shape
            assert fold_id.dtype == fold_id_model.dtype
        model_types.append(model_type_by_name(model.name))

    assert y_true is not None
    assert fold_id is not None

    res = {}
    for app in tqdm(e_approaches, f"build_ensemble per approach (n={len(y_true)})"):
        # print(f"App: {app.to_str()}")
        e = build_ensemble(
            y_pred_list=y_pred_list,
            y_true=y_true,
            fold_id=fold_id,
            model_types=model_types,
            e_approach=app,
        )
        res[app] = e

        oof_metrics, model_idx, model_coefs = e
        if verbose:
            for idx, coef in zip(model_idx, model_coefs):
                print(f"{models[idx].name} * {coef:.3f}")

    gc.collect()
    torch.cuda.empty_cache()
    return res

