from __future__ import annotations

import catboost  # type: ignore
import cupy as cp  # type: ignore
import lightgbm  # type: ignore
import numpy as np
import wandb  # type: ignore
import xgboost as xgb  # type: ignore
from lightgbm.callback import EarlyStopException  # type: ignore
from scipy.special import expit
from tqdm.auto import tqdm
from xgboost.callback import TrainingCallback  # type: ignore

from eval_metrics import compute_eval_metrics
from gbdt.configs import LoggingConfig
from gbdt.helpers import GBDT_Model_Type


def raw_gbdt_inference(
    model,
    X: np.ndarray | cp.ndarray,
    num_trees: int,
    logits: bool,
    num_threads: int | None = None,
):
    if isinstance(model, xgb.Booster):
        if num_threads is not None:
            model.set_param({"nthread": num_threads})

        return model.inplace_predict(
            X,
            iteration_range=(0, num_trees),
            predict_type="margin" if logits else "value",
        )
    elif isinstance(model, lightgbm.Booster):
        if isinstance(X, cp.ndarray):
            X = cp.asnumpy(X)

        params = {"data": X, "num_iteration": num_trees, "raw_score": logits}
        if num_threads is not None:
            params["num_threads"] = num_threads
        return model.predict(**params)
    elif isinstance(model, catboost.CatBoostClassifier):
        if isinstance(X, cp.ndarray):
            X = cp.asnumpy(X)
        if num_threads is None:
            num_threads = -1
        return model.predict(
            X,
            prediction_type="RawFormulaVal" if logits else "Probability",
            ntree_end=num_trees,
            thread_count=num_threads,
            task_type="CPU"
        )
    else:
        raise ValueError(f"Unrecognized model class: {type(model).__name__}")


class XgboostTqdmCallback(TrainingCallback):
    def __init__(self, total: int, desc: str = "XGB Training", leave: bool = True):
        self.total = total
        self.desc = desc
        self.leave = leave
        self._pbar = None

    def before_training(self, model):
        self._pbar = tqdm(total=self.total, desc=self.desc, leave=self.leave)
        return model

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        if self._pbar is not None:
            self._pbar.update(1)
        return False

    def after_training(self, model):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        return model


def tqdm_callback_lightgbm(
    total: int, desc: str = "LightGBM Training", leave: bool = True
):
    pbar = None

    def _callback(env: lightgbm.callback.CallbackEnv) -> None:
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total, desc=desc, leave=leave)

        if pbar.n < pbar.total:
            pbar.update(1)

        if pbar.n >= pbar.total:
            pbar.close()

    _callback.order = 10
    return _callback


class CustomEvalPeriodicLogic:
    absolute_tolerance: float = 0  # 5e-4

    def __init__(
        self, model, eval_feats: dict, log_config: LoggingConfig, use_wandb: bool
    ):
        self.X_eval = eval_feats["X"].data
        if model.model_type == GBDT_Model_Type.xgboost:
            self.X_eval = cp.asarray(self.X_eval)
        self.y_eval = eval_feats["y"].data
        self.class_names = list(eval_feats["y"].columns)
        self.index_df = eval_feats["index"]

        self.model = model

        self.log_config = log_config.model_copy(deep=True)

        self.best_metric_values: dict[str, float] = {}
        self.best_metric_steps: dict[str, int] = {}
        self.all_metric_values: dict[str, dict[int, float]] = {}

        self.should_stop = False
        self.use_wandb = use_wandb
        self.latest_metrics = {}

    def _update_test_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.latest_metrics = dict(metrics)
        for name, v in metrics.items():
            assert isinstance(v, float)

            if name not in self.all_metric_values:
                self.all_metric_values[name] = {}
            self.all_metric_values[name][step] = v

            if name not in self.best_metric_values:
                better = True
            else:
                if "log-loss" in name:
                    better = v + self.absolute_tolerance < self.best_metric_values[name]
                elif "pr-auc" in name or "f1-best-th" in name:
                    better = v > self.best_metric_values[name] + self.absolute_tolerance
                else:
                    raise ValueError(f"Bad metric name: {name}")

            if better:
                self.best_metric_values[name] = v
                self.best_metric_steps[name] = step

        assert "test-prod-avg/pr-auc" in self.best_metric_values

    def should_log_now(self, epoch: int) -> bool:
        cur_iter = epoch + 1
        if cur_iter < 5:
            return True
        if cur_iter < 50 and cur_iter % 5 == 0:
            return True
        if cur_iter % self.log_config.logging_steps == 0:
            return True
        return False

    def after_iteration(
        self,
        epoch: int,
        model=None,
        preds: np.ndarray | None = None,
    ) -> bool:
        assert (model is None) != (
            preds is None
        ), f"Exactly one of model and preds should be None. Model is None: {model is None}. Preds is None: {preds is None}"
        cur_iter = epoch + 1
        if self.should_log_now(epoch=epoch):
            if preds is None:
                assert model is not None
                preds = raw_gbdt_inference(
                    model=model, X=self.X_eval, num_trees=cur_iter, logits=False
                )
                if isinstance(preds, cp.ndarray):
                    preds = cp.asnumpy(preds)

            assert preds is not None
            test_metrics = compute_eval_metrics(
                y_true=self.y_eval,
                y_pred=preds,
                class_names=self.class_names,
                index_df=self.index_df,
                prefix="test-",
            )

            if self.use_wandb:
                wandb.log(test_metrics, step=cur_iter)

            self._update_test_metrics(test_metrics, step=cur_iter)

        metric_key = self.log_config.early_stop_on_metric
        if metric_key is not None:
            last_best = self.best_metric_steps[metric_key]
            no_improve_for = cur_iter - last_best
            if no_improve_for >= self.log_config.early_stop_patience_rounds:
                self.should_stop = True
        return self.should_stop


def make_lightgbm_eval_callback(custom_logic: CustomEvalPeriodicLogic):
    def _callback(env):
        if custom_logic.should_stop:
            metric_key = custom_logic.log_config.early_stop_on_metric
            assert metric_key is not None
            best_step_0based = custom_logic.best_metric_steps[metric_key] - 1
            best_value = custom_logic.best_metric_values[metric_key]
            is_higher_better = False if "log-loss" in metric_key else True
            best_score = ("custom eval", metric_key, best_value, is_higher_better)
            raise EarlyStopException(
                best_iteration=best_step_0based, best_score=[best_score]
            )

    _callback.order = 50
    return _callback


def make_lightgbm_eval_metric(custom_logic: CustomEvalPeriodicLogic):
    epoch = 0

    def _eval_metric(y_true, y_pred, weight=None):
        nonlocal epoch
        # y_pred = np.asarray(y_pred)
        y_pred = y_pred.reshape(-1, 1)

        custom_logic.after_iteration(epoch=epoch, preds=y_pred)
        epoch += 1

        return "dummy-metric", 0.0, True

    return _eval_metric


class XgboostEvalCallback(TrainingCallback):
    callback: CustomEvalPeriodicLogic

    def __init__(self, callback: CustomEvalPeriodicLogic):
        self.callback = callback

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log):
        return self.callback.after_iteration(model=model, epoch=epoch)


class CustomCatBoostEvalMetric:
    def __deepcopy__(self, memo):
        # Always reuse the same instance
        memo[id(self)] = self
        return self

    def __init__(self, custom_logic):
        self.custom_logic = custom_logic
        self.epoch = 0

    def is_max_optimal(self):
        # higher metric is better
        return True

    def get_final_error(self, error, weight):
        return error

    def evaluate(self, approxes, target, weight):
        """
        approxes: list of 1D containers (one per dimension). For binary cls: len(approxes) == 1.
        target: 1D container with true labels for *the dataset this call is for*.
        weight: 1D container with sample weights or None.
        """
        assert len(approxes) == 1
        approx = np.array(approxes[0], dtype=float)  # logits
        probs = expit(approx)
        y_true = target.reshape(-1, 1)

        custom_logic = self.custom_logic
        if len(y_true) != len(custom_logic.y_eval):
            return 0.0, 1.0

        custom_logic.after_iteration(epoch=self.epoch, preds=probs)
        self.epoch += 1

        metric_key = custom_logic.log_config.early_stop_on_metric
        if metric_key is None:
            metric_key = "test-prod-avg/pr-auc"
        metric_value = custom_logic.latest_metrics[metric_key]

        return metric_value, 1.0
