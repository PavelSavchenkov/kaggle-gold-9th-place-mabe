import time
from enum import Enum
from pathlib import Path
from tracemalloc import start

import cupy as cp  # type: ignore
import lightgbm  # type: ignore
import numpy as np
import xgboost as xgb  # type: ignore
from catboost import CatBoostClassifier  # type: ignore
from catboost import Pool  # type: ignore
from lightgbm import LGBMClassifier  # type: ignore
from tqdm import tqdm
from xgboost import XGBClassifier  # type: ignore

from common.config_utils import base_model_from_file
from common.helpers import ensure_1d_numpy
from gbdt.callbacks import (
    CustomCatBoostEvalMetric,
    CustomEvalPeriodicLogic,
    XgboostEvalCallback,
    XgboostTqdmCallback,
    make_lightgbm_eval_callback,
    make_lightgbm_eval_metric,
    raw_gbdt_inference,
    tqdm_callback_lightgbm,
)
from gbdt.configs import DataPreprocessConfig, GBDT_TrainConfig
from gbdt.helpers import (
    GBDT_Model_Type,
    get_data_preprocess_config_path,
    get_data_preprocess_config_path_from_ckpt_path,
)


class GBDT_Model:
    model_type: GBDT_Model_Type | None
    clf: XGBClassifier | LGBMClassifier | lightgbm.Booster | CatBoostClassifier
    train_config: GBDT_TrainConfig
    trained: bool = False
    data_preprocess_config: DataPreprocessConfig | None

    def __init__(self):
        self.model_type = None
        self.data_preprocess_config = None

    def set_default_data_preprocess_config(self):
        assert self.model_type is not None
        if self.model_type == GBDT_Model_Type.catboost:
            self.data_preprocess_config = DataPreprocessConfig(keep_nan=False)
        else:
            self.data_preprocess_config = DataPreprocessConfig(keep_nan=True)

    @staticmethod
    def from_config(config: GBDT_TrainConfig) -> "GBDT_Model":
        obj = GBDT_Model()
        obj.train_config = config.model_copy(deep=True)
        if config.xgboost_config is not None:
            obj.model_type = GBDT_Model_Type.xgboost
            xgb_params: dict = {
                "n_estimators": 500,
                "learning_rate": 0.09,
                "max_depth": 10,
                "min_child_weight": 5,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "device": "cuda",
                "random_state": config.seed,
                **config.xgboost_config.model_dump(exclude_none=True),
            }

            obj.clf = XGBClassifier(**xgb_params)
        elif config.lightgbm_config is not None:
            obj.model_type = GBDT_Model_Type.lightgbm
            obj.clf = LGBMClassifier(
                bagging_seed=config.seed,
                feature_fraction_seed=config.seed,
                extra_seed=config.seed,
                random_state=config.seed,
                **config.lightgbm_config.model_dump(exclude_none=True),
            )
        elif config.catboost_config is not None:
            obj.model_type = GBDT_Model_Type.catboost
            obj.clf = CatBoostClassifier(
                random_seed=config.seed,
                **config.catboost_config.model_dump(exclude_none=True),
            )
        else:
            raise ValueError(f"At least one gbdt config should be set")
        obj.set_default_data_preprocess_config()
        return obj

    @staticmethod
    def from_path(path_name_cv: Path) -> "GBDT_Model":
        obj = GBDT_Model()
        obj.train_config = base_model_from_file(
            GBDT_TrainConfig, path_name_cv / "train_config.json"
        )
        if obj.train_config.xgboost_config is not None:
            obj.model_type = GBDT_Model_Type.xgboost
            clf = XGBClassifier()
            clf.load_model(path_name_cv / "final_model" / "model.json")
            obj.clf = clf
        elif obj.train_config.lightgbm_config is not None:
            obj.model_type = GBDT_Model_Type.lightgbm
            obj.clf = lightgbm.Booster(
                model_file=path_name_cv / "final_model" / "model.txt"
            )
        elif obj.train_config.catboost_config is not None:
            obj.model_type = GBDT_Model_Type.catboost
            obj.clf = CatBoostClassifier()
            obj.clf.load_model(path_name_cv / "final_model" / "model.cbm")
        else:
            raise ValueError(
                f"At least one gbdt config should be set: {str(path_name_cv)}"
            )
        obj.set_default_data_preprocess_config()
        data_preprocess_config_path = get_data_preprocess_config_path_from_ckpt_path(
            path_name_cv
        )
        if data_preprocess_config_path.exists():
            obj.data_preprocess_config = base_model_from_file(
                DataPreprocessConfig, data_preprocess_config_path
            )
        obj.trained = True
        return obj

    def gpu_on(self):
        match self.model_type:
            case GBDT_Model_Type.xgboost:
                self.clf.set_params(device="cuda")
            case GBDT_Model_Type.lightgbm:
                raise ValueError("LightGBM does not support gpu for inference")
            case GBDT_Model_Type.catboost:
                self.clf.set_params(task_type="GPU")
            case _:
                raise ValueError

    def save(self, path: Path):
        path.mkdir(exist_ok=True, parents=True)
        assert self.trained
        match self.model_type:
            case GBDT_Model_Type.xgboost:
                self.clf.save_model(path / "model.json")
            case GBDT_Model_Type.lightgbm:
                if isinstance(self.clf, lightgbm.Booster):
                    booster = self.clf
                else:
                    booster = self.clf.booster_
                booster.save_model(path / "model.txt")
            case GBDT_Model_Type.catboost:
                self.clf.save_model(path / "model.cbm")
            case _:
                raise ValueError

    def fit(self, train_feats: dict, test_feats: dict, use_wandb: bool):
        X_train = train_feats["X"].data
        y_train = train_feats["y"].data
        sample_weight = train_feats["sample_weight"]
        self.custom_eval_logic = CustomEvalPeriodicLogic(
            model=self,
            eval_feats=test_feats,
            log_config=self.train_config.logging_config,
            use_wandb=use_wandb,
        )
        X_test = test_feats["X"].data
        y_test = test_feats["y"].data
        self._input_data_preprocess_inplace(X_train)
        self._input_data_preprocess_inplace(X_test)
        match self.model_type:
            case GBDT_Model_Type.xgboost:
                callbacks = []
                tqdm_cb = XgboostTqdmCallback(
                    total=self.clf.get_params()["n_estimators"]
                )
                callbacks.append(tqdm_cb)

                xgb_eval_cb = XgboostEvalCallback(self.custom_eval_logic)
                callbacks.append(xgb_eval_cb)

                self.clf.set_params(callbacks=callbacks)

                self.clf.fit(
                    X_train,
                    y_train,
                    sample_weight=sample_weight,
                    verbose=False,
                )
            case GBDT_Model_Type.lightgbm:
                assert isinstance(self.clf, LGBMClassifier)

                y_train = y_train.ravel()
                y_test = y_test.ravel()

                lightgbm_eval_metric = make_lightgbm_eval_metric(self.custom_eval_logic)
                lightgbm_eval_cb = make_lightgbm_eval_callback(self.custom_eval_logic)
                lightgbm_tqdm_cb = tqdm_callback_lightgbm(
                    total=self.clf.get_params()["n_estimators"]
                )

                eval_set = [(X_test, y_test)]
                self.clf.fit(
                    X_train,
                    y_train,
                    sample_weight=sample_weight,
                    eval_set=eval_set,
                    eval_metric=lightgbm_eval_metric,
                    callbacks=[
                        lightgbm_tqdm_cb,
                        lightgbm_eval_cb,
                    ],
                )
            case GBDT_Model_Type.catboost:

                def get_or_create_quantized_pools() -> Pool:
                    CACHE_DIR = Path("catboost_cache")
                    CACHE_DIR.mkdir(exist_ok=True, parents=True)

                    task_type = self.clf.get_params()["task_type"]
                    border_count = 128 if task_type == "GPU" else 254
                    nan_mode = "Min"

                    train_hash = train_feats["hash"]
                    test_hash = test_feats["hash"]

                    train_bin = CACHE_DIR / f"train_{train_hash}.bin"
                    borders_path = CACHE_DIR / f"borders_{train_hash}.bin"
                    test_bin = CACHE_DIR / f"test_{train_hash}_{test_hash}.bin"

                    if train_bin.exists():
                        print(
                            f"[DATA CACHE] Reusing existing TRAIN pool at {str(train_bin)} ..."
                        )
                        train_pool = Pool(data="quantized://" + str(train_bin))
                    else:
                        print(f"[DATA CACHE] Creating and caching TRAIN pool...")
                        start_time = time.time()
                        train_pool = Pool(
                            data=X_train,
                            label=y_train,
                            weight=sample_weight,
                            cat_features=None,
                        )
                        train_pool.quantize(
                            task_type=task_type,
                            nan_mode=nan_mode,
                            border_count=border_count,
                        )

                        train_pool.save(train_bin)
                        train_pool.save_quantization_borders(borders_path)

                        print(
                            f"[DATA CACHE] TRAIN pool created in {time.time() - start_time:.5f}"
                        )

                    if test_bin.exists():
                        print(
                            f"[DATA CACHE] Reusing existing TEST pool at {str(test_bin)} ..."
                        )
                        test_pool = Pool(data="quantized://" + str(test_bin))
                    else:
                        print(f"[DATA CACHE] Creating and caching TEST pool...")
                        start_time = time.time()
                        test_pool = Pool(
                            data=X_test,
                            label=y_test,
                            weight=None,
                            cat_features=None,
                        )
                        test_pool.quantize(
                            task_type=task_type,
                            nan_mode=nan_mode,
                            input_borders=borders_path,
                        )

                        test_pool.save(test_bin)

                        print(
                            f"[DATA CACHE] TEST pool created in {time.time() - start_time:.5f}"
                        )

                    return train_pool, test_pool

                train_pool, test_pool = get_or_create_quantized_pools()

                eval_set = test_pool

                eval_metric = CustomCatBoostEvalMetric(
                    custom_logic=self.custom_eval_logic
                )

                if self.custom_eval_logic.log_config.early_stop_on_metric is not None:
                    self.clf.set_params(
                        od_wait=self.custom_eval_logic.log_config.early_stop_patience_rounds
                    )
                else:
                    self.clf.set_params(od_pval=0.0, od_wait=1_000_000)

                self.clf.set_params(
                    eval_metric=eval_metric,
                )

                self.clf.fit(
                    train_pool,
                    eval_set=eval_set,
                    verbose=True,
                    use_best_model=False,
                    metric_period=self.custom_eval_logic.log_config.logging_steps,
                )
            case _:
                raise ValueError
        self.trained = True

    def get_after_train_metrics(self):
        assert self.trained
        return {
            "best_metric_values": self.custom_eval_logic.best_metric_values,
            "best_metric_steps": self.custom_eval_logic.best_metric_steps,
            "all_metric_values": self.custom_eval_logic.all_metric_values,
        }

    def get_booster(self):
        match self.model_type:
            case GBDT_Model_Type.xgboost:
                return self.clf.get_booster()
            case GBDT_Model_Type.lightgbm:
                if isinstance(self.clf, lightgbm.Booster):
                    booster = self.clf
                else:
                    booster = self.clf.booster_
                return booster
            case GBDT_Model_Type.catboost:
                return self.clf
            case _:
                raise ValueError

    def predict_prod(
        self,
        X: np.ndarray | cp.ndarray,
        num_trees: int,
        verbose: bool = False,
        threads_hint: int = -1,
    ) -> np.ndarray:
        num_threads = None
        batch_size = None
        if self.model_type == GBDT_Model_Type.lightgbm:
            num_threads = threads_hint
            batch_size = 20_000
        elif self.model_type == GBDT_Model_Type.xgboost:
            num_threads = threads_hint
        return self.predict_trained(
            X,
            num_trees,
            logits=False,
            num_threads=num_threads,
            batch_size=batch_size,
            verbose=verbose,
        )

    def predict_trained(
        self,
        X: np.ndarray | cp.ndarray,
        num_trees: int,
        logits: bool,
        num_threads: int | None = None,
        batch_size: int | None = None,
        verbose: bool = False,
    ) -> np.ndarray:
        assert self.trained
        if isinstance(X, np.ndarray):
            assert X.dtype == np.float32, f"actual X.dtype: {X.dtype}"

        start_time = time.time()

        X = X.copy()  # !!!!!!!!!!!!!!!!!!!
        self._input_data_preprocess_inplace(X)

        def predict(data):
            pred = raw_gbdt_inference(
                model=self.get_booster(),
                X=data,
                num_trees=num_trees,
                logits=logits,
                num_threads=num_threads,
            )
            if isinstance(pred, cp.ndarray):
                pred = cp.asnumpy(pred)
            pred = pred.astype(np.float32, copy=False)
            return pred

        n = len(X)
        if batch_size is None:
            pred = predict(X)
        else:
            preds = []
            for i in range(0, n, batch_size):
                data = X[i : i + batch_size]
                pred = predict(data)
                preds.append(pred)
            pred = np.concatenate(preds, axis=0)

        if pred.ndim == 2 and pred.shape[1] == 2:
            pred = pred[:, 1]
        pred = ensure_1d_numpy(pred)

        if verbose:
            elapsed = time.time() - start_time
            time_per = elapsed * (1_000_000 / n) * (300 / num_trees)
            print(
                f"[PREDICT] samples={n}, trees={num_trees}: {elapsed:.5f} -----------------> Per 1M samples x 300 trees: {time_per:.5f}"
            )
        return pred

    def print_post_train_stats(self):
        if self.model_type != GBDT_Model_Type.lightgbm:
            return

        booster = self.get_booster()
        info = booster.dump_model()

        print("num_features:", booster.num_feature())

        num_leaves_per_tree = [t["num_leaves"] for t in info["tree_info"]]
        print("avg num_leaves:", float(np.mean(num_leaves_per_tree)))
        print("max num_leaves:", int(np.max(num_leaves_per_tree)))

        def tree_depth(node):
            # node is a dict from "tree_structure"
            if "left_child" not in node and "right_child" not in node:
                return 1
            left_depth = tree_depth(node["left_child"]) if "left_child" in node else 0
            right_depth = (
                tree_depth(node["right_child"]) if "right_child" in node else 0
            )
            return 1 + max(left_depth, right_depth)

        depths = [tree_depth(t["tree_structure"]) for t in info["tree_info"]]
        print("avg depth:", float(np.mean(depths)))
        print("max depth:", int(np.max(depths)))

    def _input_data_preprocess_inplace(self, X: np.ndarray | cp.ndarray):
        assert self.data_preprocess_config is not None
        if self.data_preprocess_config.keep_nan:
            return X
        assert isinstance(X, np.ndarray)
        mask_nan = np.isnan(X)
        X[mask_nan] = -(2**10)
        return X

    def get_num_trees(self) -> int:
        assert (
            self.trained
        ), "Model must be trained or loaded before calling get_num_trees()"

        booster = self.get_booster()

        match self.model_type:
            case GBDT_Model_Type.xgboost:
                # booster is xgboost.core.Booster
                return int(booster.num_boosted_rounds())

            case GBDT_Model_Type.lightgbm:
                # booster is lightgbm.Booster
                return int(booster.num_trees())

            case GBDT_Model_Type.catboost:
                # booster is CatBoostClassifier
                if hasattr(booster, "tree_count_"):
                    return int(booster.tree_count_)
                return int(booster.get_tree_count())

            case _:
                raise ValueError("Unknown model_type in get_num_trees()")
