import os

import optuna  # type: ignore
from optuna.integration.wandb import WeightsAndBiasesCallback  # type: ignore
from pydantic import BaseModel

from gbdt.configs import GBDT_TrainConfig, XGBoostConfig
from gbdt.rebalance_utils import SampleCoefsParams


def suggest_params_xgboost(trial):
    params = {}

    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
    params["min_child_weight"] = trial.suggest_float("min_child_weight", 1.0, 20.0)
    params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.4, 1.0)

    params["gamma"] = trial.suggest_float("gamma", 0.0, 10.0)  # min_split_loss
    params["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)  # L1
    params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)  # L2

    params["max_bin"] = trial.suggest_int("max_bin", 256, 1024)

    params["scale_pos_weight"] = trial.suggest_float(
        "scale_pos_weight", 0.2, 10.0, log=True
    )

    params["n_estimators"] = 1000
    return params


def suggest_params_lightgbm(trial):
    params = {}

    params["learning_rate"] = trial.suggest_float(
        "learning_rate",
        5e-3,
        0.2,
        log=True,
    )

    # XGB: 1000; LGB often likes a bit more trees with small lr
    params["n_estimators"] = 300  # 3000

    # In LightGBM, num_leaves is typically more important than max_depth.
    # Keep max_depth unlimited (-1), and control complexity via num_leaves.
    params["max_depth"] = -1

    params["num_leaves"] = trial.suggest_int(
        "num_leaves",
        8,
        512,
        log=True,
    )

    params["min_child_samples"] = trial.suggest_int(
        "min_child_samples",
        5,
        100,
    )

    params["subsample"] = trial.suggest_float(
        "subsample",
        0.5,
        1.0,
    )
    params["subsample_freq"] = trial.suggest_int(
        "subsample_freq",
        1,
        10,
    )

    params["colsample_bytree"] = trial.suggest_float(
        "colsample_bytree",
        0.4,
        1.0,
    )

    params["reg_alpha"] = trial.suggest_float(
        "reg_alpha",
        1e-8,
        10.0,
        log=True,
    )
    params["reg_lambda"] = trial.suggest_float(
        "reg_lambda",
        1e-8,
        10.0,
        log=True,
    )

    params["max_bin"] = trial.suggest_int(
        "max_bin",
        63,
        255,
    )

    # params["scale_pos_weight"] = trial.suggest_float(
    #     "scale_pos_weight",
    #     2,
    #     16,
    #     log=True,
    # )

    params["passive_percentage"] = trial.suggest_float("passive_percentage", 0.3, 0.95)

    return params


def suggest_params_catboost(trial):
    params = {}

    params["learning_rate"] = trial.suggest_float(
        "learning_rate",
        0.05,
        0.25,
        log=True,
    )

    params["n_estimators"] = 500

    # params["depth"] = trial.suggest_int(
    #     "depth",
    #     4,
    #     10,
    # )

    # Main L2 regularization in CatBoost
    params["l2_leaf_reg"] = trial.suggest_float(
        "l2_leaf_reg",
        0.1,
        10.0,
        log=True,
    )

    # Random noise in score for splits (helps regularization)
    params["random_strength"] = trial.suggest_float(
        "random_strength",
        0.0,
        10.0,
    )

    # Stochasticity of bagging; 0 = no bagging, higher = more conservative trees
    params["bagging_temperature"] = trial.suggest_float(
        "bagging_temperature",
        0.0,
        5.0,
    )

    params["min_data_in_leaf"] = trial.suggest_int(
        "min_data_in_leaf",
        1,
        256,
        log=True,
    )

    # params["subsample"] = trial.suggest_float(
    #     "subsample",
    #     0.5,
    #     1.0,
    # )

    # params["colsample_bylevel"] = trial.suggest_float(
    #     "colsample_bylevel",
    #     0.4,
    #     1.0,
    # )

    # Lower = faster, stronger regularization.
    params["max_bin"] = trial.suggest_int(
        "max_bin",
        63,
        255,
    )

    # ---- passive_percentage: [-1] U [0, 1] ----
    # passive_mode = trial.suggest_categorical(
    #     "passive_mode",
    #     ["disabled", "enabled"],
    # )
    passive_mode = "disabled"

    if passive_mode == "disabled":
        params["passive_percentage"] = -1.0
    else:
        params["passive_percentage"] = trial.suggest_float(
            "passive_percentage",
            0.3,
            0.95,
        )

    return params


def apply_params_to_config(config, params: dict):
    return config.model_copy(update=params)


def _apply_params_pre(config: GBDT_TrainConfig, params: dict):
    key = "passive_percentage"
    assert key in params
    if config.sample_coefs_params_train is not None:
        config.sample_coefs_params_train.passive_percentage = params[key]
    else:
        config.sample_coefs_params_train = SampleCoefsParams(
            labs_baseline="uniform", passive_percentage=params[key]
        )


def apply_params_xgboost(config: GBDT_TrainConfig, params: dict):
    config = config.model_copy(deep=True)
    _apply_params_pre(config, params)
    config.xgboost_config = apply_params_to_config(config.xgboost_config, params)
    return config


def apply_params_lightgbm(config: GBDT_TrainConfig, params: dict):
    config = config.model_copy(deep=True)
    _apply_params_pre(config, params)
    config.lightgbm_config = apply_params_to_config(config.lightgbm_config, params)
    return config


def apply_params_catboost(config: GBDT_TrainConfig, params: dict):
    config = config.model_copy(deep=True)
    _apply_params_pre(config, params)
    config.catboost_config = apply_params_to_config(config.catboost_config, params)
    return config


def run_optuna_multirun(
    objective, group: str, study_name: str, n_trials: int, metric_name: str = "pr-auc"
):
    wandb_kwargs = {
        "project": os.environ.get("WANDB_PROJECT"),
        "group": group,
    }

    wandb_callback = WeightsAndBiasesCallback(
        metric_name=metric_name,
        wandb_kwargs=wandb_kwargs,
        as_multirun=True,
    )

    STORAGE = "sqlite:///optuna.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=STORAGE,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=42),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[wandb_callback],
        n_jobs=1,
        show_progress_bar=True,
        catch=(Exception,),
    )

    print(f"Study best_value: {study.best_value:.5f}")
