import copy
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from common.config_utils import (
    base_model_from_file,
    base_model_to_file,
    base_model_to_str,
)
from common.constants import (
    ACTION_NAMES_IN_TEST,
    ALL_ACTION_NAMES,
    ALL_PAIR_ACTIONS,
    ALL_SELF_ACTIONS,
)
from gbdt.configs import GBDT_TrainConfig, LightGBMConfig
from gbdt.helpers import get_train_config, is_fully_trained, read_metrics
from gbdt.ready_configs import make_default_data_split_config
from gbdt.rebalance_utils import (
    DownsampleParams,
    SampleCoefsParams,
    TestDownsampleParams,
    get_labs_in_test_with_action,
)
from optuna_utils import (
    apply_params_lightgbm,
    run_optuna_multirun,
    suggest_params_lightgbm,
)
from postprocess.submission_utils import Submission


def run_training(base_config: GBDT_TrainConfig) -> bool:
    config = copy.deepcopy(base_config)
    name = config.group
    cv = config.data_split_config.test_fold

    config.run_name = f"{name}_cv{cv}"
    save_dir = Path("train_logs") / name / f"cv{cv}"

    if is_fully_trained(name, cv):
        print(
            f"Final model for {config.run_name} already exists. Do NOT run training..."
        )
        return False
    else:
        print(f"Running training for '{config.run_name}'...")

    config.save_dir = str(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = save_dir / "train_config.json"

    base_model_to_file(config, config_path)
    subprocess.run(["python", "-m", "gbdt.train", "--config", config_path], check=False)
    return True


import numpy as np

from gbdt.helpers import GBDT_Model_Type, get_any_train_config, model_type_by_name
from postprocess.postprocess_utils import (
    top_models_by_oof_per_action,
    top_models_by_oof_per_action_per_lab,
)


def run_hflip():
    submission = base_model_from_file(
        Submission, "submissions/e-nested-f1-all/submission.json"
    )

    configs = [
        get_train_config(model.name, cv)
        for model in submission.models
        for cv in model.folds
    ]
    configs.sort(key=lambda config: (config.group, config.data_split_config.test_fold))

    for base_config in configs:
        if base_config.action != "sniff":
            continue
        if "xgboost" not in base_config.group:
            continue
        if base_config.action not in ALL_PAIR_ACTIONS:
            continue
        for seed in range(5):
            config = base_config.model_copy(deep=True)
            config.features_config.hflip = True
            config.features_config.flip_sides_if_hflip = True
            config.group += f"-hflip-sides-fix-seed{seed}"
            config.seed = seed
            config.train_downsample_params.seed = seed
            config.use_wandb = True
            run_training(config)

    # for train_size_coef in [0.5, 1.0]:
    #     for config in tqdm(configs, desc="All configs from best submission"):
    #         config.features_config.hflip = True
    #         config.group += f"-hflip-train{train_size_coef}"
    #         config.train_downsample_params.total_duration_cap *= train_size_coef
    #         run_training(config)


def run_full_models():
    submission = base_model_from_file(
        Submission, "submissions/e-nested-f1-all/submission.json"
    )

    NAME = "lgbm-climb-base-NiftyGoldfinch-x16-5k"

    for seed in range(5):
        num_trees_by_name = defaultdict(int)
        for model in submission.models:
            num_trees = max(model.steps)
            name = model.name
            name = get_any_train_config(model.name).group
            if name == NAME:
                print(f"CUR NUM TREES IS {num_trees}")
            num_trees_by_name[name] = max(num_trees_by_name[name], num_trees)

        print(f"trees needed for {NAME} is {num_trees_by_name[NAME]}")

        configs = []
        seen = set()
        for model in submission.models:
            config = get_any_train_config(model.name)
            assert config is not None
            config.data_split_config.train_folds = list(range(5))
            config.train_downsample_params.total_duration_cap *= 5 / 4
            config.logging_config.early_stop_on_metric = None
            config.seed = seed
            config.train_downsample_params.seed = seed
            config.group += f"-full-seed{seed}"
            config.data_split_config.test_fold = 0
            config.use_wandb = True
            if model.name == NAME:
                print(
                    f"CHECKING {NAME}, config.group is {config.group}, seen: {config.group in seen}"
                )
            # if config.group in seen:
            #     continue
            # seen.add(config.group)
            num_trees = num_trees_by_name[model.name]
            assert is_fully_trained(config.group, 0)
            if model.name == NAME:
                print(f"!!!!!!!!!!!!!!!!!!!!!! {NAME}")
            config_trained = get_train_config(config.group, 0)
            if model.name == NAME:
                print(
                    f"get_num_trees in config_trained: {config_trained.get_num_trees()}, num_trees is {num_trees}"
                )
            if config_trained.get_num_trees() >= num_trees:
                continue
            config.group += "-fix"
            config.set_num_trees(num_trees)
            if model.name == NAME:
                print(f"APPEND CONFIG WITH GROUP: {config.group} AND TREES {num_trees}")
            configs.append(config)
        configs.sort(key=lambda config: config.group)
        for config in tqdm(configs, desc=f"FULL MODELS FOR SEED={seed}"):
            print(f"TREES: {config.get_num_trees()}")
            run_training(config)


run_hflip()
# run_full_models()
