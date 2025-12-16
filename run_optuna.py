import copy
import subprocess
from pathlib import Path

from tqdm import tqdm

from common.constants import ACTION_NAMES_IN_TEST, ALL_ACTION_NAMES, ALL_SELF_ACTIONS
from common.config_utils import base_model_from_file, base_model_to_file, base_model_to_str
from gbdt.configs import GBDT_TrainConfig, LightGBMConfig
from gbdt.helpers import get_train_config, is_fully_trained, read_metrics
from gbdt.ready_configs import make_default_data_split_config
from gbdt.rebalance_utils import (
    DownsampleParams,
    SampleCoefsParams,
    TestDownsampleParams,
    get_labs_in_test_with_action,
)
from gbdt.train import try_fill_test_fold_for_model_cv
from optuna_utils import (
    apply_params_lightgbm,
    run_optuna_multirun,
    suggest_params_lightgbm,
)

# from postprocess_utils import top_models_per_action, top_models_per_action_per_lab


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


from gbdt.configs import CatBoostConfig
from optuna_utils import apply_params_catboost, suggest_params_catboost
from postprocess.postprocess_utils import top_models_by_oof_per_action_per_lab


baseline_name = "catb-sniff-optuna-model88-3k-feats2"
action = "sniff"

print(f"baseline = {baseline_name}")

def to_maximize(params):
    config = get_train_config(baseline_name, 0)

    pref = f"catb-{action}-optuna"
    it = 0
    while True:
        name = f"{pref}-model{it}"
        path = Path("train_logs") / name
        if not path.exists():
            break
        it += 1
    config.lightgbm_config = None
    config.catboost_config = CatBoostConfig()
    config = apply_params_catboost(config, params)
    config.catboost_config.task_type = "GPU"
    config.group = name
    config.use_wandb = False
    config.logging_config.early_stop_on_metric = "test-prod-avg/pr-auc"
    run_training(base_config=config)
    metrics = read_metrics(name=name, cv=config.data_split_config.test_fold)
    return metrics["best_metric_values"]["test-prod-avg/pr-auc"]

def objective(trial):
    params = suggest_params_catboost(trial)
    return to_maximize(params)

run_optuna_multirun(
    objective=objective,
    group=f"catb-{action}-optuna-group",
    study_name=f"catb-{action}-optuna-study",
    n_trials=45,
)
