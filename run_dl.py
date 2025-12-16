import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from common.config_utils import base_model_from_file
from dl.postprocess import (
    full_training_steps_from_config_dl,
    get_best_pr_auc_ckpt_for_each_action_lab,
    get_latest_saved_checkpoint,
    get_train_config_dl,
    iter_all_checkpoints_for_model_cv,
    step_to_ratio,
)
from dl.submission import Submission
from dl.train import run_training_process

# base_name = "dl-tcn-base5"
# base_name = "dl-tcn-base9"
# base_name = "dl-tcn-base7bodypart-drop-v1-p0.1"
# base_config = get_train_config_dl(name=base_name, cv=0)
# assert base_config
# config = base_config.model_copy(deep=True)
# config.remove_25fps_adaptable_snail_from_train = True
# config.name += "-remove-25fps-adap-snail"
# run_training_process(config)
# exit(0)

SRC_SUBMISSION = Path("submissions/dl-tcn-ensemble6-9-dec")
submission = base_model_from_file(Submission, SRC_SUBMISSION / "submission.json")
names = list(sorted(set([model.name() for model in submission.models])))
names = [name + "-remove-25fps-adap-snail" for name in names]


def run_full_models(proc_id: int):
    assert proc_id == 0 or proc_id == 1
    remaining_steps = 0
    for seed in range(7):
        all_configs_to_train = []
        for name in names:
            configs = []
            for cv in range(5):
                config = get_train_config_dl(name=name, cv=cv)
                assert config is not None
                configs.append(config)

            ratios_per_action_lab = defaultdict(list)
            for config in configs:
                best_map = get_best_pr_auc_ckpt_for_each_action_lab(
                    save_dir=config.save_dir()
                )
                for (action, lab), (_, ckpt) in best_map.items():
                    ratio = step_to_ratio(config=config, step=ckpt.step)
                    ratios_per_action_lab[(action, lab)].append(ratio)

            max_ratio = 0.0
            for ratios in ratios_per_action_lab.values():
                ratio = float(np.median(ratios))
                max_ratio = max(max_ratio, ratio)

            config = get_train_config_dl(name=name, cv=0)
            assert config is not None
            config.data_split_config.test_fold = 0
            config.data_split_config.train_folds = list(range(5))
            config.seed = seed
            config.train_balance_config.seed = seed
            config.name += f"-full-seed{seed}"

            full_steps = full_training_steps_from_config_dl(train_config=config)
            config.max_steps_to_run = int(full_steps * max_ratio)
            config.save_steps = 300
            config.eval_steps = config.max_steps_to_run // 2

            config.resume_from_latest_ckpt = True

            full_steps = config.max_steps_to_run
            latest_ckpt = get_latest_saved_checkpoint(save_dir=config.save_dir())
            done_steps = 0
            if latest_ckpt is not None:
                done_steps = latest_ckpt.step
            remaining_steps += max(0, full_steps - done_steps)

            all_configs_to_train.append(config)

        print(f"remaining steps: {remaining_steps}")

        random.seed(0)
        random.shuffle(all_configs_to_train)
        all_configs_to_train = all_configs_to_train[proc_id::2]
        for config in all_configs_to_train:
            run_training_process(config)


def run_cv_models(proc_id: int):
    remaining_steps = 0
    assert proc_id == 0 or proc_id == 1
    all_configs_to_train = []
    for name in names:
        for test_fold in range(5):
            configs = []
            for cv in range(5):
                config = get_train_config_dl(name=name, cv=cv)
                assert config is not None
                configs.append(config)

            ratios_per_action_lab = defaultdict(list)
            for config in configs:
                best_map = get_best_pr_auc_ckpt_for_each_action_lab(
                    save_dir=config.save_dir()
                )
                for (action, lab), (_, ckpt) in best_map.items():
                    ratio = step_to_ratio(config=config, step=ckpt.step)
                    ratios_per_action_lab[(action, lab)].append(ratio)

            max_ratio = 0.0
            for ratios in ratios_per_action_lab.values():
                ratio = float(np.median(ratios))
                max_ratio = max(max_ratio, ratio)

            config = get_train_config_dl(name=name, cv=0)
            assert config is not None
            config.data_split_config.test_fold = test_fold
            config.remove_25fps_adaptable_snail_from_train = True
            config.name += f"-remove-25fps-adap-snail"

            full_steps = full_training_steps_from_config_dl(train_config=config)
            config.max_steps_to_run = int(full_steps * max_ratio)

            assert config.train_bs % 2 == 0
            config.train_bs //= 2
            config.gradient_accumulation_steps *= 2

            config.resume_from_latest_ckpt = True

            steps_ready = 0
            latest_ckpt = get_latest_saved_checkpoint(config.save_dir())
            if latest_ckpt is not None:
                steps_ready = latest_ckpt.step
            steps_full = config.max_steps_to_run
            assert steps_full > 0
            cur_remaining = max(0, steps_full - steps_ready)
            remaining_steps += cur_remaining

            all_configs_to_train.append((config, cur_remaining))

    print(f"Total Remain // 300: {remaining_steps//300}")

    random.seed(0)
    random.shuffle(all_configs_to_train)

    cur_remain = 0
    all_configs_to_train = all_configs_to_train[proc_id::2]
    for config, rem in all_configs_to_train:
        cur_remain += rem
    print(f"Remain for this proc // 300: {cur_remain//300}")
    # for config, _ in all_configs_to_train:
    #     run_training_process(config)


import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cv-list",
        "--cv_list",  # allow both spellings
        dest="cv_list",
        type=int,
        nargs="+",
        help="List of integers for cross-validation, e.g. --cv-list 0 1 2 3",
        required=False,
    )
    parser.add_argument(
        "--proc_id", "--proc-id", dest="proc_id", type=int, required=True
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # print("cv_list =", args.cv_list)
    print("proc_id =", args.proc_id)
    # run_cv_models(proc_id=args.proc_id)
    run_full_models(proc_id=args.proc_id)
