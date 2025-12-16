import json
import random
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from common.config_utils import base_model_from_file
from common.constants import ACTION_NAMES_IN_TEST, LABS_IN_TEST_PER_ACTION
from common.helpers import get_train_meta
from dl.configs import AugmentationsConfig, FeaturesConfigDL
from dl.features_lookup_gpu import FeaturesLookupGPU
from dl.oof_utils import (
    lazy_calc_oof_all_actions,
    lazy_calc_oof_all_actions_labs,
    ratio_to_step,
    step_to_ratio,
)
from dl.postprocess import (
    Ckpt,
    get_best_pr_auc_ckpt_for_each_action_lab,
    iter_all_checkpoints_for_model_cv,
    iter_all_configs_for_model_name_dl,
    iter_all_dl_model_names,
)
from dl.submission import Submission


def calc_all_metrics(proc_id: int, procs: int):
    names = list(
        sorted(iter_all_dl_model_names(take_filter=lambda name: "adap" in name))
    )
    to_run = []
    for name in names:
        for config in sorted(
            iter_all_configs_for_model_name_dl(name), key=lambda c: c.cv()
        ):
            for ckpt in sorted(
                iter_all_checkpoints_for_model_cv(config.save_dir()),
                key=lambda ck: ck.step,
            ):
                to_run.append((config, ckpt))

    random.seed(1)
    random.shuffle(to_run)

    to_run = to_run[proc_id::procs]

    meta = get_train_meta()
    meta = meta[meta.has_annotation]
    feats_lookup = FeaturesLookupGPU(
        meta, feats_config=FeaturesConfigDL(), aug_config=AugmentationsConfig()
    )

    for config, ckpt in to_run:
        if not ckpt.metrics_path().exists():
            feats_lookup.reset_configs(train_config=config)
            ckpt.lazy_calc_and_save_eval_data(feats_lookup=feats_lookup)


def calc_all_eval_data():
    meta = get_train_meta()
    meta = meta[meta.has_annotation]
    feats_lookup = FeaturesLookupGPU(
        meta, feats_config=FeaturesConfigDL(), aug_config=AugmentationsConfig()
    )

    to_train = json.load(open("to_train.json"))
    for name in to_train.keys():
        for config in iter_all_configs_for_model_name_dl(name):
            map_best = get_best_pr_auc_ckpt_for_each_action_lab(config.save_dir())
            for ckpt in iter_all_checkpoints_for_model_cv(config.save_dir()):
                action_lab_list = []
                for key, (_, ckpt_map) in map_best.items():
                    if ckpt_map.step == ckpt.step:
                        action_lab_list.append(key)
                if action_lab_list:
                    feats_lookup.reset_configs(config)
                    ckpt.lazy_calc_and_save_eval_data(
                        feats_lookup=feats_lookup, action_lab_list=action_lab_list
                    )
                    for action, lab in action_lab_list:
                        gt_path = ckpt.gt_npy_path(action=action, lab=lab)
                        assert gt_path.exists()
                        preds_path = ckpt.preds_npy_path(action=action, lab=lab)
                        assert preds_path.exists()


def calc_all_split_json():
    save_dirs = []
    for name in iter_all_dl_model_names():
        for config in iter_all_configs_for_model_name_dl(name):
            stats_path = config.save_dir() / "info" / "split.json"
            if not stats_path.exists():
                save_dirs.append(config.save_dir())
    print(f"LEN: {len(save_dirs)}")

    for save_dir in tqdm(save_dirs):
        subprocess.run(
            [
                "python",
                "-m",
                "dl.train",
                "--config",
                save_dir / "train_config.json",
            ],
            check=False,
        )
        assert (save_dir / "stats.npz").exists(), f"save_dir: {save_dir}"


def calc_oof_for_name(name: str):
    print(f"OOF per-action for {name}", flush=True)
    lazy_calc_oof_all_actions(name, median=True)


def calc_all_oof():
    names = []
    for name in iter_all_dl_model_names():
        configs = list(iter_all_configs_for_model_name_dl(name=name))
        if len(configs) < 5:
            continue
        assert len(configs) == 5
        names.append(name)

    with ProcessPoolExecutor(max_workers=8) as ex:
        futures = []
        for name in names:
            future = ex.submit(calc_oof_for_name, name)
            futures.append(future)
        for future in tqdm(as_completed(futures), total=len(names)):
            future.result()

    # for name in tqdm(names):
    #     print(f"OOF for: {name}")
    #     lazy_calc_oof_all_actions_labs(name, median=True)


def calc_all_eval_data_median():
    meta = get_train_meta()
    meta = meta[meta.has_annotation]
    feats_lookup = FeaturesLookupGPU(
        meta, feats_config=FeaturesConfigDL(), aug_config=AugmentationsConfig()
    )

    total_need = 0
    to_train = json.load(open("to_train.json"))
    for name in to_train.keys():
        config_per_fold = {}
        map_best_per_fold = {}
        all_steps_per_fold = defaultdict(list)

        for config in iter_all_configs_for_model_name_dl(name):
            config_per_fold[config.cv()] = config
            map_best_per_fold[config.cv()] = get_best_pr_auc_ckpt_for_each_action_lab(
                config.save_dir()
            )
            for ckpt in iter_all_checkpoints_for_model_cv(config.save_dir()):
                all_steps_per_fold[config.cv()].append(ckpt.step)
        if len(map_best_per_fold) != 5:
            continue

        action_lab_list_per_fold_step = defaultdict(lambda: defaultdict(list))
        for action in ACTION_NAMES_IN_TEST:
            for lab in LABS_IN_TEST_PER_ACTION[action]:
                key = (action, lab)
                folds = []
                for fold in range(5):
                    if key not in map_best_per_fold[fold]:
                        continue
                    folds.append(fold)
                assert len(folds) > 1

                for fold in folds:
                    ratios_other = []
                    for cv in folds:
                        if cv != fold:
                            step = map_best_per_fold[cv][key][1].step
                            ratio = step_to_ratio(config=config_per_fold[cv], step=step)
                            ratios_other.append(ratio)
                    median_ratio = float(np.median(ratios_other))
                    target_step = ratio_to_step(
                        config=config_per_fold[fold], ratio=median_ratio
                    )
                    action_lab_list_per_fold_step[fold][target_step].append(key)

                    for cv in folds:
                        if cv == fold:
                            continue
                        step = ratio_to_step(
                            config=config_per_fold[cv], ratio=median_ratio
                        )
                        action_lab_list_per_fold_step[cv][step].append(key)

                all_ratios = []
                for cv in folds:
                    step = map_best_per_fold[cv][key][1].step
                    ratio = step_to_ratio(config=config_per_fold[cv], step=step)
                    all_ratios.append(ratio)
                median_ratio = float(np.median(all_ratios))
                for fold in folds:
                    step = ratio_to_step(
                        config=config_per_fold[fold], ratio=median_ratio
                    )
                    action_lab_list_per_fold_step[fold][step].append(key)

        for fold in range(5):
            config = config_per_fold[fold]
            feats_lookup.reset_configs(config)
            for step, action_lab_list in action_lab_list_per_fold_step[fold].items():
                ckpt = Ckpt(path=config.save_dir() / f"checkpoint-{step}", step=step)
                for action, lab in action_lab_list:
                    gt_path = ckpt.gt_npy_path(action=action, lab=lab)
                    preds_path = ckpt.preds_npy_path(action=action, lab=lab)
                    if not gt_path.exists() or not preds_path.exists():
                        total_need += 1
                        print(f"need: {total_need}")
                ckpt.lazy_calc_and_save_eval_data(
                    feats_lookup=feats_lookup, action_lab_list=action_lab_list
                )
                for action, lab in action_lab_list:
                    gt_path = ckpt.gt_npy_path(action=action, lab=lab)
                    preds_path = ckpt.preds_npy_path(action=action, lab=lab)
                    assert gt_path.exists()
                    assert preds_path.exists()

        # for ckpt in iter_all_checkpoints_for_model_cv(config.save_dir()):
        #     action_lab_list = []
        #     for key, (_, ckpt_map) in map_best.items():
        #         if ckpt_map.step == ckpt.step:
        #             action_lab_list.append(key)
        #     if action_lab_list:
        #         feats_lookup.reset_configs(config)
        #         ckpt.try_calc_and_save_eval_data(
        #             feats_lookup=feats_lookup, action_lab_list=action_lab_list
        #         )
        #         for action, lab in action_lab_list:
        #             gt_path = ckpt.gt_npy_path(action=action, lab=lab)
        #             assert gt_path.exists()
        #             preds_path = ckpt.preds_npy_path(action=action, lab=lab)
        #             assert preds_path.exists()

    print(f"total_need is {total_need}")


def calc_absolutely_all_eval_data():
    meta = get_train_meta()
    meta = meta[meta.has_annotation]
    feats_lookup = FeaturesLookupGPU(
        meta, feats_config=FeaturesConfigDL(), aug_config=AugmentationsConfig()
    )

    # PREF = "dl-tcn-base4-seed"

    submission = base_model_from_file(
        Submission, "submissions/dl-tcn-median-6-dec/submission.json"
    )
    names = [model.name() for model in submission.models]
    names = list(sorted(set(names)))

    cnt = 0
    # to_train = json.load(open("to_train.json"))
    # for name in to_train.keys():
    for name in names:
        # for name in iter_all_dl_model_names():
        #     if not name.startswith(PREF):
        #         continue
        for config in iter_all_configs_for_model_name_dl(name):
            for ckpt in iter_all_checkpoints_for_model_cv(config.save_dir()):
                for action in ACTION_NAMES_IN_TEST:
                    for lab in LABS_IN_TEST_PER_ACTION[action]:
                        gt_path = ckpt.gt_npy_path(action=action, lab=lab)
                        preds_path = ckpt.preds_npy_path(action=action, lab=lab)
                        if not gt_path.exists() or preds_path.exists():
                            cnt += 1
                            print(f"cnt: {cnt}")
                feats_lookup.reset_configs(config)
                ckpt.lazy_calc_and_save_eval_data(
                    feats_lookup=feats_lookup, action_lab_list=None
                )


# import sys
# proc_id = int(sys.argv[1])
# procs = int(sys.argv[2])
# assert 0 <= proc_id < procs
# calc_all_metrics(proc_id=proc_id, procs=procs)

calc_all_oof()
# calc_all_eval_data_median()
# calc_absolutely_all_eval_data()
