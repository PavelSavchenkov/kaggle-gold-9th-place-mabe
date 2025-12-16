from collections import defaultdict
from functools import lru_cache

import numpy as np
from sklearn.metrics import f1_score

from common.constants import (
    ACTION_NAMES_IN_TEST,
    LAB_NAMES_IN_TEST,
    LABS_IN_TEST_PER_ACTION,
)
from common.helpers import get_train_meta
from common.metrics_common import calc_best_f1_threshold, calc_pr_auc
from dl.configs import AugmentationsConfig, DL_TrainConfig, FeaturesConfigDL
from dl.features_lookup_gpu import FeaturesLookupGPU
from dl.metrics import ProdMetricCompHelper
from dl.postprocess import (
    Ckpt,
    get_best_pr_auc_ckpt_for_each_action_lab,
    get_train_config_dl,
    iter_all_checkpoints_for_model_cv,
    ratio_to_step,
    step_to_ratio,
    train_test_meta_from_config_dl,
)
from dl.train import run_training_process

REF_NAME = "dl-tcn-base3-5.0s-45"  # for retrieving generic info, e.g. data from cv0, same for all models
names = ["dl-tcn-base9", "dl-tcn-base3-5.0s-45"]


def get_config(base_name: str, valid: int, test: int, seed: int) -> DL_TrainConfig:
    assert base_name in names
    name = base_name + f"-no-fold-{valid}-seed{seed}"
    cv = test
    if test == valid:
        name += "-full"
    config = get_train_config_dl(name=name, cv=cv)
    assert config is not None
    return config


@lru_cache
def get_median_ratio(
    base_name: str, valid: int, seed: int, action: str, lab: str
) -> float:
    key = (action, lab)
    ratios = []
    for cv in range(5):
        if cv == valid:
            continue
        config = get_config(base_name=base_name, valid=valid, test=cv, seed=seed)
        best_map = get_best_pr_auc_ckpt_for_each_action_lab(config.save_dir())
        if key not in best_map:
            continue
        _, ckpt = best_map[key]
        step = ckpt.step
        ratio = step_to_ratio(config=config, step=step)
        ratios.append(ratio)
    median = float(np.median(ratios))
    return median


def get_suff(valid: int) -> str:
    return f"_valid{valid}"


def lazy_generate_valid_preds(
    base_name: str,
    valid: int,
    seed: int,
    test: int,
    feats_lookup: FeaturesLookupGPU | None = None,
):
    # if test == valid:
    #     return

    suff = get_suff(valid=valid)
    ref_config = get_train_config_dl(name=REF_NAME, cv=valid)
    assert ref_config is not None, f"name={REF_NAME}, cv={valid}"
    ref_ckpt = list(iter_all_checkpoints_for_model_cv(ref_config.save_dir()))[0]
    action_lab_set_in_valid = set(ref_ckpt.all_action_lab_from_metrics())

    action_lab_list_by_step = defaultdict(list)
    config = get_config(base_name=base_name, valid=valid, test=test, seed=seed)
    for action in ACTION_NAMES_IN_TEST:
        for lab in LABS_IN_TEST_PER_ACTION[action]:
            if (action, lab) not in action_lab_set_in_valid:
                continue
            ratio = get_median_ratio(
                base_name=base_name, valid=valid, seed=seed, action=action, lab=lab
            )
            step = ratio_to_step(config=config, ratio=ratio)
            action_lab_list_by_step[step].append((action, lab))

    if not action_lab_list_by_step:
        return

    if feats_lookup is None:
        meta = get_train_meta()
        feats_lookup = FeaturesLookupGPU(
            meta=meta, feats_config=FeaturesConfigDL(), aug_config=AugmentationsConfig()
        )

    _, test_meta = train_test_meta_from_config_dl(train_config=ref_config)
    for step, action_lab_list in action_lab_list_by_step.items():
        ckpt = Ckpt.from_config_and_step(config=config, step=step)
        ckpt.lazy_calc_and_save_eval_data(
            feats_lookup=feats_lookup,
            action_lab_list=action_lab_list,
            test_meta=test_meta,
            suff=suff,
        )


def run_training():
    for valid in range(5):
        for name in ["dl-tcn-base9", "dl-tcn-base3-5.0s-45"]:
            base_config = get_train_config_dl(name=name, cv=0)

            def run(train_folds, test_fold, seed, suff):
                assert base_config is not None
                config = base_config.model_copy(deep=True)
                config.data_split_config.test_fold = test_fold
                config.data_split_config.train_folds = train_folds
                config.name += suff
                config.max_steps_to_run = 1200
                config.seed = seed
                config.train_balance_config.seed = seed
                run_training_process(config)

            for seed in range(4):
                train_folds = [f for f in range(5) if f != valid]
                for test in train_folds:
                    run(
                        train_folds=[f for f in train_folds if f != test],
                        test_fold=test,
                        seed=seed,
                        suff=f"-no-fold-{valid}-seed{seed}",
                    )
                run(
                    train_folds=train_folds,
                    test_fold=valid,
                    seed=seed,
                    suff=f"-no-fold-{valid}-seed{seed}-full",
                )


def get_oof_th(base_name: str, valid: int, seed: int, action: str, lab: str) -> float:
    ratio = get_median_ratio(
        base_name=base_name, valid=valid, seed=seed, action=action, lab=lab
    )
    pred_list = []
    gt_list = []
    for test in range(5):
        if test == valid:
            continue
        config = get_config(base_name=base_name, valid=valid, test=test, seed=seed)
        step = ratio_to_step(config=config, ratio=ratio)
        ckpt = Ckpt.from_config_and_step(config=config, step=step)

        pred_path = ckpt.preds_npy_path(action=action, lab=lab)
        if not pred_path.exists():
            continue
        pred_list.append(np.load(pred_path).astype(np.float32))
        gt_list.append(
            np.load(ckpt.gt_npy_path(action=action, lab=lab)).astype(np.int8)
        )

    assert pred_list
    assert gt_list

    pred = np.concatenate(pred_list, axis=0)
    gt = np.concatenate(gt_list, axis=0)
    return calc_best_f1_threshold(y_true=gt, y_pred=pred)


def get_action_lab_set(valid: int) -> set[tuple[str, str]]:
    ref_config = get_train_config_dl(name=REF_NAME, cv=valid)
    assert ref_config is not None
    ref_ckpt = list(iter_all_checkpoints_for_model_cv(ref_config.save_dir()))[0]
    return set(ref_ckpt.all_action_lab_from_metrics())


def print_fold_avg(
    base_name: str,
    valid: int,
    full_pred_per_action_lab: dict[tuple[str, str], np.ndarray] | None = None,
):
    action_lab_set = get_action_lab_set(valid=valid)

    SEEDS = 4

    valid_gt_per_action_lab = {}
    oof_th_per_action_lab = {}
    valid_preds_per_action_lab = defaultdict(list)
    print(f"Fold avg for {base_name}, valid: {valid}")
    for seed in range(SEEDS):
        f1_avg = ProdMetricCompHelper()
        pr_auc_avg = ProdMetricCompHelper()
        f1_with_full = ProdMetricCompHelper()
        pr_auc_with_full = ProdMetricCompHelper()
        for action, lab in action_lab_set:
            ratio = get_median_ratio(
                base_name=base_name, valid=valid, seed=seed, action=action, lab=lab
            )
            oof_th = get_oof_th(
                base_name=base_name, valid=valid, seed=seed, action=action, lab=lab
            )

            valid_pred_list = []
            valid_gt = None
            for test in range(5):
                if test == valid:
                    continue
                config = get_config(
                    base_name=base_name, valid=valid, test=test, seed=seed
                )
                step = ratio_to_step(config=config, ratio=ratio)
                ckpt = Ckpt.from_config_and_step(config=config, step=step)

                valid_pred = np.load(
                    ckpt.preds_npy_path(
                        action=action, lab=lab, suff=get_suff(valid=valid)
                    )
                ).astype(np.float32)
                valid_pred_list.append(valid_pred)

                if valid_gt is None:
                    valid_gt = np.load(
                        ckpt.gt_npy_path(
                            action=action, lab=lab, suff=get_suff(valid=valid)
                        )
                    ).astype(np.int8)

            assert valid_pred_list
            assert valid_gt is not None
            valid_pred = np.average(valid_pred_list, axis=0)

            if (valid_gt == 0).all():
                # print(f"Entire valid_gt is zero (action={action}, lab={lab}). Skip.")
                continue
            if (valid_gt == 1).all():
                # print(f"Entire valid_gt is one (action={action}, lab={lab}). Skip.")
                continue

            key = (action, lab)
            if seed == 0:
                oof_th_per_action_lab[key] = oof_th
                valid_gt_per_action_lab[key] = valid_gt
            valid_preds_per_action_lab[key].append(valid_pred)

            valid_pr_auc = calc_pr_auc(y_true=valid_gt, y_pred=valid_pred)
            valid_f1 = float(f1_score(y_true=valid_gt, y_pred=valid_pred >= oof_th))
            f1_avg.add(lab=lab, val=valid_f1)
            pr_auc_avg.add(lab=lab, val=valid_pr_auc)

            if full_pred_per_action_lab is not None:
                valid_pred_with_full = np.average(
                    [valid_pred, full_pred_per_action_lab[key]], axis=0
                )
                pr_auc = calc_pr_auc(y_true=valid_gt, y_pred=valid_pred_with_full)
                f1 = float(
                    f1_score(y_true=valid_gt, y_pred=valid_pred_with_full >= oof_th)
                )
                f1_with_full.add(lab=lab, val=f1)
                pr_auc_with_full.add(lab=lab, val=pr_auc)
        print(
            f"   Seed: {seed}. Fold average (4 folds): f1={f1_avg.calc():.5f}, pr_auc={pr_auc_avg.calc():.5f}."
        )
        print(
            f"      + full model: f1={f1_with_full.calc():.5f}, pr_auc={pr_auc_with_full.calc():.5f}"
        )

    f1_avg = ProdMetricCompHelper()
    pr_auc_avg = ProdMetricCompHelper()
    for action in ACTION_NAMES_IN_TEST:
        for lab in LABS_IN_TEST_PER_ACTION[action]:
            key = (action, lab)
            if key not in oof_th_per_action_lab:
                continue
            oof_th = oof_th_per_action_lab[key]
            gt = valid_gt_per_action_lab[key]
            pred = np.average(valid_preds_per_action_lab[key], axis=0)
            f1 = float(f1_score(y_true=gt, y_pred=pred >= oof_th))
            pr_auc = calc_pr_auc(y_true=gt, y_pred=pred)
            f1_avg.add(lab=lab, val=f1)
            pr_auc_avg.add(lab=lab, val=pr_auc)
    print(
        f"   All folds x All seeds: f1={f1_avg.calc():.5f}, pr_auc={pr_auc_avg.calc():.5f}"
    )


def print_seed_avg_full(
    base_name: str, valid: int
) -> dict[tuple[str, str], np.ndarray]:
    action_lab_set = get_action_lab_set(valid=valid)

    SEEDS = 4
    suff = get_suff(valid)

    f1_avg = ProdMetricCompHelper()
    pr_auc_avg = ProdMetricCompHelper()
    f1_per_seed = defaultdict(lambda: ProdMetricCompHelper())
    pr_auc_per_seed = defaultdict(lambda: ProdMetricCompHelper())

    pred_per_action_lab = {}
    for action, lab in action_lab_set:
        valid_pred_list = []
        valid_gt = None
        oof_th = get_oof_th(
            base_name=base_name, valid=valid, seed=0, action=action, lab=lab
        )

        for seed in range(SEEDS):
            ratio = get_median_ratio(
                base_name=base_name, valid=valid, seed=seed, action=action, lab=lab
            )
            config = get_config(base_name=base_name, valid=valid, test=valid, seed=seed)
            step = ratio_to_step(config=config, ratio=ratio)
            ckpt = Ckpt.from_config_and_step(config=config, step=step)

            valid_pred = np.load(
                ckpt.preds_npy_path(action=action, lab=lab, suff=suff)
            ).astype(np.float32)
            valid_pred_list.append(valid_pred)

            if valid_gt is None:
                valid_gt = np.load(
                    ckpt.gt_npy_path(action=action, lab=lab, suff=suff)
                ).astype(np.int8)

        assert valid_pred_list
        assert valid_gt is not None
        valid_pred = np.average(valid_pred_list, axis=0)

        if (valid_gt == 0).all():
            # print(f"Entire valid_gt is zero (action={action}, lab={lab}). Skip.")
            continue
        if (valid_gt == 1).all():
            # print(f"Entire valid_gt is one (action={action}, lab={lab}). Skip.")
            continue

        for seed, valid_pred_seed in zip(range(SEEDS), valid_pred_list):
            pr_auc = calc_pr_auc(y_true=valid_gt, y_pred=valid_pred_seed)
            f1 = float(f1_score(y_true=valid_gt, y_pred=valid_pred_seed >= oof_th))
            pr_auc_per_seed[seed].add(lab=lab, val=pr_auc)
            f1_per_seed[seed].add(lab=lab, val=f1)

        pred_per_action_lab[(action, lab)] = valid_pred

        pr_auc = calc_pr_auc(y_true=valid_gt, y_pred=valid_pred)
        f1 = float(f1_score(y_true=valid_gt, y_pred=valid_pred >= oof_th))
        pr_auc_avg.add(lab=lab, val=pr_auc)
        f1_avg.add(lab=lab, val=f1)
    print(f"Seed avg for {base_name}, valid: {valid}")
    print(f"   ({SEEDS} seeds): f1={f1_avg.calc():.5f}, pr_auc={pr_auc_avg.calc():.5f}")
    for seed in range(SEEDS):
        print(
            f"   One full model (seed {seed}): f1={f1_per_seed[seed].calc():.5f}, pr_auc={pr_auc_per_seed[seed].calc():.5f}"
        )
    return pred_per_action_lab


# run_training()

for name in names:
    for valid in range(2):
        pred_per_action_lab = print_seed_avg_full(base_name=name, valid=valid)
        print_fold_avg(
            base_name=name, valid=valid, full_pred_per_action_lab=pred_per_action_lab
        )
        print()

# meta = get_train_meta()
# feats_lookup = FeaturesLookupGPU(
#     meta=meta, feats_config=FeaturesConfigDL(), aug_config=AugmentationsConfig()
# )
# for base_name in names:
#     for valid in range(2):
#         for seed in range(4):
#             for test in range(5):
#                 lazy_generate_valid_preds(
#                     base_name=base_name,
#                     valid=valid,
#                     seed=seed,
#                     test=test,
#                     feats_lookup=feats_lookup,
#                 )
