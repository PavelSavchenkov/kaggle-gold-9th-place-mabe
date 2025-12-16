import copy
import json
import pprint
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from common.ensemble_building_primitives import EnsembleObjective
from common.constants import ACTION_NAMES_IN_TEST
from gbdt.helpers import (
    get_any_train_config,
    get_best_pr_auc_step,
    get_pred_test_np,
    get_test_ground_truth_np,
    get_test_index_df,
    read_metrics,
)
from postprocess.ensemble_utils import EnsembleApproach
from postprocess.postprocess_utils import (
    top_models_by_oof_per_action,
    top_models_by_oof_per_action_per_lab,
    try_calc_oof_metrics,
)
from postprocess.submission_utils import Submission

# names = []
# for model_path in Path("train_logs").iterdir():
#     name = model_path.name
#     names.append(name)

# with ThreadPoolExecutor(max_workers=8) as ex:
#     def process(name):
#         try_calc_oof_metrics(name=name, rewrite=True, verbose=True)
#     for _ in tqdm(ex.map(process, names), total=len(names)):
#         pass
# exit(0)


def run_for_action(ACTION: str):
    NUM_FOLDS = 5
    VALID_LIST = [0, 1] #list(range(NUM_FOLDS))
    VERBOSE = False

    cache_path = Path(f"approaches/{ACTION}.json")
    cache_path.parent.mkdir(exist_ok=True, parents=True)

    if cache_path.exists():
        return

    approaches = []
    for obj in [EnsembleObjective.nested_f1, EnsembleObjective.logloss]:
        approaches.append(
            EnsembleApproach(
                max_models=3,
                filter_same_model_type=False,
                objective=obj,
                num_weight_bins=10,
                pool_size=1,
                avg_k_best=False,
                refit_with_fold_avg=False,
            )
        )

    # approaches = approaches[:3]
    print(f"Cnt approaches: {len(approaches)}")

    def take_filter(name: str):
        config = get_any_train_config(name)
        if not config:
            return False
        return config.action == ACTION

    submission_maps_per_valid = {}
    for valid in VALID_LIST:
        print(f"[VALID={valid}] Building all submissions (action={ACTION})")
        train_folds = [fold for fold in range(NUM_FOLDS) if fold != valid]
        models_dict = top_models_by_oof_per_action_per_lab(
            return_list=True, folds=train_folds, take_filter=take_filter
        )
        # models_dict = top_models_by_oof_per_action(
        #     return_list=True, folds=train_folds, take_filter=take_filter
        # )
        submission_maps_per_valid[valid] = (
            Submission.build_ensembles_per_approach(
                models_dict=models_dict,
                e_approaches=approaches,
                verbose=VERBOSE,
            )
        )

    cache = {"maps": {}, "f1_valid": {}, "f1_train": {}}

    for app in approaches:
        print(f"[APPROACH INFER] {app.to_str()}")
        f1_map = {}  # lab -> action -> list[float]
        for valid in VALID_LIST:
            submission = submission_maps_per_valid[valid][app]
            pred_per_action_lab = {}
            true_per_action_lab = {}
            for model in submission.models:
                action = model.action
                lab = model.lab
                metrics = read_metrics(model.name, valid)
                try:
                    step = get_best_pr_auc_step(metrics, lab=lab)
                except Exception as ex:
                    print(f"[WARNING] ex in finding best_pr_auc_step: {ex}")
                    continue
                y_pred = get_pred_test_np(model.name, valid, step)
                index = get_test_index_df(model.name, valid, usecols=("lab_id",))
                mask_lab = (index.lab_id == lab).to_numpy()
                y_pred = y_pred[mask_lab]
                y_pred = y_pred * model.coef

                key = (action, lab)
                if key not in true_per_action_lab:
                    y_true = get_test_ground_truth_np(model.name, valid)
                    true_per_action_lab[key] = y_true[mask_lab]

                if key not in pred_per_action_lab:
                    pred_per_action_lab[key] = y_pred
                else:
                    pred_per_action_lab[key] += y_pred

            per_lab = defaultdict(list)
            for (action, lab), pred in pred_per_action_lab.items():
                th = submission.get_threshold(lab=lab, action=action)
                true = true_per_action_lab[(action, lab)]
                f1 = float(f1_score(y_true=true, y_pred=pred >= th))
                if lab not in f1_map:
                    f1_map[lab] = defaultdict(list)
                f1_map[lab][action].append(f1)
                per_lab[lab].append(f1)

            # f1 for this valid fold
            f1_list = []
            for lab in per_lab.keys():
                f1 = float(np.mean(per_lab[lab]))
                f1_list.append(f1)
            f1 = float(np.mean(f1_list))
            print(f"  [valid fold score] (={valid}): {f1:.5f}")

        f1 = []
        for lab in f1_map.keys():
            f1_inner = []
            for action in f1_map[lab].keys():
                f1_over_folds = f1_map[lab][action]
                f1_inner.append(float(np.mean(f1_over_folds)))
            f1_inner = float(np.mean(f1_inner))
            f1.append(f1_inner)
        f1 = float(np.mean(f1))

        train_score = []
        for valid in VALID_LIST:
            train_score.append(
                submission_maps_per_valid[valid][app].dataset_score.totals["nested_f1"]
            )
        train_score = float(np.mean(train_score))

        cache_key = app.to_str()
        cache["maps"][cache_key] = f1_map
        cache["f1_valid"][cache_key] = f1
        cache["f1_train"][cache_key] = train_score

        print(
            f"[APPROACH SCORE] {app.to_str()}:\ntrain={train_score:.5f}, valid={f1:.5f}"
        )
        print("-----------------------------------------------------")

    json.dump(cache, open(cache_path, "w"))

from postprocess.submission_utils import best_ensemble_approach_for_action
actions = []
for action in ACTION_NAMES_IN_TEST:
    try:
        best_ensemble_approach_for_action(action)
    except:
        actions.append(action)

print(actions)
for action in actions:
    run_for_action(action)
