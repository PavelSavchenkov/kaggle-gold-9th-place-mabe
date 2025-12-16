import dataclasses
import json
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from common.config_utils import base_model_from_dict, base_model_from_file
from common.constants import (
    ACTION_NAMES_IN_TEST,
    LABS_IN_TEST_PER_ACTION,
    VIDEO_CATEGORICAL_FEATURES,
)
from common.ensemble_building_primitives import (
    EnsembleObjective,
    build_ensembles_beam_search,
)
from common.metrics_common import DatasetScore, OOF_Metrics
from common.paths import MODELS_ROOT
from dl.ensembling_dl import EnsembleApproachDL, build_ensemble_map_per_approach
from dl.gbdt_dl_stype import GBDT_DL_Style
from dl.metrics import ProdMetricCompHelper
from dl.oof_utils import (
    OOF_Data,
    OOF_Full,
    get_median_ratio_by_pr_auc,
    get_oof_data,
    read_oof_map,
)
from dl.postprocess import (
    Ckpt,
    get_action_lab_set_from_config_dl,
    get_actions_set_from_config_dl,
    get_best_pr_auc_ckpt_for_each_action,
    get_best_pr_auc_ckpt_for_each_action_lab,
    get_train_config_dl,
    iter_all_configs_for_model_name_dl,
    iter_all_dl_model_names,
    ratio_to_step,
)
from dl.submission import Submission

MAX_WORKERS = 8
submission = base_model_from_file(
    Submission, "submissions/dl-tcn-median-6-dec/submission.json"
)
names = list(sorted(set([model.name() for model in submission.models])))
# take_filter_dl = lambda name: len(list(iter_all_configs_for_model_name_dl(name))) == 5 and "adap-snail" in name
# names = list(iter_all_dl_model_names(take_filter=take_filter_dl))
print(f"# DL names: {len(names)}")
print(names)
# print(names)

REF_NAME = names[-1]

# JSON_PATH = Path("e_apps_dl_6.json")
# JSON_PATH = Path("e_apps_dl_6_logits.json")
# JSON_PATH = Path("e_apps_dl_9.json")
# JSON_PATH = Path("e_apps_dl_gbdt.json")
JSON_PATH = Path("e_apps_dl_gbdt_14_dec_old_code.json")
# JSON_PATH = Path("e_apps_dl_gbdt_nested_cnt.json")
# JSON_PATH = Path("e_apps_dl_plus_no_adap_snail_25fps_gbdt.json")
# JSON_PATH = Path("e_apps_dl_only_no_adap_snail_25fps_gbdt.json")
# JSON_PATH = Path("e_apps_only_gbdt.json")

ONLY_GBDT = False


@lru_cache
def get_action_lab_list(valid: int):
    REF_CONFIG = get_train_config_dl(name=REF_NAME, cv=valid)
    assert REF_CONFIG is not None
    res = list(sorted(get_action_lab_set_from_config_dl(REF_CONFIG)))
    # res = res[:4]
    return res


@lru_cache
def get_action_list(valid: int):
    REF_CONFIG = get_train_config_dl(name=REF_NAME, cv=valid)
    assert REF_CONFIG is not None
    res = list(sorted(get_actions_set_from_config_dl(REF_CONFIG)))
    # res = res[:4]
    return res


gbdt_by_name: dict[str, GBDT_DL_Style] = {}


def get_oof_data_foo(t):
    valid, name, action, lab = t
    train_folds = [f for f in range(5) if f != valid]
    if not name.startswith("gbdt"):
        return get_oof_data(
            name=name,
            action=action,
            lab=lab,
            median_ckpt=True,
            folds=train_folds,
            verbose=False,
            float_dtype=np.float16,
        )
    else:
        step = gbdt_by_name[name].get_median_step(
            action=action, lab=lab, folds=train_folds
        )
        assert step in gbdt_by_name[name].steps
        steps = [step] * len(train_folds)
        return gbdt_by_name[name].get_oof_data(
            action=action,
            lab=lab,
            folds=train_folds,
            steps=steps,
            float_dtype=np.float16,
        )


# def build_json_per_action():
#     approaches = [
#         EnsembleApproachDL(objective=EnsembleObjective.nested_f1, max_models=6),
#         EnsembleApproachDL(objective=EnsembleObjective.logloss, max_models=6),
#     ]
#     # valid_list = [0]
#     valid_list = list(range(5))

#     # ----- OOF data -----

#     oof_params_list = []
#     for valid in valid_list:
#         train_folds = [f for f in range(5) if f != valid]
#         action_list = get_action_list(valid=valid)
#         for action in action_list:
#             for name in names:
#                 oof_params_list.append((valid, name, action, None))
#     oof_data_map = {}
#     with ProcessPoolExecutor(max_workers=10) as ex:
#         for oof_params, oof_data in tqdm(
#             zip(oof_params_list, ex.map(get_oof_data_foo, oof_params_list)),
#             total=len(oof_params_list),
#             desc="Reading oof data",
#         ):
#             oof_data_map[oof_params] = oof_data

#     def push_to_results_json(**kwargs):
#         item = kwargs
#         if not JSON_PATH.exists():
#             js = []
#         else:
#             js = json.load(open(JSON_PATH))
#         js.append(item)
#         json.dump(js, open(JSON_PATH, "w"))

#     for valid in valid_list:
#         train_folds = [f for f in range(5) if f != valid]
#         action_list = get_action_list(valid=valid)
#         for action in action_list:
#             ref_name = names[0]
#             ref_config = get_train_config_dl(name=ref_name, cv=valid)
#             assert ref_config
#             ref_best_map = get_best_pr_auc_ckpt_for_each_action(
#                 save_dir=ref_config.save_dir()
#             )
#             ref_pr_auc, _ = ref_best_map[action]
#             print(
#                 f"Doing valid={valid}, action={action}. Reference pr-auc: {ref_pr_auc:.5f} ..."
#             )

#             y_pred_list = []
#             y_true = None
#             fold_id = None
#             y_pred_list_valid = []
#             y_true_valid = None
#             lab_idx_mask = None
#             for name in names:
#                 oof_data = oof_data_map[(valid, name, action, None)]
#                 mask_train = oof_data.fold_id != valid
#                 train_fold_id = oof_data.fold_id[mask_train]
#                 train_y_pred = oof_data.y_pred[mask_train]
#                 train_y_true = oof_data.y_true[mask_train]
#                 y_pred_list.append(train_y_pred)
#                 if y_true is None:
#                     y_true = train_y_true
#                 else:
#                     assert (y_true == train_y_true).all()
#                 if fold_id is None:
#                     fold_id = train_fold_id
#                 else:
#                     assert (fold_id == train_fold_id).all()

#                 ratio = get_median_ratio_by_pr_auc(
#                     name=name, action=action, lab=None, folds=train_folds
#                 )
#                 config = get_train_config_dl(name=name, cv=valid)
#                 assert config is not None
#                 step = ratio_to_step(config=config, ratio=ratio)
#                 ckpt = Ckpt.from_config_and_step(config=config, step=step)
#                 cur_y_pred_valid = ckpt.get_pred_for_action(action=action)
#                 cur_y_true_valid, cur_lab_idx_mask = (
#                     ckpt.get_gt_for_action_with_lab_idx_mask(action=action)
#                 )
#                 y_pred_list_valid.append(cur_y_pred_valid)
#                 if y_true_valid is None:
#                     y_true_valid = cur_y_true_valid
#                 else:
#                     assert (y_true_valid == cur_y_true_valid).all()
#                 if lab_idx_mask is None:
#                     lab_idx_mask = cur_lab_idx_mask
#                 else:
#                     assert (lab_idx_mask == cur_lab_idx_mask).all()

#             assert fold_id is not None
#             assert y_true is not None
#             assert y_true.size > 0
#             assert y_pred_list
#             assert y_true_valid is not None
#             assert y_true_valid.size > 0
#             assert y_pred_list_valid
#             assert lab_idx_mask is not None

#             if (y_true == 0).all() or (y_true == 1).all():
#                 print(f"y_true is degenerate. Skip.")
#                 continue

#             if (y_true_valid == 0).all() or (y_true_valid == 1).all():
#                 print(f"y_true_valid is degenerate. Skip.")
#                 continue

#             es = build_ensemble_map_per_approach(
#                 names=names,
#                 y_true=y_true,
#                 y_pred_list=y_pred_list,
#                 fold_id=fold_id,
#                 approaches=approaches,
#             )

#             print()
#             print(f"===== Results for valid={valid}, action={action} =====")
#             lab_idx_list = list(sorted(np.unique(lab_idx_mask)))
#             for app, e in es.items():
#                 pred_all_labs = e.infer(names=names, preds=y_pred_list_valid)
#                 for lab_idx in lab_idx_list:
#                     cur_lab_mask = lab_idx_mask == lab_idx
#                     lab = VIDEO_CATEGORICAL_FEATURES["lab_id"][lab_idx]
#                     f1 = float(
#                         f1_score(
#                             y_true=y_true_valid[cur_lab_mask],
#                             y_pred=pred_all_labs[cur_lab_mask] >= e.th,
#                         )
#                     )
#                     print(f"    app: {app.to_str()}, f1: {f1:.5f}")
#                     push_to_results_json(
#                         valid=valid,
#                         action=action,
#                         lab=lab,
#                         app=app.model_dump(mode="json"),
#                         f1=f1,
#                     )
#             print(f"=================================================================")
#             print()


def build_json(include_gbdt: bool = False):
    if ONLY_GBDT:
        global names
        names = names[:1]
    if include_gbdt:
        for gbdt_path in MODELS_ROOT.rglob("gbdt-*"):
            name = gbdt_path.name
            names.append(name)
            gbdt_by_name[name] = GBDT_DL_Style(model_path=gbdt_path)

    approaches = [
        EnsembleApproachDL(
            objective=EnsembleObjective.nested_f1,
            max_models=12,
            allow_neg=False,
            num_bins=10,
            pool_size=1,
            logits=False,
        ),
        # EnsembleApproachDL(
        #     objective=EnsembleObjective.nested_f1,
        #     max_models=6,
        #     allow_neg=False,
        #     num_bins=10,
        #     pool_size=1,
        #     logits=True,
        # ),
        # EnsembleApproachDL(
        #     objective=EnsembleObjective.nested_f1,
        #     max_models=6,
        #     allow_neg=True,
        #     num_bins=10,
        #     pool_size=1,
        #     logits=True,
        # ),
    ]
    # valid_list = [0]
    valid_list = list(range(5))

    # ----- OOF data -----

    oof_params_list = []
    for valid in valid_list:
        train_folds = [f for f in range(5) if f != valid]
        action_lab_set = get_action_lab_list(valid=valid)
        for action, lab in action_lab_set:
            for name in names:
                if name.startswith("gbdt"):
                    if (action, lab) not in gbdt_by_name[name].get_action_lab_set():
                        continue
                oof_params_list.append((valid, name, action, lab))
    oof_data_map = {}
    oof_msg = f"Reading oof data, #names {len(names)}, #workers {MAX_WORKERS}"
    if MAX_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for oof_params, oof_data in tqdm(
                zip(oof_params_list, ex.map(get_oof_data_foo, oof_params_list)),
                total=len(oof_params_list),
                desc=oof_msg,
            ):
                oof_data_map[oof_params] = oof_data
    else:
        for oof_params in tqdm(oof_params_list, desc=oof_msg):
            oof_data_map[oof_params] = get_oof_data_foo(oof_params)

    def push_to_results_json(**kwargs):
        item = kwargs
        if not JSON_PATH.exists():
            js = []
        else:
            js = json.load(open(JSON_PATH))
        js.append(item)
        json.dump(js, open(JSON_PATH, "w"))

    total_used_by_model_type = defaultdict(int)
    for valid in valid_list:
        train_folds = [f for f in range(5) if f != valid]
        action_lab_set = get_action_lab_list(valid=valid)
        for action, lab in action_lab_set:
            ref_name = names[0]
            ref_config = get_train_config_dl(name=ref_name, cv=valid)
            assert ref_config
            ref_best_map = get_best_pr_auc_ckpt_for_each_action_lab(
                save_dir=ref_config.save_dir()
            )
            ref_pr_auc, _ = ref_best_map[(action, lab)]
            print(
                f"Doing valid={valid}, action={action}, lab={lab}. Reference pr-auc: {ref_pr_auc:.5f} ..."
            )

            y_pred_list = []
            y_true = None
            fold_id = None
            y_pred_list_valid = []
            y_true_valid = None
            relevant_names = []
            for name in names:
                is_gbdt = name.startswith("gbdt-")
                oof_key = (valid, name, action, lab)
                if oof_key not in oof_data_map:
                    assert is_gbdt, f"name: {name}, lab: {lab}, action: {action}"
                    continue
                oof_data = dataclasses.replace(oof_data_map[oof_key])
                oof_data.y_pred = oof_data.y_pred.astype(np.float32)
                mask_train = oof_data.fold_id != valid
                train_fold_id = oof_data.fold_id[mask_train]
                train_y_pred = oof_data.y_pred[mask_train]
                if fold_id is None:
                    assert not is_gbdt, f"name: {name}, lab: {lab}, action: {action}"
                    fold_id = train_fold_id
                else:
                    if is_gbdt:
                        if fold_id.shape != train_fold_id.shape:
                            print(
                                f"[WARNING] Weird fold_id in {name}, lab: {lab}, action: {action}. Skip."
                            )
                            continue
                    assert (fold_id == train_fold_id).all()

                y_pred_list.append(train_y_pred)

                if not name.startswith("gbdt-"):
                    train_y_true = oof_data.y_true[mask_train]
                    if y_true is None:
                        y_true = train_y_true
                    else:
                        assert (y_true == train_y_true).all()
                    ratio = get_median_ratio_by_pr_auc(
                        name=name, action=action, lab=lab, folds=train_folds
                    )
                    config = get_train_config_dl(name=name, cv=valid)
                    assert config is not None
                    step = ratio_to_step(config=config, ratio=ratio)
                    ckpt = Ckpt.from_config_and_step(config=config, step=step)
                    cur_y_pred_valid = np.load(
                        ckpt.preds_npy_path(action=action, lab=lab)
                    ).astype(np.float32)
                    cur_y_true_valid = np.load(
                        ckpt.gt_npy_path(action=action, lab=lab)
                    ).astype(np.int8)
                    y_pred_list_valid.append(cur_y_pred_valid)
                    if y_true_valid is None:
                        y_true_valid = cur_y_true_valid
                    else:
                        assert (y_true_valid == cur_y_true_valid).all()
                else:
                    step = gbdt_by_name[name].get_median_step(
                        action=action, lab=lab, folds=train_folds
                    )
                    cur_y_pred_valid = gbdt_by_name[name].get_pred(
                        action=action, lab=lab, fold=valid, step=step
                    )
                    y_pred_list_valid.append(cur_y_pred_valid)
                relevant_names.append(name)

            assert fold_id is not None
            assert y_true is not None
            assert y_true.size > 0
            assert y_pred_list
            assert y_true_valid is not None
            assert y_true_valid.size > 0
            assert y_pred_list_valid
            assert relevant_names

            if (y_true == 0).all() or (y_true == 1).all():
                print(f"y_true is degenerate. Skip.")
                continue

            if (y_true_valid == 0).all() or (y_true_valid == 1).all():
                print(f"y_true_valid is degenerate. Skip.")
                continue

            if ONLY_GBDT and len(relevant_names) > 1:
                relevant_names = relevant_names[1:]
                y_pred_list = y_pred_list[1:]
                y_pred_list_valid = y_pred_list_valid[1:]

            es = build_ensemble_map_per_approach(
                names=relevant_names,
                y_true=y_true,
                y_pred_list=y_pred_list,
                fold_id=fold_id,
                approaches=approaches,
            )

            print()
            print(f"===== Results for valid={valid}, action={action}, lab={lab} =====")
            cur_used_by_model_type = defaultdict(int)
            for app, e in es.items():
                for used_name in e.names:
                    if used_name.startswith("gbdt-"):
                        cur_used_by_model_type["gbdt"] += 1
                    else:
                        cur_used_by_model_type["dl"] += 1
                y_pred_valid = e.infer(names=relevant_names, preds=y_pred_list_valid)
                f1 = float(f1_score(y_true=y_true_valid, y_pred=y_pred_valid >= e.th))
                print(f"    app: {app.to_str()}, f1: {f1:.5f}")
                push_to_results_json(
                    valid=valid,
                    action=action,
                    lab=lab,
                    app=app.model_dump(mode="json"),
                    f1=f1,
                )
            print(f"=================================================================")
            for k, v in cur_used_by_model_type.items():
                total_used_by_model_type[k] += v
            print(f"Totals: ")
            for k, v in sorted(total_used_by_model_type.items()):
                print(f"    {k:4}: {v} times")
            print(f"Now: ")
            for k, v in sorted(cur_used_by_model_type.items()):
                print(f"    {k:4}: {v} times")
            print()


def json_app_to_str(js: dict) -> str:
    app = base_model_from_dict(EnsembleApproachDL, js)
    return app.to_str()


def analyse_json():
    # LAB_REMOVE = "AdaptableSnail"
    LAB_REMOVE = None
    data = json.load(open(JSON_PATH))
    for item in data:
        item["app"] = json_app_to_str(item["app"])
    data = [item for item in data if "logloss" not in item["app"]]
    data = [item for item in data if item["lab"] != LAB_REMOVE]
    apps = set()
    for item in data:
        apps.add(item["app"])
    apps = list(sorted(apps))
    for app in apps:
        print(f"'{app}' everythere:")
        f1_list = []
        for valid in range(5):
            f1_avg = ProdMetricCompHelper()
            for item in data:
                if item["app"] == app and item["valid"] == valid:
                    f1_avg.add(lab=item["lab"], val=item["f1"])
            f1 = f1_avg.calc()
            print(f"{f1:.5f} ", end="")
            f1_list.append(f1)
        avg = float(np.mean(f1_list))
        print(f"  ---> {avg:.5f}")
        print()

    f1_list_per_action_lab = defaultdict(lambda: defaultdict(lambda: [np.nan] * 5))
    for item in data:
        f1_list_per_action_lab[(item["action"], item["lab"])][item["app"]][
            item["valid"]
        ] = item["f1"]

    def get_best_app_for(action: str, lab: str, leave_one_fold: int) -> str:
        best_app = None
        best_f1 = -1
        for app, f1_list in f1_list_per_action_lab[(action, lab)].items():
            assert len(f1_list) == 5
            f1_list_filtered = [f1_list[i] for i in range(5) if i != leave_one_fold]
            f1 = float(np.nanmean(f1_list_filtered))
            if f1 > best_f1:
                best_f1 = f1
                best_app = app
        assert best_app is not None
        return best_app

    print()
    print(f"===== Select best app on 4 folds, validate on 5th =====")
    f1_avg_list = []
    for leave_one_fold in range(5):
        f1_avg = ProdMetricCompHelper()
        for action, lab in f1_list_per_action_lab.keys():
            app = get_best_app_for(
                action=action, lab=lab, leave_one_fold=leave_one_fold
            )
            f1 = f1_list_per_action_lab[(action, lab)][app][leave_one_fold]
            if np.isnan(f1):
                continue
            f1_avg.add(lab=lab, val=f1)
        f1 = f1_avg.calc()
        f1_avg_list.append(f1)
        print(f"{f1:.5f} ", end="")
    print(f"  --->  {float(np.mean(f1_avg_list)):.5f}")
    print()

    return

    print(f"===== BREAKDOWN PER ACTION-LAB-APP =====")
    max_diff_per_action_lab = {}
    for key in f1_list_per_action_lab.keys():
        per_app = []
        for app, per_fold in f1_list_per_action_lab[key].items():
            per_app.append(float(np.mean(per_fold)))
        max_diff_per_action_lab[key] = np.max(per_app) - np.min(per_app)
    action_lab_order = list(f1_list_per_action_lab.keys())
    action_lab_order.sort(key=lambda k: max_diff_per_action_lab[k])
    for action, lab in action_lab_order:
        app_map = f1_list_per_action_lab[(action, lab)]
        print()
        print(f"----- action: {action}, lab: {lab} -----")
        for app, f1_list in sorted(app_map.items()):
            print(f"    {app:30} [ ", end="")
            for f1 in f1_list:
                print(f"{f1:.5f} ", end="")
            print("]", end="")
            avg = float(np.nanmean(f1_list))
            print(f" ---> {avg:.5f}")
        print("---------------------------------------------")
        print()


# build_json_per_action()
# build_json(include_gbdt=True)
analyse_json()
