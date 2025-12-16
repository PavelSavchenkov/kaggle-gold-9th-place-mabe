from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.config_utils import base_model_from_file
from common.prod_score import SubmissionScore, score_map
from postprocess.submission_utils import Submission

PREDS_DIR = Path("preds_csv")


def print_maps_side_by_side(
    names: list[str], labs: list[str], maps: dict[str, SubmissionScore]
):
    for lab in labs:
        print()
        print(f"Lab {lab}")
        print("=" * 80)

        # Collect all actions appearing in this lab for any submission
        actions = sorted(
            set().union(*(maps[name].per_lab[lab].per_action.keys() for name in names))
        )

        # Header: action + each submission name as a column
        header = f"{'action':16}"
        for name in names:
            header += f"{name:20}"
        print(header)
        print("-" * len(header))

        # One row per action, with F1 for each submission
        for action in actions:
            row = f"{action:16}"
            for name in names:
                f1 = maps[name].per_lab[lab].per_action.get(action, float("nan"))
                row += f"{f1:12.5f}            "
            print(row)

        # Separator and per-lab F1 (summary) across submissions
        print("-" * len(header))
        summary_row = f"{'[lab F1]':16}"
        for name in names:
            summary_row += f"{maps[name].per_lab[lab].f1:12.5f}            "
        print(summary_row)
        print("\n")


def main(
    dst_per_lab: Path | str | None = None, dst_per_action: Path | str | None = None
):
    if dst_per_lab is not None:
        dst_per_lab = Path(dst_per_lab)
    if dst_per_action is not None:
        dst_per_action = Path(dst_per_action)

    # names = ["e-nested-f1", "e-pr-auc-optim", "e-pr-auc", "e-logloss"]
    # names = ["e-nested-f1", "e-pr-auc", "e-nested-f1-all"]
    # names = ["e-nested-f1", "e-pr-auc-optim", "all-sanity-check"]
    # names = ["e-nested-f1-all", "e-nested-f1-23-nov"]
    # names = ["e-nested-f1-23-nov", "dl-tcn-4-dec"]
    # names = ["all-best-21-nov", "dl-tcn-4-dec"]
    # names = ["all-best-21-nov", "dl-tcn-4-dec-smooth-13"]
    # names = ["e-nested-f1-23-nov", "dl-tcn-4-dec-smooth-13"]
    names = ["dl-tcn-ensemble6-9-dec", "e-nested-f1-all"]
    # submissions = {}
    csvs = {}
    for name in names:
        # path = Path("submissions") / name
        # assert path.exists()
        path_csv = PREDS_DIR / f"out_{name}.csv"
        if not path_csv.exists():
            print(f"No csv for {name}")
            continue
        csv = pd.read_csv(path_csv)
        csvs[name] = csv
        # submissions[name] = base_model_from_file(Submission, path / "submission.json")
    names.sort()

    gt = pd.read_csv("data/test_gt.csv")

    maps = {}
    for name in names:
        maps[name] = score_map(solution=gt, submission=csvs[name])

    labs = list(sorted(set(maps[names[0]].per_lab.keys())))
    for name in names:
        assert set(labs) == set(maps[name].per_lab.keys())

    print_maps_side_by_side(names=names, labs=labs, maps=maps)

    BEST_MAP = defaultdict(list)

    # per_lab = Submission()
    per_lab_f1 = []
    for lab in labs:
        best_name = ""
        best_f1 = 0.0
        actions = list(maps[names[0]].per_lab[lab].per_action.keys())
        for name in names:
            per_action = []
            for action in actions:
                per_action.append(maps[name].per_lab[lab].per_action[action])
            per_action = float(np.mean(per_action))
            if per_action > best_f1:
                best_f1 = per_action
                best_name = name
        # for action in actions:
        #     per_lab.fill_from(lab=lab, action=action, other=submissions[best_name])
        per_lab_f1.append(best_f1)
    # per_lab.sanity_check_labs_actions_present()
    per_lab_f1 = float(np.mean(per_lab_f1))

    # per_action = Submission()
    per_action_f1 = []
    for lab in labs:
        actions = list(maps[names[0]].per_lab[lab].per_action.keys())
        f1s = []
        for action in actions:
            best_name = ""
            best_f1 = 0.0
            for name in names:
                f1 = maps[name].per_lab[lab].per_action[action]
                if f1 > best_f1:
                    best_f1 = f1
                    best_name = name
            BEST_MAP[best_name].append([action, lab])
            f1s.append(best_f1)
            # per_action.fill_from(lab=lab, action=action, other=submissions[best_name])
        per_action_f1.append(float(np.mean(f1s)))
    # per_action.sanity_check_labs_actions_present()
    per_action_f1 = float(np.mean(per_action_f1))

    json.dump(BEST_MAP, open("best_map.json", "w"))
    json.dump(BEST_MAP[names[0]], open("gbdt_limit_to.json", "w"))
    json.dump(BEST_MAP[names[1]], open("dl_limit_to.json", "w"))

    print(f"Originals:")
    for name in names:
        f1 = maps[name].f1
        print(f"{name:23}: {f1:.5f}")

    # if dst_per_lab is not None:
    #     per_lab.write_dir(dst_per_lab)
    #     json.dump({"names": names}, open(dst_per_lab / "names.json", "w"))
    print(f"Best per lab           : {per_lab_f1:.5f}")
    # if dst_per_action is not None:
    #     per_action.write_dir(dst_per_action)
    #     json.dump({"names": names}, open(dst_per_action / "names.json", "w"))
    print(f"Best per lab per action: {per_action_f1:.5f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--per-lab", default=None)
    parser.add_argument("--per-action", default=None)
    args = parser.parse_args()
    main(dst_per_lab=args.per_lab, dst_per_action=args.per_action)
