"""F Beta customized for the data format of the MABe challenge."""

import json
from collections import defaultdict

import pandas as pd
import polars as pl
from pydantic import BaseModel, Field
from sympy import O

PRINT_ACTIONS = False


class SubmissionScoreLab(BaseModel):
    f1: float = -1.0
    per_action: dict[str, float] = Field(default_factory=dict)


class SubmissionScore(BaseModel):
    f1: float = -1.0
    per_lab: dict[str, SubmissionScoreLab] = Field(default_factory=dict)


class HostVisibleError(Exception):
    pass


LAB = "UppityFerret"


def single_lab_f1(
    lab_solution: pl.DataFrame, lab_submission: pl.DataFrame, beta: float = 1
) -> SubmissionScoreLab:
    label_frames: defaultdict[str, set[int]] = defaultdict(set)
    prediction_frames: defaultdict[str, set[int]] = defaultdict(set)

    for row in lab_solution.to_dicts():
        label_frames[row["label_key"]].update(
            range(row["start_frame"], row["stop_frame"])
        )

    for video in lab_solution["video_id"].unique():
        active_labels: str = lab_solution.filter(pl.col("video_id") == video)[
            "behaviors_labeled"
        ].first()  # ty: ignore
        active_labels: set[str] = set(json.loads(active_labels))
        predicted_mouse_pairs: defaultdict[str, set[int]] = defaultdict(set)

        for row in lab_submission.filter(pl.col("video_id") == video).to_dicts():
            # Since the labels are sparse, we can't evaluate prediction keys not in the active labels.
            if (
                ",".join([str(row["agent_id"]), str(row["target_id"]), row["action"]])
                not in active_labels
            ):
                continue

            new_frames = set(range(row["start_frame"], row["stop_frame"]))
            # Ignore truly redundant predictions.
            new_frames = new_frames.difference(prediction_frames[row["prediction_key"]])
            prediction_pair = ",".join([str(row["agent_id"]), str(row["target_id"])])
            if predicted_mouse_pairs[prediction_pair].intersection(new_frames):
                # A single agent can have multiple targets per frame (ex: evading all other mice) but only one action per target per frame.
                raise HostVisibleError(
                    "Multiple predictions for the same frame from one agent/target pair"
                )
            prediction_frames[row["prediction_key"]].update(new_frames)
            predicted_mouse_pairs[prediction_pair].update(new_frames)

    tps = defaultdict(int)
    fns = defaultdict(int)
    fps = defaultdict(int)
    for key, pred_frames in prediction_frames.items():
        action = key.split("_")[-1]
        matched_label_frames = label_frames[key]
        tps[action] += len(pred_frames.intersection(matched_label_frames))
        fns[action] += len(matched_label_frames.difference(pred_frames))
        fps[action] += len(pred_frames.difference(matched_label_frames))

    distinct_actions = set()
    for key, frames in label_frames.items():
        action = key.split("_")[-1]
        distinct_actions.add(action)
        if key not in prediction_frames:
            fns[action] += len(frames)

    res = SubmissionScoreLab()
    action_f1s = []
    for action in sorted(distinct_actions):
        f1 = 0.0
        if tps[action] + fns[action] + fps[action] > 0:
            f1 = (
                (1 + beta**2)
                * tps[action]
                / ((1 + beta**2) * tps[action] + beta**2 * fns[action] + fps[action])
            )
        action_f1s.append(f1)
        res.per_action[action] = f1
    res.f1 = sum(action_f1s) / len(action_f1s)
    return res


def mouse_fbeta(
    solution: pd.DataFrame, submission: pd.DataFrame, beta: float = 1
) -> SubmissionScore:
    """
    Doctests:
    >>> solution = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 10, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 10},
    ... ])
    >>> mouse_fbeta(solution, submission)
    1.0

    >>> solution = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 10, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'mount', 'start_frame': 0, 'stop_frame': 10}, # Wrong action
    ... ])
    >>> mouse_fbeta(solution, submission)
    0.0

    >>> solution = pd.DataFrame([
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 9, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'mount', 'start_frame': 15, 'stop_frame': 24, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 9},
    ... ])
    >>> "%.12f" % mouse_fbeta(solution, submission)
    '0.500000000000'

    >>> solution = pd.DataFrame([
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 9, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'mount', 'start_frame': 15, 'stop_frame': 24, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 345, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 9, 'lab_id': 2, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 345, 'agent_id': 1, 'target_id': 2, 'action': 'mount', 'start_frame': 15, 'stop_frame': 24, 'lab_id': 2, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 9},
    ... ])
    >>> "%.12f" % mouse_fbeta(solution, submission)
    '0.250000000000'

    >>> # Overlapping solution events, one prediction matching both.
    >>> solution = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 10, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 10, 'stop_frame': 20, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 20},
    ... ])
    >>> mouse_fbeta(solution, submission)
    1.0

    >>> solution = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 10, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 30, 'stop_frame': 40, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 40},
    ... ])
    >>> mouse_fbeta(solution, submission)
    0.6666666666666666
    """
    if len(solution) == 0 or len(submission) == 0:
        raise ValueError("Missing solution or submission data")

    expected_cols = [
        "video_id",
        "agent_id",
        "target_id",
        "action",
        "start_frame",
        "stop_frame",
    ]

    for col in expected_cols:
        if col not in solution.columns:
            raise ValueError(f"Solution is missing column {col}")
        if col not in submission.columns:
            raise ValueError(f"Submission is missing column {col}")

    solution: pl.DataFrame = pl.DataFrame(solution)
    submission: pl.DataFrame = pl.DataFrame(submission)
    assert (solution["start_frame"] <= solution["stop_frame"]).all()
    assert (submission["start_frame"] <= submission["stop_frame"]).all()
    solution_videos = set(solution["video_id"].unique())
    # Need to align based on video IDs as we can't rely on the row IDs for handling public/private splits.
    submission = submission.filter(pl.col("video_id").is_in(solution_videos))

    solution = solution.with_columns(
        pl.concat_str(
            [
                pl.col("video_id").cast(pl.Utf8),
                pl.col("agent_id").cast(pl.Utf8),
                pl.col("target_id").cast(pl.Utf8),
                pl.col("action"),
            ],
            separator="_",
        ).alias("label_key"),
    )
    submission = submission.with_columns(
        pl.concat_str(
            [
                pl.col("video_id").cast(pl.Utf8),
                pl.col("agent_id").cast(pl.Utf8),
                pl.col("target_id").cast(pl.Utf8),
                pl.col("action"),
            ],
            separator="_",
        ).alias("prediction_key"),
    )

    res = SubmissionScore()
    lab_scores = []
    for lab in sorted(solution["lab_id"].unique()):
        lab_solution = solution.filter(pl.col("lab_id") == lab).clone()
        lab_videos = set(lab_solution["video_id"].unique())
        lab_submission = submission.filter(pl.col("video_id").is_in(lab_videos)).clone()
        lab_score = single_lab_f1(lab_solution, lab_submission, beta=beta)
        res.per_lab[lab] = lab_score
        lab_scores.append(lab_score.f1)

    res.f1 = sum(lab_scores) / len(lab_scores)
    return res


def score_map(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str = "row_id",
    beta: float = 1,
):
    solution = solution.drop(row_id_column_name, axis="columns", errors="ignore")
    submission = submission.drop(row_id_column_name, axis="columns", errors="ignore")
    return mouse_fbeta(solution, submission, beta=beta)


if __name__ == "__main__":
    import argparse

    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--gt", default="data/test_gt.csv")
        parser.add_argument("submission", default=None)
        parser.add_argument("--actions", required=False, action="store_true")
        return parser.parse_args()

    args = parse_args()

    solution = pd.read_csv(args.gt)
    submission = pd.read_csv(args.submission)
    PRINT_ACTIONS = args.actions
    score = score_map(
        solution=solution, submission=submission, row_id_column_name="row_id"
    )
    for lab in sorted(score.per_lab.keys()):
        if args.actions:
            lab_score = score.per_lab[lab]
            for action in sorted(lab_score.per_action.keys()):
                f1 = lab_score.per_action[action]
                print(f"   {action:16} : {f1:.5f}")
        print(f"{lab:20}: {score.per_lab[lab].f1:.5f}")
    print(f"Score is: {score.f1:.5f}")
