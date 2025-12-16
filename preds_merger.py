import argparse
import enum
import pickle
from pathlib import Path

import pandas as pd  # type: ignore

from common.submission_common import (
    PredictedProbsForBehavior,
    moving_average,
    predicted_probs_to_segments,
)
from dl.calibration_postprocess import (
    optimise_f1_oof_calibrate_probs_and_refit_thresholds,
)
from dl.pairs_postprocessor import maximise_f1_one_action_pairwise_oof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preds_files",
        nargs="+",
        required=True,
        help="Paths to .pkl files with lists of PredictedProbsForBehavior",
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        help="Path to .csv file to save merged postprocessed segments",
    )
    parser.add_argument(
        "--pairs_json",
        required=False,
        default=None,
    )
    parser.add_argument("--moving_average", required=False, type=int, default=None)
    return parser.parse_args()


def load_predictions_from_file(path: Path) -> list[PredictedProbsForBehavior]:
    print(f"[LOAD] Reading predictions from {path}")
    with path.open("rb") as f:
        preds = pickle.load(f)
    if not isinstance(preds, list):
        raise RuntimeError(f"Expected list in {path}, got {type(preds)}")
    print(f"[LOAD] Loaded {len(preds)} predictions from {path}")
    return preds


def main() -> None:
    args = parse_args()

    preds_files = [Path(p) for p in args.preds_files]

    print()
    print("===== MERGING PREDICTION SHARDS =====")
    print(f"NUM PREDS FILES: {len(preds_files)}")
    for i, p in enumerate(preds_files):
        print(f"  [{i}] {p}")
    print(f"OUT CSV: {args.out_csv}")
    print("=====================================")
    print()

    all_predictions: list[PredictedProbsForBehavior] = []
    for p in preds_files:
        if not p.exists():
            raise FileNotFoundError(f"Preds file does not exist: {p}")
        shard_preds = load_predictions_from_file(p)
        all_predictions.extend(shard_preds)

    print(f"[MERGE] Total predictions: {len(all_predictions)}")

    print(f"[MERGING BY ACTION] ...")
    pos_of = {}
    predictions_merged = []
    for i, pred in enumerate(all_predictions):
        assert (
            pred.threshold < 1.0
        ), f"action: {pred.behavior.action}, lab: {pred.lab_name}"
        key = (
            pred.video_id,
            pred.behavior.agent,
            pred.behavior.target,
            pred.behavior.action,
        )
        if key not in pos_of:
            pos_of[key] = len(predictions_merged)
            predictions_merged.append(pred)
        else:
            j = pos_of[key]
            assert (
                abs(predictions_merged[j].threshold - pred.threshold) < 1e-3
            ), f"prev.th: {predictions_merged[j].threshold}, cur.th: {pred.threshold}"
            predictions_merged[j].probs += pred.probs
    del all_predictions
    all_predictions = predictions_merged

    # maximise_f1_one_action_pairwise_oof(
    #     predictions=all_predictions,
    #     save_json_path="pairs_14_dec_ens12_v2.json",
    #     use_ge2_active_samples=False,
    # )
    # optimise_f1_oof_calibrate_probs_and_refit_thresholds(predictions=all_predictions)

    if args.moving_average is not None:
        print(f"[MOVING AVERAGE] window={args.moving_average}")
        assert args.moving_average % 2 == 1
        for pred in all_predictions:
            pred.probs = moving_average(pred.probs, window=args.moving_average)

    print("[CONVERT] Converting predictions to segments DataFrame...")
    df = predicted_probs_to_segments(
        all_predictions, pairs_params_json_path=args.pairs_json
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"[SAVE] Writing segments to {out_csv}")
    df.index.name = "row_id"
    df.to_csv(out_csv)
    print("[DONE] Finished writing merged segments CSV.")


if __name__ == "__main__":
    main()
