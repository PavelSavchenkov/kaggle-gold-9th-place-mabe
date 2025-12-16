import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch  # type: ignore
import wandb  # type: ignore
from transformers import Trainer, TrainingArguments  # type: ignore

from common.config_utils import (
    base_model_from_file,
    base_model_to_dict,
    base_model_to_file,
)
from common.folds_split_utils import train_test_split
from common.helpers import get_train_meta
from dl.callbacks import PeriodicEvalSaveCallback  # type: ignore
from dl.callbacks import (
    EMACallback,
    SetEpochOnDatasetCallback,
    SetupStopStepCallback,
    StopAfterNStepsCallback,
)
from dl.configs import DL_TrainConfig
from dl.data_balancer import DataBalancerForLabeledTest, DataBalancerForLabeledTrain
from dl.dataset import MiceDataset
from dl.features_lookup_gpu import FeaturesLookupGPU
from dl.model_helpers import model_factory
from dl.postprocess import (
    full_training_steps_from_config_dl,
    get_latest_saved_checkpoint,
    get_latest_saved_checkpoint_with_optim,
    is_fully_trained_according_to_saved_checkpoints,
    iter_all_checkpoints_for_model_cv,
    remove_25fps_adaptable_snail_from_meta,
    total_actual_training_steps_from_config_dl,
)
from dl.stats import calc_hash_for_stats, calc_stats  # type: ignore
from dl.trainer_subclasses import EMATrainer  # type: ignore
from dl.trainer_subclasses import AWPTrainer, calculate_total_training_steps


def run_training_process(train_config: DL_TrainConfig):
    save_dir = train_config.save_dir()
    save_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_dir / "train_config.json"

    if is_fully_trained_according_to_saved_checkpoints(save_dir):
        print(f"{save_dir} according to Trainer checkpoints. Skip.")
        return

    full_training_steps = full_training_steps_from_config_dl(train_config)
    total_actual_training_steps = total_actual_training_steps_from_config_dl(
        train_config
    )
    max_existing_saved_step = -1
    latest_ckpt = get_latest_saved_checkpoint(train_config.save_dir())
    if latest_ckpt is not None:
        max_existing_saved_step = latest_ckpt.step
        latest_step_to_save = total_actual_training_steps
        latest_step_to_save -= latest_step_to_save % train_config.eval_steps  # !!!!!!!
        if latest_ckpt.step >= latest_step_to_save:
            print(
                f"{save_dir} already has latest checkpoint saved at step={latest_ckpt.step}. Skip."
            )
            return

    desc = "===== "
    desc += "[RUN TRAINING] "
    desc += f"{train_config.name} | cv{train_config.data_split_config.test_fold}"
    desc += f" | train for {total_actual_training_steps}/{full_training_steps}"
    if max_existing_saved_step != -1:
        desc += f" | max existing saved step is {max_existing_saved_step}"
    desc += " ====="

    print(desc)

    base_model_to_file(train_config, config_path)
    subprocess.run(["python", "-m", "dl.train", "--config", config_path], check=False)


class MyTrainer(
    EMATrainer,
    AWPTrainer,
    Trainer,
):
    def __init__(self, *args, train_config: DL_TrainConfig, **kwargs):
        self.train_config = train_config
        super().__init__(*args, **kwargs)


def prepare_data(
    train_config: DL_TrainConfig,
    stats_cache: Path | str = "cache/stats",
) -> dict:
    meta = get_train_meta()
    train_meta, test_meta = train_test_split(
        meta,
        train_config.data_split_config,
        remove_25fps_adaptable_snail_from_train=train_config.remove_25fps_adaptable_snail_from_train,
    )

    save_dir = train_config.save_dir()

    info_dir = save_dir / "info"
    info_dir.mkdir(parents=True, exist_ok=True)

    def get_video_ids(df: pd.DataFrame) -> list[int]:
        return [int(x) for x in sorted(df.video_id.unique())]

    split_json = {
        "train_videos": get_video_ids(train_meta),
        "test_videos": get_video_ids(test_meta),
    }
    json.dump(split_json, open(info_dir / "split.json", "w"))

    if train_config.remove_25fps_adaptable_snail_from_train:
        train_meta = remove_25fps_adaptable_snail_from_meta(
            meta=train_meta, verbose=True
        )

    # train_meta = train_meta[:10]
    # test_meta = test_meta[:5]

    train_balancer = DataBalancerForLabeledTrain(
        meta=train_meta,
        balance_config=train_config.train_balance_config,
        aug_config=train_config.aug,
    )
    test_balancer = DataBalancerForLabeledTest(
        meta=test_meta,
    )
    train = MiceDataset(
        data_balancer_for_labeled=train_balancer,
    )
    test = MiceDataset(
        data_balancer_for_labeled=test_balancer,
    )

    meta_both = pd.concat([train_meta, test_meta], ignore_index=True)
    features_lookup = FeaturesLookupGPU(
        meta=meta_both,
        feats_config=train_config.features_config,
        aug_config=train_config.aug,
    )

    stats_cache = Path(stats_cache)
    stats_cache_key = calc_hash_for_stats(
        meta=train_meta, features_config=train_config.features_config
    )
    stats_path = stats_cache / f"{stats_cache_key}.npz"
    if not stats_path.exists():
        stats_path.parent.mkdir(exist_ok=True, parents=True)
        stats = calc_stats(
            balancer=train_balancer,
            lookup=features_lookup,
        )
        np.savez(stats_path, **stats)
    else:
        print(f"Mean/std stats found at {stats_path}.\n")
    shutil.copy2(stats_path, save_dir / "stats.npz")

    return {
        "train_dataset": train,
        "test_dataset": test,
        "features_lookup": features_lookup,
        "stats_path": stats_path,
    }


def init_wandb(train_config: DL_TrainConfig):
    assert train_config.use_wandb

    id_file_path = train_config.save_dir() / "wandb_run_id.txt"

    run_id = None
    if id_file_path.exists():
        run_id = id_file_path.read_text().strip()
        print(f"\nWandb id file exists: {id_file_path}. Run id: {run_id}\n")

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT"),
        name=train_config.run_name(),
        group=train_config.name,
        id=run_id,
        resume="allow",
        config=base_model_to_dict(train_config),
    )

    if run_id is None:
        id_file_path.write_text(run.id)

    return run


def train(train_config: DL_TrainConfig):
    torch.set_float32_matmul_precision("high")

    prepared = prepare_data(train_config)

    model = model_factory(
        feats_lookup=prepared["features_lookup"],
        stats_path=prepared["stats_path"],
        train_config=train_config,
    )

    cnt_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cnt_all_params = sum(p.numel() for p in model.parameters())
    print(
        f"\n[Model Params] Trainable: {cnt_trainable/1e6:.2f}M, Overall: {cnt_all_params/1e6:.2f}M\n"
    )

    save_steps = train_config.save_steps
    if save_steps is None:
        save_steps = train_config.eval_steps
    args = TrainingArguments(
        output_dir=train_config.save_dir(),
        per_device_train_batch_size=train_config.train_bs,
        per_device_eval_batch_size=train_config.eval_bs,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        # learning hparams
        learning_rate=train_config.lr,
        weight_decay=train_config.wd,
        lr_scheduler_type=train_config.lr_scheduler_type,
        warmup_ratio=train_config.warmup_ratio,
        max_grad_norm=1.0,
        num_train_epochs=train_config.epochs,
        max_steps=train_config.max_steps,
        # misc
        remove_unused_columns=False,
        report_to=["wandb"] if train_config.use_wandb else "none",
        bf16=True,
        bf16_full_eval=True,
        torch_compile=True,
        torch_compile_backend="inductor",
        # dataloader performance
        dataloader_num_workers=12,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,  # PyTorch default is 2
        # logging and eval
        eval_strategy="no",
        save_strategy="steps",
        save_steps=save_steps,
        # save_only_model=True,
        logging_strategy="steps",
        logging_steps=train_config.logging_steps,
        include_for_metrics=["labels"],
    )

    trainer = MyTrainer(
        model=model,
        args=args,
        train_dataset=prepared["train_dataset"],
        eval_dataset=prepared["test_dataset"],
        train_config=train_config,
    )

    trainer.add_callback(SetupStopStepCallback(trainer))
    trainer.add_callback(SetEpochOnDatasetCallback(trainer))
    trainer.add_callback(PeriodicEvalSaveCallback(trainer))
    trainer.add_callback(EMACallback(trainer=trainer))
    trainer.add_callback(StopAfterNStepsCallback(trainer=trainer))

    if train_config.use_wandb:
        init_wandb(train_config=train_config)

    train_kwargs = {}
    if train_config.resume_from_latest_ckpt:
        ckpt = get_latest_saved_checkpoint_with_optim(train_config.save_dir())
        if ckpt is not None:
            train_kwargs["resume_from_checkpoint"] = ckpt.path
            print(f"\n[RESUME TRAINING] Weights from {ckpt.path}\n")

    print(
        f"\nTotal training steps (from trainer): {calculate_total_training_steps(trainer=trainer)}\n"
    )

    trainer.train(**train_kwargs)

    if train_config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    import argparse

    # import cProfile
    # from pstats import Stats
    # profiler = cProfile.Profile()
    # profiler.enable()

    parser = argparse.ArgumentParser("DL training")
    parser.add_argument("--config_path", required=True, type=str)
    args = parser.parse_args()

    config = base_model_from_file(DL_TrainConfig, args.config_path)
    train(config)

    # profiler.disable()
    # Stats(profiler).strip_dirs().sort_stats("cumtime").print_stats(50)
