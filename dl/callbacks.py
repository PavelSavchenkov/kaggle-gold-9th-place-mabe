import json
import time
from pathlib import Path

import numpy as np
import torch  # type: ignore
import wandb  # type: ignore
from atomicwrites import atomic_write
from tqdm import tqdm
from transformers import TrainerCallback  # type: ignore
from transformers import TrainerControl  # type: ignore
from transformers import Trainer, TrainerState, TrainingArguments  # type: ignore

from common.helpers import get_default_cuda_float_dtype
from dl.metrics import MetricsComputer  # type: ignore
from dl.metrics import LAB_FEATURE_NAME, EvalData
from dl.postprocess import Ckpt


class SetupStopStepCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        assert state.max_steps > 0
        self.trainer.steps_full_training = state.max_steps
        self.trainer.actual_stop_step = state.max_steps

        max_steps_to_run = self.trainer.train_config.max_steps_to_run
        if max_steps_to_run is not None:
            self.trainer.actual_stop_step = min(
                self.trainer.actual_stop_step, max_steps_to_run
            )

        return control


class SetEpochOnDatasetCallback(TrainerCallback):
    trainer: Trainer

    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def on_epoch_begin(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        epoch_idx = int(np.floor(state.epoch)) if state.epoch is not None else 0
        self.trainer.train_dataset.set_epoch(epoch_idx)


class PeriodicEvalSaveCallback(TrainerCallback):
    trainer: Trainer
    metrics_computer: MetricsComputer

    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.last_eval_data = None

    @torch.no_grad()
    def _run_eval_and_compute_data(self, trainer: Trainer) -> EvalData:
        time_start = time.time()

        model = trainer.model
        model.eval()

        eval_ds = trainer.eval_dataset
        assert eval_ds is not None

        dataloader = trainer.get_eval_dataloader(eval_ds)

        probs_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []
        labels_known_list: list[np.ndarray] = []
        lab_ids_list: list[np.ndarray] = []

        device = trainer.args.device

        progress = tqdm(
            dataloader,
            desc=f"Test Infer",
            total=len(dataloader),
            dynamic_ncols=True,
            leave=False,
        )

        computer_empty = not hasattr(self, "metrics_computer")

        for batch in progress:
            batch = trainer._prepare_inputs(batch)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=get_default_cuda_float_dtype()
            ):
                outputs = model(**batch)
                probs = outputs["probs"]  # [B, C]
                lab_ids = outputs[LAB_FEATURE_NAME]

            probs_list.append(probs.cpu().numpy())
            if computer_empty:
                labels_list.append(batch["labels"].cpu().numpy())
                labels_known_list.append(batch["labels_known"].cpu().numpy())
                lab_ids_list.append(lab_ids.cpu().numpy())

        probs = np.concatenate(probs_list, axis=0)
        if computer_empty:
            labels = np.concatenate(labels_list, axis=0)
            labels_known = np.concatenate(labels_known_list, axis=0)
            lab_ids = np.concatenate(lab_ids_list, axis=0)
            self.metrics_computer = MetricsComputer(
                labels=labels, labels_known=labels_known, lab_ids=lab_ids
            )

        assert not self.metrics_computer is None
        eval_data = self.metrics_computer.compute_eval_data(probs=probs)

        eval_data.metrics = {
            f"test-{key}": value for key, value in eval_data.metrics.items()
        }
        wall_time = time.time() - time_start
        print(f"\nEval Wall Time: {wall_time:.2f} s.")
        for key in ["test-prod/pr-auc", "test-prod/argmax-pr-auc"]:
            print(f"{key}: {eval_data.metrics[key]:.6f}")
        print()
        return eval_data

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if state.global_step == 0:
            return control
        if state.global_step == self.trainer.actual_stop_step:
            control.should_save = True
        if state.global_step % self.trainer.train_config.eval_steps != 0:
            return control

        eval_data = self._run_eval_and_compute_data(self.trainer)

        eval_data.metrics["step"] = int(state.global_step)
        eval_data.metrics["epoch"] = float(state.epoch)

        self.last_eval_data = eval_data

        if self.trainer.train_config.use_wandb and self.trainer.is_world_process_zero():
            wandb.log(eval_data.metrics, step=state.global_step)

        control.should_save = True
        return control

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        step = state.global_step
        if self.last_eval_data is None or self.last_eval_data.metrics["step"] != step:
            return control

        ckpt_dir = Path(args.output_dir) / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = Ckpt(path=ckpt_dir, step=step)

        json.dump(self.last_eval_data.metrics, ckpt.metrics_path().open("w"), indent=2)

        gt_dir = ckpt.gt_dir()
        gt_dir.mkdir(exist_ok=True)
        for (action, lab), gt in self.last_eval_data.gt_map.items():
            dst = ckpt.gt_npy_path(action=action, lab=lab)
            if not dst.exists():
                gt = gt > 0.5
                np.save(dst, gt)
        for (action, lab), preds in self.last_eval_data.preds_map.items():
            preds_dir = ckpt.preds_dir()
            preds_dir.mkdir(exist_ok=True, parents=True)
            preds = preds.astype(np.float16)
            np.save(ckpt.preds_npy_path(action=action, lab=lab), preds)
        return control


class EMACallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_optimizer_step(self, *args, **kwargs):
        self.trainer.maybe_update_ema()


class StopAfterNStepsCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.trainer.actual_stop_step:
            control.should_training_stop = True
        return control
