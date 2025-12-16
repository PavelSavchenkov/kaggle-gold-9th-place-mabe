from __future__ import annotations

import contextlib
import copy

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.optim.swa_utils import get_ema_multi_avg_fn  # type: ignore

from dl.awp import AWP  # type: ignore
from dl.configs import DL_TrainConfig


def calculate_total_training_steps(trainer):
    args = trainer.args
    if args.max_steps > 0:
        return args.max_steps
    dl = trainer.get_train_dataloader()
    total_bs = trainer.get_total_train_batch_size(trainer.args)
    vals = trainer.set_initial_training_values(trainer.args, dl, total_bs)
    return vals[-1]


class _SelectiveEMA:
    """
    Tracks EMA for trainable params only (requires_grad=True).
    Buffers are optional and are MIRRORED (not averaged) if enabled.
    """

    def __init__(self, model: nn.Module, multi_avg_fn, track_buffers: bool = False):
        self.multi_avg_fn = multi_avg_fn
        self.device = torch.device("cuda")

        self.param_names: list[str] = [
            n for n, p in model.named_parameters() if p.requires_grad
        ]

        self.shadow: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.param_names:
                    self.shadow[n] = p.detach().clone().to(self.device)

        self.track_buffers = bool(track_buffers)
        self.buffer_names: list[str] = []
        self.buffers_shadow: dict[str, torch.Tensor] = {}
        if self.track_buffers:
            with torch.no_grad():
                for n, b in model.named_buffers():
                    self.buffer_names.append(n)
                    self.buffers_shadow[n] = b.detach().clone().to(self.device)

        self.num_updates: int = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1

        model_params = dict(model.named_parameters())
        for n in self.param_names:
            assert n in model_params
            ema_t = self.shadow[n]
            cur_t = model_params[n].detach()
            self.multi_avg_fn([ema_t], [cur_t], self.num_updates)

        if self.track_buffers:
            model_buffers = dict(model.named_buffers())
            for n in self.buffer_names:
                if n in model_buffers:
                    self.buffers_shadow[n].copy_(model_buffers[n])

    @contextlib.contextmanager
    def swap_into(self, model: nn.Module):
        backups: dict[str, torch.Tensor] = {}
        buf_backups: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            model_params = dict(model.named_parameters())
            for n in self.param_names:
                if n in model_params:
                    p = model_params[n]
                    backups[n] = p.detach().clone()
                    p.copy_(self.shadow[n].to(p.device))

            if self.track_buffers:
                model_buffers = dict(model.named_buffers())
                for n in self.buffer_names:
                    if n in model_buffers:
                        b = model_buffers[n]
                        buf_backups[n] = b.detach().clone()
                        b.copy_(self.buffers_shadow[n].to(b.device))

        try:
            yield
        finally:
            with torch.no_grad():
                model_params = dict(model.named_parameters())
                for n, t in backups.items():
                    if n in model_params:
                        model_params[n].copy_(t)

                if self.track_buffers:
                    model_buffers = dict(model.named_buffers())
                    for n, t in buf_backups.items():
                        if n in model_buffers:
                            model_buffers[n].copy_(t)


class EMATrainer:
    train_config: DL_TrainConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ema_config = self.train_config.ema
        if not self.ema_config.enabled:
            return

        avg_fn = get_ema_multi_avg_fn(self.ema_config.decay)
        track_buffers = False
        self._ema = _SelectiveEMA(
            self.model,  # type: ignore
            multi_avg_fn=avg_fn,
            track_buffers=track_buffers,
        )

        assert self.ema_config.update_every is not None
        assert self.ema_config.update_every > 0

        total_steps = calculate_total_training_steps(self)
        start_step = int(self.train_config.warmup_ratio * total_steps)
        self.ema_start_step = start_step
        self.ema_update_every = self.ema_config.update_every
        self.ema_started = False

        print()
        print(
            f"[EMA ON] decay: {self.ema_config.decay:.4f}, every: {self.ema_config.update_every}, start_step: {self.ema_start_step}"
        )
        print()

    def maybe_update_ema(self):
        if not self.ema_config.enabled:
            return
        step = self.state.global_step  # type: ignore
        if step < self.ema_start_step:
            return
        if step % self.ema_update_every != 0:
            return
        if not self.ema_started:
            print(f"[EMA] first time at step: {step}")
            self.ema_started = True

        self._ema.update(self.model)  # type: ignore

    @contextlib.contextmanager
    def use_ema_weights(self):
        assert self.ema_config.enabled
        with self._ema.swap_into(self.model):  # type: ignore
            yield

    def evaluate(self, *args, **kwargs):
        if self.ema_config.enabled:
            self.maybe_update_ema()
            with self.use_ema_weights():
                return super().evaluate(*args, **kwargs)  # type: ignore
        return super().evaluate(*args, **kwargs)  # type: ignore

    def predict(self, *args, **kwargs):
        if self.ema_config.enabled:
            self.maybe_update_ema()
            with self.use_ema_weights():
                return super().predict(*args, **kwargs)  # type: ignore
        return super().predict(*args, **kwargs)  # type: ignore

    def save_model(self, *args, **kwargs):
        if self.ema_config.enabled:
            self.maybe_update_ema()
            with self.use_ema_weights():
                return super().save_model(*args, **kwargs)  # type: ignore
        return super().save_model(*args, **kwargs)  # type: ignore

    def _save_checkpoint(self, model, trial):
        if self.ema_config.enabled:
            self.maybe_update_ema()
            with self.use_ema_weights():
                return super()._save_checkpoint(model, trial)  # type: ignore
        return super()._save_checkpoint(model, trial)  # type: ignore


class AWPTrainer:
    train_config: DL_TrainConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.awp_config = self.train_config.awp
        if self.awp_config is None:
            return

        self.awp_start_step = int(
            calculate_total_training_steps(self) * self.awp_config.start_ratio
        )
        self.awp_started = False

        print()
        print(
            f"[AWP ON] lr: {self.awp_config.lr}, eps: {self.awp_config.eps}, to_names: {self.awp_config.apply_to_names_with}, start_step: {self.awp_start_step}"
        )
        print()

    def training_step(self, model, inputs, num_items_in_batch):
        model.train()
        inputs = self._prepare_inputs(inputs)  # type: ignore

        with self.compute_loss_context_manager():  # type: ignore
            loss = self.compute_loss(model, inputs)  # type: ignore
        loss = loss / self.args.gradient_accumulation_steps  # type: ignore
        loss.backward()

        if self.awp_config is not None:
            if not self.awp_started and self.state.global_step >= self.awp_start_step:  # type: ignore
                print(f"[AWP START] step: {self.state.global_step}")  # type: ignore
                self.awp_started = True

            if self.awp_started:  # type: ignore
                awp = AWP(model, config=self.awp_config)

                awp.backup_and_perturb()

                with self.compute_loss_context_manager():  # type: ignore
                    adv_loss = self.compute_loss(model, inputs)  # type: ignore
                adv_loss = adv_loss / self.args.gradient_accumulation_steps  # type: ignore
                adv_loss.backward()

                awp.restore()

        return loss.detach()
