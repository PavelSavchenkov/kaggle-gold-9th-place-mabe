from pathlib import Path
from typing import Any

import numpy as np
import torch  # type: ignore
from safetensors.torch import load_model  # type: ignore

from common.constants import ACTION_NAMES_IN_TEST
from common.helpers import get_default_cuda_float_dtype
from dl.configs import DL_TrainConfig
from dl.conv_transformer import MiceConvTransformerModel
from dl.multiscale_tcn import MiceMultiScaleTCNModel
from dl.tcn import MiceTCNModel


def model_factory(**kwargs):
    kwargs = dict(kwargs)
    train_config: DL_TrainConfig = kwargs["train_config"]
    if "stats_path" not in kwargs or kwargs["stats_path"] is None:
        kwargs["stats_path"] = train_config.save_dir() / "stats.npz"
    model_cls = None
    if train_config.tcn_config is not None:
        model_cls = MiceTCNModel
    elif train_config.multiscale_tcn_config is not None:
        model_cls = MiceMultiScaleTCNModel
    elif train_config.conv_transformer_config is not None:
        model_cls = MiceConvTransformerModel
    else:
        raise ValueError(
            f"At least one model config should be set in train_config: {train_config.name} | cv{train_config.data_split_config.test_fold}"
        )
    model = model_cls(**kwargs)
    return model


class InferenceModel:
    def __init__(self, ckpt_dir: Path, **model_kwargs) -> None:
        self.device = torch.device("cuda")
        self.model = model_factory(**model_kwargs)

        weights_path = ckpt_dir / "model.safetensors"
        _, _ = load_model(
            self.model,
            str(weights_path),
            device=str(self.device),
            strict=False,
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = dict(batch)
        # n = None
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.device != self.device:
                batch[k] = v.to(self.device, non_blocking=True)
                # n = batch[k].size(0)

        # res = {}
        # res["probs"] = torch.zeros((n, len(ACTION_NAMES_IN_TEST)), dtype=torch.float32, device=self.device)
        # return res

        with torch.autocast(
            device_type=self.device.type, dtype=get_default_cuda_float_dtype()
        ):
            out = self.model(**batch)
            out["probs"] = torch.sigmoid(out["logits"])
        return out
