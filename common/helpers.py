import os
import shutil
import sys
import tempfile
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy
import torch  # type: ignore

from common.paths import ANNOTATIONS_ROOT, MODELS_ROOT, TRACKING_ROOT, VIDEOS_META_ROOT


@lru_cache
def get_tracking_df(lab_id: str, video_id: int) -> pd.DataFrame:
    return pd.read_parquet(TRACKING_ROOT / lab_id / f"{video_id}.parquet")


def get_tracking_by_video_meta(video_meta: dict) -> pd.DataFrame:
    video_id = video_meta["video_id"]
    lab_id = video_meta["lab_id"]
    return get_tracking_df(lab_id=lab_id, video_id=video_id)


@lru_cache
def get_annotation_df(lab_id: str, video_id: int) -> pd.DataFrame:
    return pd.read_parquet(ANNOTATIONS_ROOT / lab_id / f"{video_id}.parquet")


def has_annotation_df(lab_id: str, video_id: int) -> bool:
    return (ANNOTATIONS_ROOT / lab_id / f"{video_id}.parquet").exists()


def get_annotation_by_video_meta(video_meta: dict) -> pd.DataFrame:
    video_id = video_meta["video_id"]
    lab_id = video_meta["lab_id"]
    return get_annotation_df(lab_id=lab_id, video_id=video_id)


@lru_cache
def get_train_meta(only_annotated: bool = True) -> pd.DataFrame:
    res = pd.read_csv(VIDEOS_META_ROOT, low_memory=False)
    if only_annotated:
        res = res[res.has_annotation]
    return res


def get_model_cv_path(name: str, cv: str | int) -> Path:
    if isinstance(cv, int):
        cv = f"cv{cv}"
    return MODELS_ROOT / name / cv


def get_config_path(name: str, cv: str | int) -> Path:
    return get_model_cv_path(name, cv) / "train_config.json"


def ensure_1d_numpy(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    if x.ndim == 2 and x.shape[1] == 1:
        return x.ravel()  # (n, )
    raise ValueError(f"Cannot make array 1d: {x.shape}.")


def str_uint32_hash(s: str) -> int:
    """Deterministic 32-bit unsigned hash of a string."""
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def copy_all_source_code_to(dst: Path | str) -> None:
    SRC_ROOT = Path("/home/pavel/Programming/Social-Action-Recognition-in-Mice")
    IGNORE_ROOTS = [SRC_ROOT / "submissions"]

    src_root = SRC_ROOT.expanduser().resolve()
    dst_path = Path(dst).expanduser().resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Source folder does not exist: {src_root}")

    assert not dst_path.exists()
    dst_path.mkdir(parents=True, exist_ok=True)

    items = []
    for path in src_root.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".ipynb"}:
            skip = False
            for p in IGNORE_ROOTS:
                if path.is_relative_to(p):
                    skip = True
                    break
            if skip:
                continue
            rel = path.relative_to(src_root)
            target = dst_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            items.append((path, target))
    for path, target in items:
        shutil.copy2(path, target)


def is_local():
    return "PAVEL_DESKTOP" in os.environ


def raise_if_not_local():
    if not is_local():
        raise RuntimeError(f"Code is not running locally, but must be")


def get_default_cuda_float_dtype():
    if torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.bfloat16
    else:
        return torch.float32


def write_to_dir_atomically(dst_dir: Path | str, func: Callable[[Path], None]):
    dst_dir = Path(dst_dir)
    if dst_dir.exists():
        raise FileExistsError(dst_dir)
    dst_dir.parent.mkdir(exist_ok=True, parents=True)
    tmp_dir = Path(
        tempfile.mkdtemp(
            dir=dst_dir.parent,
            prefix=dst_dir.name + "_tmp_",
        )
    )
    try:
        func(tmp_dir)
        tmp_dir.rename(dst_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def get_visible_gpu_ids() -> list[str]:
    if not torch.cuda.is_available():
        return []

    env_val = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_val:
        ids = [x.strip() for x in env_val.split(",") if x.strip()]
        if ids:
            return ids

    return [str(i) for i in range(torch.cuda.device_count())]


def prob_to_logit(
    x: np.ndarray | torch.Tensor, eps: float = 1e-6
) -> np.ndarray | torch.Tensor:
    if isinstance(x, np.ndarray):
        assert x.dtype == np.float32
        x = np.clip(x, eps, 1 - eps)
        return np.log(x) - np.log1p(-x)
    else:
        assert x.dtype == torch.float32
        return torch.logit(x, eps=eps)


def logit_to_prob(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(x, np.ndarray):
        return scipy.special.expit(x)
    else:
        return torch.sigmoid(x)
