import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, Field


class SplitByStrategy(str, Enum):
    cnt_videos = "cnt_videos"
    total_duration = "total_duration"
    cnt_segments = "cnt_segments"


class FoldsIntegerProgrammingParams(BaseModel):
    balance_counts: bool = Field(default=False)
    time_limit_s: float = Field(default=5.0)
    mip_rel_gap: Optional[float] = Field(default=None, ge=0.0)
    use_warm_start: bool = Field(default=False)
    minimise_nan_cells_first: bool = Field(default=True)


class DataSplitConfig(BaseModel):
    seed: int = Field(default=0)
    num_folds: int = Field(default=0, gt=0)
    test_fold: int = Field(default=0, ge=0)
    train_folds: list[int] | None = None
    actions: list[str] = Field(default=[])
    split_strategy: SplitByStrategy = Field(default=SplitByStrategy.total_duration)
    folds_integer_programming_params: FoldsIntegerProgrammingParams = Field(
        default_factory=lambda: FoldsIntegerProgrammingParams()
    )


# ----- general config utils -----


def base_model_to_dict(m) -> dict:
    return m.model_dump(by_alias=True)


def base_model_to_str(m) -> str:
    return m.model_dump_json(indent=4, by_alias=True)


def base_model_to_file(m, path: str | Path):
    Path(path).write_text(base_model_to_str(m))


T = TypeVar("T", bound=BaseModel)


def base_model_from_dict(cls: Type[T], d: dict, strict=False) -> T:
    return cls.model_validate(d, strict=strict)


def base_model_from_str(cls: Type[T], s: str, strict=False) -> T:
    return cls.model_validate_json(s, strict=strict)


def base_model_from_file(cls: Type[T], path: str | Path, strict=False) -> T:
    return base_model_from_str(cls, Path(path).read_text(), strict=strict)


def base_model_to_hash(m) -> str:
    data = m.model_dump(mode="json")
    s = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(s.encode("utf-8")).hexdigest()
