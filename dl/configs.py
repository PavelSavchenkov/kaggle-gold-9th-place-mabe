from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from common.config_utils import DataSplitConfig
from common.constants import ACTION_NAMES_IN_TEST, BODYPART_GROUPS_V0
from common.paths import MODELS_ROOT


class SingleMouseNumericFeaturesConfig(BaseModel):
    origin: str = "tail_base"
    heading_left: str = "ear_left"
    heading_right: str = "ear_right"

    # keypoints to keep in body-centric coords
    normalised_keypoints: list[str] = Field(
        default_factory=lambda: [
            "nose",
            "neck",
            "body_center",
            "tail_base",
            "hip_left",
            "hip_right",
            "lateral_left",
            "lateral_right",
            "ear_left",
            "ear_right",
            "tail_tip",
        ]
    )

    # finite differences in body coords
    include_normalised_velocity: bool = True
    include_normalised_speed: bool = True

    # pairwise distances in body coords
    intra_distance_pairs: list[tuple[str, str]] = Field(
        default_factory=lambda: [
            ("tail_base", "nose"),
            ("tail_base", "neck"),
            ("tail_base", "body_center"),
            ("hip_left", "hip_right"),
            ("lateral_left", "lateral_right"),
            ("neck", "nose"),
            ("tail_base", "tail_tip"),
            ("body_center", "nose"),
            ("body_center", "neck"),
            ("ear_left", "ear_right"),
        ]
    )

    # heading and simple shape angles
    include_heading_angle: bool = True  # θ
    include_heading_angle_trig: bool = True  # sin θ, cos θ
    include_ear_splay: bool = True  # nose→ear_left vs nose→ear_right
    include_torso_bend: bool = True  # neck→hip_left vs neck→hip_right

    # scalar body length (||H|| from origin to ear-midpoint)
    include_body_length: bool = True

    # curvature of selected keypoints in body CS
    curvature_keypoints: list[str] = Field(
        default_factory=lambda: ["nose", "body_center"]
    )

    include_ear_distance: bool = False
    include_ear_distance_velocity: bool = False  # central diff of ear distance

    # head vs tail-axis coupling angle
    include_body_axis_coupling: bool = False

    # multi-lag nose displacement (in body CS, /||H||^2)
    include_nose_lag_displacements: bool = False
    nose_lag_displacement_frames: list[int] = Field(
        default_factory=lambda: [2, 4, 8]  # ~short/med lags within 25-frame window
    )


class PairNumericFeaturesConfig(BaseModel):
    # (agent_keypoint, target_keypoint), all expressed in agent's body CS
    distance_pairs: list[tuple[str, str]] = Field(
        default_factory=lambda: [
            ("nose", "tail_base"),
            ("nose", "nose"),
            ("tail_base", "tail_base"),
            ("nose", "body_center"),
            ("body_center", "body_center"),
        ]
    )

    include_facing_target_to_agent: bool = True
    include_facing_agent_to_target: bool = True
    # which keypoints we use to measure facing (nose & tail-base are usually enough)
    facing_keypoints: list[str] = Field(default_factory=lambda: ["nose", "tail_base"])

    include_body_axis_alignment: bool = True  # nose→tail vs nose→tail cosine
    include_nose_nose_delta: bool = True  # Δ distance between noses

    # pursuit-like feature: agent velocity projected onto agent→target LOS
    pursuit_keypoints: list[str] = Field(default_factory=lambda: ["nose"])

    # raw nose–nose distance & "close" flag
    include_nose_nose_distance: bool = False
    include_nose_nose_close_flag: bool = False
    nose_nose_close_threshold: float = (
        0.20  # same norm as GBDT: ||nose-nose|| / ||H_agent||
    )

    # --- body_center co-movement (cosine of vel. vectors)
    include_center_velocity_alignment: bool = False


class FeaturesConfigDL(BaseModel):
    bodyparts: list[str] = Field(
        default_factory=lambda: [
            "nose",
            "neck",
            "body_center",
            "tail_base",
            "hip_left",
            "hip_right",
            "lateral_left",
            "lateral_right",
            "ear_left",
            "ear_right",
            "tail_tip",
        ]
    )
    window_len_seconds: float = 3.0
    frames_in_window: int = 25

    categorical_dims: dict[str, int] = Field(
        default_factory=lambda: {
            # per-mouse metadata (used for agent & target via shared tables)
            "strain": 8,  # 10 categories (incl. None)
            "color": 4,  # 5 categories
            "sex": 2,  # 3 categories
            "age": 8,  # 13 categories
            "condition": 8,  # 16 categories
            # video-level metadata
            "lab_id": 8,  # 21 labs
            "arena_shape": 4,  # 4 shapes
            "arena_type": 4,  # 6 types (incl. None)
            "tracking_method": 4,  # 4 methods
        }
    )

    include_masks: bool = True

    single_mouse_numeric: SingleMouseNumericFeaturesConfig = Field(
        default_factory=SingleMouseNumericFeaturesConfig
    )
    pair_numeric: PairNumericFeaturesConfig = Field(
        default_factory=PairNumericFeaturesConfig
    )


class BodyWidthAugConfig(BaseModel):
    L: float = 0.8
    R: float = 1.2
    apply_prob: float = 0.5


class TimeStretchConfig(BaseModel):
    L: float = 0.8
    R: float = 1.2
    apply_prob: float = 0.7


class BodypartDropoutConfig(BaseModel):
    apply_prob: float = 0.5
    min_groups: int = 1
    max_groups: int = 2
    groups_mapping: dict[str, list[str]] = Field(
        default_factory=lambda: dict(BODYPART_GROUPS_V0)
    )
    update_masks: bool = True


class SpatialCutoutConfig(BaseModel):
    apply_prob: float = 0.2
    min_size_ratio: float = 0.2
    max_size_ratio: float = 0.6
    update_masks: bool = True
    use_interp: bool = False


# replace true label at frame t with true label at frame t+dt, there dt is random <= sec
class LabelShiftAugConfig(BaseModel):
    apply_prob: float = 0.5
    sec: float = 0.1


class AugmentationsConfig(BaseModel):
    hflip: bool = True
    flip_sides_if_flip_coords: bool = False
    random_flip_sides_agent: bool = False
    random_flip_sides_target: bool = False
    random_flip_sides_agent_masks: bool = False
    random_flip_sides_target_masks: bool = False
    flip_masks_if_flip_sides: bool = False

    time_stretch_config: TimeStretchConfig | None = None
    body_width_config: BodyWidthAugConfig | None = None
    bodypart_dropout_config: BodypartDropoutConfig | None = None
    spatial_cutout_config: SpatialCutoutConfig | None = None
    label_shift_config: LabelShiftAugConfig | None = None

    coord_jitter_std: float = 0.0


class BalanceConfig(BaseModel):
    labs_baseline: Literal["uniform", "original"] = "uniform"
    labs_mult: dict[str, float] = Field(default_factory=dict)

    # probability that you get frame with at least one active action
    # None if keep original proportions
    active_action_prob: float | None = None

    # loss weight of positive samples for each action
    # pos_w[action] = (total_negative / (total_positive + eps)) ** gamma
    # gamma=0 => original proportions
    # gamma=1 => expected contribution of positive and negative are equal
    gamma_per_action: dict[str, float] = Field(
        default_factory=lambda: {action: 0.0 for action in ACTION_NAMES_IN_TEST}
    )

    seed: int = 0


class SelfSupervised_Config(BaseModel):
    loss_weight: float = 0.1
    # future offsets in *frames* relative to center
    # e.g. 3 and 10 for windows ~25–35 frames
    future_steps: list[int] = Field(default_factory=lambda: [3, 10])


class MultiScaleTCN_Config(BaseModel):
    hidden_dim: int = 384
    branches_dilations: list[list[int]] = Field(
        default_factory=lambda: [
            [1, 2, 4, 8],  # short-term
            [1, 2, 4, 8, 16],  # mid-term
            [2, 4, 8, 16, 32],  # long-term
        ]
    )
    kernel_size: int = 3
    dropout_prob: float = 0.2
    extended_head: bool = True


class ConvTransformer_Config(BaseModel):
    hidden_dim: int = 256

    n_conv_layers: int = 1
    kernel_size: int = 5
    conv_dropout_prob: float = 0.1
    dilations: list[int] | None = None

    num_heads: int = 8
    ff_dim: int = 256 * 3
    transformer_dropout_prob: float = 0.1
    num_transformer_layers: int = 2

    extended_head: bool = False


class PoolingHeadConfig_v0(BaseModel):
    downsample_coef: float = 0.5
    dropout_prob: float = 0.3


class RankingLossConfig(BaseModel):
    margin: float = 0.3
    coef: float = 0.1


class TCN_Config(BaseModel):
    hidden_dim: int = 256
    dilations: list[int] = [1, 2, 4, 8]
    kernel_size: int = 3
    dropout_prob: float = 0.1
    extended_head: bool = False
    pooling_head_config_v0: PoolingHeadConfig_v0 | None = None
    ranking_loss_config: RankingLossConfig | None = None


class EMA_Config(BaseModel):
    enabled: bool = False
    decay: float | None = None
    update_every: int | None = None


class AWP_Config(BaseModel):
    lr: float = 0.02
    eps: float = 1e-4
    apply_to_names_with: list[str] = Field(default_factory=lambda: ["head"])
    start_ratio: float = 0.05


class FocalLoss_Config(BaseModel):
    gamma: float = 2.0


class LabelSmoothingConfig(BaseModel):
    eps: float = 1e-2


class DL_TrainConfig(BaseModel):
    data_split_config: DataSplitConfig = Field(
        default_factory=lambda: DataSplitConfig(
            num_folds=5, actions=list(ACTION_NAMES_IN_TEST)
        )
    )
    features_config: FeaturesConfigDL = Field(default_factory=FeaturesConfigDL)
    train_balance_config: BalanceConfig = Field(default_factory=BalanceConfig)

    tcn_config: TCN_Config | None = None
    multiscale_tcn_config: MultiScaleTCN_Config | None = None
    conv_transformer_config: ConvTransformer_Config | None = None

    lr: float = 3e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    train_bs: int = 32000
    eval_bs: int = 32000
    gradient_accumulation_steps: int = 1
    wd: float = 0.01
    epochs: int = 6
    max_steps: int = -1
    max_steps_to_run: int | None = None

    ema: EMA_Config = Field(default_factory=EMA_Config)
    awp: AWP_Config | None = None

    aug: AugmentationsConfig = Field(default_factory=AugmentationsConfig)
    focal: FocalLoss_Config | None = None

    label_smoothing_config: LabelSmoothingConfig | None = None

    self_supervised_config: SelfSupervised_Config | None = None

    seed: int = 0
    use_wandb: bool = True
    eval_steps: int = 300
    save_steps: int | None = None
    logging_steps: int = 10
    resume_from_latest_ckpt: bool = False

    name: str = ""

    remove_25fps_adaptable_snail_from_train: bool = False

    def cv(self) -> int:
        return self.data_split_config.test_fold

    def save_dir(self) -> Path:
        return MODELS_ROOT / self.name / f"cv{self.cv()}"

    def run_name(self) -> str:
        return f"{self.name}_cv{self.cv()}"
