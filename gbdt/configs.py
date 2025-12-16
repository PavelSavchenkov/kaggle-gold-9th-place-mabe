from pydantic import BaseModel, Field

from common.config_utils import DataSplitConfig
from gbdt.rebalance_utils import (
    DownsampleParams,
    SampleCoefsParams,
    TestDownsampleParams,
)


class SingleMouseFeaturesConfig(BaseModel):
    # O(t) = (x0, y0) - origin point at time t (usually tail_base)
    # H(t) = (h_x, h_y) - heading vector at time t (usually (ear_avg - tail_base))
    # O and H are always taken from agent mouse

    include_is_missing_keypoint: bool = False
    include_norm_in_derivatives: bool = False
    include_categorical: bool = True
    lag_windows_ms: list[float] = Field(default_factory=lambda: [100, 200, 400, 800])

    # keypoint[name](t) = p(t)

    # ---- single keypoint coordinates ----

    # p(t) translated by O(t) and scaled & rotated by H(t)
    normalised_coordinates: list[str] = Field(default_factory=lambda: [])

    # (p(t + lag) - p(t - lag)) / (2 * lag) scaled & rotated by H(t)
    normalised_velocities: list[str] = Field(default_factory=lambda: [])
    # (p(t + lag) - 2 * p(t) + p(t - lag)) / lag**2 scaled & rotated by H(t)
    normalised_accelerations: list[str] = Field(default_factory=lambda: [])

    # (p(t + lag) - p(t - lag)) / (2 * lag) scaled by H(t) norm. I.e. in raw arena coordinates, but normalised by scale
    scaled_velocities: list[str] = Field(default_factory=lambda: [])
    # (p(t + lag) - 2 * p(t) + p(t - lag)) / lag**2 scaled by H(t) norm
    scaled_accelerations: list[str] = Field(default_factory=lambda: [])

    # ---- curvatures ----

    # v(t) - velocity vector, a(t) - acceleration vector, k(t) = cross2d(v, a) / (||v||**3 + eps). Invariant to rotation and scale
    # sign(k(t)) = left/right turning
    curvatures: list[str] = Field(default_factory=lambda: [])

    # ---- pairwise distances between keypoints ----

    # (p(t) - q(t)) / ||H(t)||
    pairwise_distances: list[tuple[str, str]] = Field(default_factory=lambda: [])

    # normalised to median body length across video
    include_normalised_body_length: bool = False

    # ---- relative motions ----

    # dot(v_1(t), v_2(t)) / ||v_1|| / ||v_2||, dot product of velocities
    # cross2d(v_1(t), v_2(t)) / ||v_1|| / ||v_2||, cross product of velocities
    relative_motions: list[tuple[str, str]] = Field(default_factory=lambda: [])

    # ---- angles ----

    # radians, sin, cos
    include_heading_angle: bool = False
    # radians
    include_heading_angle_velocity: bool = False
    # radians
    include_heading_angle_acceleration: bool = False

    # radians, sin, cos. Angle between segments (p0, p1) and (q0, q1). If (q0, q1) == H, then take heading segment (O(t), H(t))
    angles: list[tuple[tuple[str, str], tuple[str, str] | str]] = Field(
        default_factory=lambda: []
    )
    include_angles_velocities: bool = False

    # ---- curvature mean over windows (from reference keypoint trajectories) ----
    # For each keypoint k in curvature_keypoints:
    #   κ_k(t) as in “curvatures”
    # For each window Δt_ms in curvature_mean_windows_msec:
    #   curv_mean_{k}_{Δt}(t) = mean( |κ_k(t)| over Δt_ms )
    curvature_keypoints: list[str] = Field(default_factory=lambda: [])
    curvature_mean_windows_msec: list[int] = Field(default_factory=lambda: [])

    # ---- turning rate (sum of absolute heading changes) ----
    # Using the same reference keypoint(s) as for curvature if desired:
    #   θ(t) = atan2( v_y(t), v_x(t) ),  turn_step(t) = |wrap_to_pi( θ(t) - θ(t-δt) )|
    # For each window Δt_ms in turning_rate_windows_msec:
    #   turn_rate{Δt}(t) = sum( turn_step over Δt_ms )
    turning_rate_windows_msec: list[int] = Field(default_factory=lambda: [])

    # ---- multiscale speed statistics (normalised, per second) ----
    # For each keypoint k in speed_keypoints:
    #   sp_k(t) = ( || P_k(t) - P_k(t-δt) || · (1000/δt_ms) ) / ||H(t)||     # body-lengths per second
    # For each window Δt_ms in speed_windows_msec:
    #   sp_m{Δt}_{k}(t) = mean( sp_k over Δt_ms )
    #   sp_s{Δt}_{k}(t) = std(  sp_k over Δt_ms )
    speed_keypoints: list[str] = Field(default_factory=lambda: [])
    speed_windows_msec: list[int] = Field(default_factory=lambda: [])

    # ---- speed ratio between two scales ----
    # For each keypoint k and each (w_fast_ms, w_slow_ms) in speed_ratio_pairs_msec:
    #   sp_ratio_{k}_{w_fast_ms,w_slow_ms}(t) = sp_m{w_fast_ms}_{k}(t) / ( sp_m{w_slow_ms}_{k}(t) + eps )
    speed_ratio_pairs_msec: list[tuple[int, int]] = Field(default_factory=lambda: [])

    # ---- long-range trends of reference point (arena coords; normalise distance-like stats) ----
    # Using C(t) = body_center(t) or O(t) + 0.5·H(t):
    # For each window Δt_ms in center_stats_windows_msec:
    #   cx_m{Δt}(t) = mean( C_x over Δt_ms ),   cy_m{Δt}(t) = mean( C_y over Δt_ms )
    #   cx_s{Δt}(t) = std(  C_x over Δt_ms ) / ||H(t)||,   cy_s{Δt}(t) = std( C_y over Δt_ms ) / ||H(t)||
    #   x_rng{Δt}(t) = (max C_x - min C_x) / ||H(t)||,     y_rng{Δt}(t) = (max C_y - min C_y) / ||H(t)||
    #   disp{Δt}(t)  = || sum ΔC || / ||H(t)||,            act{Δt}(t)   = sqrt( var(ΔC_x)+var(ΔC_y) ) / ||H(t)||
    center_stats_windows_msec: list[int] = Field(default_factory=lambda: [])

    # ---- exponentially weighted means of reference point (arena coords) ----
    # For each span τ_ms in ewm_spans_msec:
    #   x_e{τ}(t) = EWM(C_x; span=τ_ms),   y_e{τ}(t) = EWM(C_y; span=τ_ms)
    ewm_spans_msec: list[int] = Field(default_factory=lambda: [])

    # ---- rolling speed percentile ranks (normalised, per second) ----
    # Using sp_k(t) from “multiscale speed” for selected keypoints (same as speed_keypoints):
    # For each window Δt_ms in speed_percentile_windows_msec:
    #   sp_pct{Δt}_{k}(t) ∈ [0,1] = percentile rank of sp_k(t) within the last Δt_ms
    speed_percentile_windows_msec: list[int] = Field(default_factory=lambda: [])

    # ---- lagged displacements (duration-aware; self and cross) ----
    # For each keypoint k in lagged_displacement_self_keypoints and lag τ_ms in lagged_displacement_lags_msec:
    #   disp2_self_{k,τ}(t) = ( || P_k(t) - P_k(t-τ_ms) || / ||H(t)|| )^2
    # For each (k1, k2) in lagged_displacement_cross_pairs and lag τ_ms:
    #   disp2_cross_{k1,k2,τ}(t) = ( || P_{k1}(t) - P_{k2}(t-τ_ms) || / ||H(t)|| )^2
    lagged_displacement_self_keypoints: list[str] = Field(default_factory=lambda: [])
    lagged_displacement_cross_pairs: list[tuple[str, str]] = Field(
        default_factory=lambda: []
    )
    lagged_displacement_lags_msec: list[int] = Field(default_factory=lambda: [])

    # ---- ear separation with signed time offsets & consistency (normalised) ----
    # ear_d(t) = || P_ear_left(t) - P_ear_right(t) || / ||H(t)||
    # For each signed offset τ_ms in ear_distance_signed_offsets_msec:
    #   ear_o{τ}(t) = ear_d( t - τ_ms )
    # With window Δt_ms = ear_distance_consistency_window_msec:
    #   ear_con(t) = std(ear_d over Δt_ms) / ( mean(ear_d over Δt_ms) + eps )
    ear_distance_signed_offsets_msec: list[int] = Field(default_factory=lambda: [])
    ear_distance_consistency_window_msec: int | None = None


class MicePairFeaturesConfig(BaseModel):
    # O_agent(t) = agent origin at time t (e.g., tail_base)
    # H_agent(t) = agent heading vector at time t (e.g., ear_avg - tail_base)
    # All distance-like quantities below are normalised by ||H_agent(t)||.
    # All lags and window lengths are specified in milliseconds (ms) and converted to frames internally.
    # δt denotes the sampling interval (per frame) in ms.
    #
    # Notation used below:
    # - For any selected keypoints k_a (agent) and k_b (target):
    #       A(t) = agent keypoint k_a at time t
    #       B(t) = target keypoint k_b at time t
    # - Body centers (with fallback when body_center is unavailable):
    #       C_A(t) = agent body_center at time t, or O_agent(t) + 0.5 · H_agent(t)
    #       C_B(t) = target body_center at time t, or O_target(t) + 0.5 · H_target(t)

    # ---- normalised pairwise distances (per frame) ----
    # For each (k_a, k_b) in pairwise_distance_normalised:
    #   r_norm(t) = || A(t) - B(t) || / ||H_agent(t)||
    pairwise_distance_normalised: list[tuple[str, str]] = Field(
        default_factory=lambda: []
    )

    # ---- normalised approach rate (per frame; lags in ms) ----
    # For each (k_a, k_b) in pairwise_approach_rate and for each lag τ_ms in pairwise_approach_rate_lags_msec:
    #   d(t)      = || A(t) - B(t) || / ||H_agent(t)||
    #   d'(t; τ)  = ( d(t + τ_ms) - d(t - τ_ms) ) / ( 2 * τ_ms )
    # Units of d'(t; τ): 1/ms (convert to 1/s if needed)
    pairwise_approach_rate: list[tuple[str, str]] = Field(default_factory=lambda: [])
    pairwise_approach_rate_lags_msec: list[int] = Field(default_factory=lambda: [])

    # ---- normalised time to contact (per frame; in seconds) ----
    # For each (k_a, k_b) in time_to_contact_sec (uses center-to-center separation by default):
    #   d(t)          = || C_A(t) - C_B(t) || / ||H_agent(t)||
    #   closure_rate  = time derivative of d(t) (negative when closing), in 1/s
    #   ttc(t)        = min(10, d(t) / max(eps, -closure_rate))
    # Note: only meaningful when closure_rate < 0
    time_to_contact_sec: list[tuple[str, str]] = Field(default_factory=lambda: [])

    # ---- facing angles (target-to-agent) ----
    # For each target keypoint name k in facing_angles_target_to_agent:
    #   Angle between the agent’s heading segment (O_agent(t) → O_agent(t) + H_agent(t))
    #   and the segment (O_agent(t) → B_k(t)).
    #   Output: radians, sin, cos.
    facing_angles_target_to_agent: list[str] = Field(default_factory=lambda: [])

    # ---- facing angles (agent-to-target) ----
    # Same as above but roles swapped:
    #   Angle between the target’s heading segment (O_target(t) → O_target(t) + H_target(t))
    #   and the segment (O_target(t) → A_k(t)).
    #   Output: radians, sin, cos.
    facing_angles_agent_to_target: list[str] = Field(default_factory=lambda: [])

    # ======================= pairwise features (lags in milliseconds) =======================

    # ---- rolling stats of normalised pairwise distance for arbitrary keypoint pairs ----
    # For each (k_a, k_b) in distance_stats_pairs:
    #   d_norm(t)  = || A(t) - B(t) || / ||H_agent(t)||
    #   d2_norm(t) = d_norm(t)^2
    # For each window Δt_ms in distance_stats_windows_msec:
    #   d_m{Δt}(t)  = mean( d2_norm over Δt_ms )
    #   d_s{Δt}(t)  = std(  d2_norm over Δt_ms )
    #   d_mn{Δt}(t) = min(  d2_norm over Δt_ms )
    #   d_mx{Δt}(t) = max(  d2_norm over Δt_ms )
    #   int{Δt}(t)  = 1 / (1 + var(d2_norm over Δt_ms))
    distance_stats_pairs: list[tuple[str, str]] = Field(default_factory=lambda: [])
    distance_stats_windows_msec: list[int] = Field(default_factory=lambda: [])

    # ---- interaction continuity (normalised) ----
    # Using d2_norm(t) as above and window Δt_ms:
    #   int_con(t) = std(d2_norm over Δt_ms) / (mean(d2_norm over Δt_ms) + eps)
    include_interaction_continuity: bool = False
    interaction_continuity_window_msec: int | None = None

    # ---- velocity alignment with signed time offsets ----
    # Select keypoints (k_a, k_b) for velocity computation.
    # Per-sample velocities (finite differences in arena coordinates):
    #   v_A(t) = A(t) - A(t - δt)
    #   v_B(t) = B(t) - B(t - δt)
    # Alignment cosine:
    #   val(t) = ( v_A(t) · v_B(t) ) / ( ||v_A(t)|| · ||v_B(t)|| + eps )
    # For each signed offset τ_ms in velocity_alignment_offsets_msec:
    #   va_{τ}(t) = val( t - τ_ms )
    velocity_alignment_offsets_msec: list[int] = Field(default_factory=lambda: [])

    # ---- windowed stats of raw velocity dot product (co-movement) ----
    # Using the same v_A(t), v_B(t):
    #   dotv(t) = v_A(t) · v_B(t)
    # For each Δt_ms in velocity_dot_windows_msec:
    #   co_m{Δt}(t) = mean(dotv over Δt_ms)
    #   co_s{Δt}(t) = std(dotv  over Δt_ms)
    velocity_dot_windows_msec: list[int] = Field(default_factory=lambda: [])

    # ---- pursuit / escape alignment along line-of-sight (LOS) ----
    # LOS uses centers with fallback:
    #   C_A(t) = agent body_center(t) or O_agent(t) + 0.5 · H_agent(t)
    #   C_B(t) = target body_center(t) or O_target(t) + 0.5 · H_target(t)
    #   rel(t) = C_A(t) - C_B(t)
    #   r(t)   = ||rel(t)||
    #   u(t)   = rel(t) / (r(t) + eps)              # unit vector A→B
    # Using v_A(t), v_B(t) from the chosen keypoints:
    #   A_lead(t) = ( v_A(t) · u(t) )    / ( ||v_A(t)|| + eps )   # agent motion along A→B
    #   B_lead(t) = ( v_B(t) · (-u(t)) ) / ( ||v_B(t)|| + eps )   # target motion along B→A
    # For each Δt_ms in pursuit_alignment_windows_msec:
    #   A_ld{Δt}(t) = mean(A_lead over Δt_ms)
    #   B_ld{Δt}(t) = mean(B_lead over Δt_ms)
    pursuit_alignment_windows_msec: list[int] = Field(default_factory=lambda: [])

    # ---- “chase score” (normalised separation) ----
    # Using centers with fallback:
    #   r_norm(t)      = || C_A(t) - C_B(t) || / ||H_agent(t)||
    #   approach_norm  = -( r_norm(t) - r_norm(t - δt) )         # >0 when closing
    #   chase_inst(t)  = approach_norm * B_lead(t)
    # For window Δt_ms in chase_window_msec:
    #   chase_{Δt}(t)  = mean(chase_inst over Δt_ms)
    chase_window_msec: int | None = None

    # ---- speed correlation over short windows ----
    # Using v_A(t), v_B(t):
    #   s_A(t) = ||v_A(t)||,  s_B(t) = ||v_B(t)||
    # For each Δt_ms in speed_correlation_windows_msec:
    #   sp_cor{Δt}(t) = PearsonCorr( s_A, s_B over Δt_ms )
    speed_correlation_windows_msec: list[int] = Field(default_factory=lambda: [])

    # ---- nose–nose temporal dynamics (normalised) ----
    # N_A(t) = agent nose at time t;  N_B(t) = target nose at time t.
    # nn_norm(t) = || N_A(t) - N_B(t) || / ||H_agent(t)||
    # For each lag τ_ms in nose_nose_lags_msec:
    #   nn_lg{τ}(t) = nn_norm(t - τ_ms)
    #   nn_ch{τ}(t) = nn_norm(t) - nn_norm(t - τ_ms)
    #   cl_ps{τ}(t) = mean( [ nn_norm(τ') < nose_nose_close_threshold_norm ] for τ' ∈ (t-τ_ms..t) )
    nose_nose_lags_msec: list[int] = Field(default_factory=lambda: [])
    # Threshold in normalised units (e.g., fraction of body length); if None, close-proportion is skipped.
    nose_nose_close_threshold_norm: float | None = None
    include_nose_nose_close_proportion: bool = False

    # ---- body-axis alignment (nose→tail) between mice ----
    # a_A(t) = agent nose − agent tail_base
    # a_B(t) = target nose − target tail_base
    # rel_ori(t) = ( a_A(t) · a_B(t) ) / ( ||a_A(t)|| · ||a_B(t)|| + eps )   # cosine alignment (scale/rotation invariant)
    include_body_axis_alignment: bool = False


class FeaturesConfig(BaseModel):
    single_mouse_config: SingleMouseFeaturesConfig = Field(
        default_factory=lambda: SingleMouseFeaturesConfig()
    )
    mice_pair_config: MicePairFeaturesConfig = Field(
        default_factory=lambda: MicePairFeaturesConfig()
    )
    one_hot_metadata: bool = Field(default=False)
    lab_id_feature: bool = False
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
    target_fps: float = 30.0
    hflip: bool = False
    flip_sides_if_hflip: bool = False


class DataPreprocessConfig(BaseModel):
    keep_nan: bool = False


class LoggingConfig(BaseModel):
    logging_steps: int = 50
    early_stop_on_metric: str | None = None
    early_stop_patience_rounds: int = -1


class CatBoostConfig(BaseModel):
    n_estimators: int | None = None
    learning_rate: float | None = None
    task_type: str = "GPU"
    devices: str | None = "0"
    loss_function: str = "Logloss"
    od_type: str = "Iter"
    one_hot_max_size: int = 256
    nan_mode: str = "Min"
    depth: int | None = 10
    l2_leaf_reg: float | None = None
    use_best_model: bool = False
    bootstrap_type: str | None = None
    device_config: str | None = None
    model_size_reg: float | None = None
    leaf_estimation_iterations: int | None = None
    leaf_estimation_method: str | None = None
    thread_count: int | None = None
    # CatBoost accepts verbose as bool, int (period), or logging level string
    verbose: bool | int | str | None = None
    logging_level: str | None = None
    metric_period: int | None = None
    has_time: bool | None = None
    allow_const_label: bool | None = None
    classes_count: int | None = None
    random_strength: float | None = None
    name: str | None = None
    train_dir: str | None = None
    bagging_temperature: float | None = None
    boosting_type: str | None = None
    subsample: float | None = None
    sampling_unit: str | None = None
    colsample_bylevel: float | None = None
    reg_lambda: float | None = None
    objective: str | None = None
    eta: float | None = None
    max_bin: int | None = None
    scale_pos_weight: float | None = None
    gpu_cat_features_storage: str | None = None
    early_stopping_rounds: int | None = None
    grow_policy: str | None = None
    min_data_in_leaf: int | None = None
    min_child_samples: int | None = None
    max_leaves: int | None = None
    num_leaves: int | None = None
    score_function: str | None = None


class LightGBMConfig(BaseModel):
    boosting_type: str = "gbdt"
    learning_rate: float = 1e-2
    n_estimators: int = 2000
    num_leaves: int | None = None
    max_depth: int | None = None
    min_child_samples: int | None = None
    subsample: float | None = None
    subsample_freq: int | None = None
    colsample_bytree: float | None = None
    reg_alpha: float | None = None
    reg_lambda: float | None = None
    n_jobs: int = 20
    num_threads: int = 20
    importance_type: str = "gain"
    scale_pos_weight: float = 1.0
    is_enable_sparse: bool = False

    # Common core / learning-control params from generic LightGBM parameters
    max_bin: int = 63
    min_data_in_leaf: int | None = None
    min_sum_hessian_in_leaf: float | None = None
    bagging_fraction: float | None = None
    bagging_freq: int | None = None
    pos_bagging_fraction: float | None = None
    neg_bagging_fraction: float | None = None
    bagging_by_query: bool | None = None
    feature_fraction: float | None = None
    feature_fraction_bynode: float | None = None
    extra_trees: bool | None = None
    early_stopping_round: int | None = None  # a.k.a early_stopping_rounds
    early_stopping_min_delta: float | None = None
    first_metric_only: bool | None = None
    max_delta_step: float | None = None
    lambda_l1: float | None = None
    lambda_l2: float | None = None
    linear_lambda: float | None = None

    # Objective / metric details
    metric: str | list[str] | None = None
    is_unbalance: bool | None = None

    # Device / determinism / threading
    device_type: str = "gpu"  # "cpu" / "gpu" / "cuda"
    deterministic: bool | None = None
    force_col_wise: bool = True
    force_row_wise: bool | None = None
    gpu_device_id: int = 0
    gpu_use_dp: bool = False


class XGBoostConfig(BaseModel):
    # Core boosters / general
    n_estimators: int | None = None
    learning_rate: float | None = None
    objective: str | None = "binary:logistic"
    eval_metric: str | list[str] | None = None
    base_score: float | None = 0.5
    random_state: int | None = None
    n_jobs: int | None = None
    verbosity: int | None = 1

    # Tree growth / structure
    max_depth: int | None = None
    min_child_weight: float | None = None
    max_leaves: int | None = 0
    grow_policy: str | None = "depthwise"  # 'depthwise' or 'lossguide'
    max_bin: int | None = 256

    # Subsampling / column sampling
    subsample: float | None = None
    colsample_bytree: float | None = None
    colsample_bylevel: float | None = 1.0
    colsample_bynode: float | None = 1.0
    sampling_method: str | None = "uniform"  # e.g. 'uniform', 'gradient_based'

    # Regularization
    reg_alpha: float | None = 0.0  # L1
    reg_lambda: float | None = 1.0  # L2
    gamma: float | None = 0.0  # minimum loss reduction to make a split
    max_delta_step: float | None = 0.0

    # Class imbalance and constraints
    scale_pos_weight: float | None = 1.0
    monotone_constraints: dict[str, int] | None = None
    interaction_constraints: list[list[str]] | None = None

    # Hardware / algorithm
    tree_method: str | None = None  # e.g. 'hist'
    device: str | None = None  # e.g. 'cuda' or 'cpu'


class GBDT_TrainConfig(BaseModel):
    data_split_config: DataSplitConfig = Field(
        default_factory=lambda: DataSplitConfig()
    )
    features_config: FeaturesConfig = Field(default_factory=lambda: FeaturesConfig())
    train_downsample_params: DownsampleParams = Field(
        default_factory=lambda: DownsampleParams()
    )
    test_downsample_params: TestDownsampleParams = Field(
        default_factory=lambda: TestDownsampleParams()
    )
    sample_coefs_params_train: SampleCoefsParams | None = Field(default=None)
    action: str = ""
    logging_config: LoggingConfig = Field(default_factory=lambda: LoggingConfig())
    xgboost_config: XGBoostConfig | None = Field(default=None)
    lightgbm_config: LightGBMConfig | None = Field(default=None)
    catboost_config: CatBoostConfig | None = Field(default=None)

    use_wandb: bool = True

    seed: int = 0

    group: str = ""
    save_dir: str = ""
    run_name: str = ""

    def get_num_trees(self) -> int:
        if self.xgboost_config:
            return self.xgboost_config.n_estimators
        elif self.lightgbm_config:
            return self.lightgbm_config.n_estimators
        else:
            assert self.catboost_config is not None
            return self.catboost_config.n_estimators

    def set_num_trees(self, num_trees: int):
        if self.xgboost_config:
            self.xgboost_config.n_estimators = num_trees
        elif self.lightgbm_config:
            self.lightgbm_config.n_estimators = num_trees
        else:
            assert self.catboost_config is not None
            self.catboost_config.n_estimators = num_trees
