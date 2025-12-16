from gbdt.configs import MicePairFeaturesConfig, SingleMouseFeaturesConfig

from common.config_utils import DataSplitConfig, FoldsIntegerProgrammingParams, SplitByStrategy


def make_default_data_split_config(
    num_folds: int, test_fold: int, actions: list[str]
) -> DataSplitConfig:
    return DataSplitConfig(
        seed=0,
        num_folds=num_folds,
        test_fold=test_fold,
        actions=list(actions),
        split_strategy=SplitByStrategy.total_duration,
        folds_integer_programming_params=FoldsIntegerProgrammingParams(
            balance_counts=False,
            time_limit_s=5.0,
            use_warm_start=False,
            minimise_nan_cells_first=True,
        ),
    )


def make_default_single_mouse_features_config() -> SingleMouseFeaturesConfig:
    return SingleMouseFeaturesConfig(
        include_is_missing_keypoint=True,
        include_norm_in_derivatives=True,
        include_categorical=True,
        lag_windows_ms=[100, 200, 400, 800],
        # ---- normalized by O=tail_base, rotated/scaled by H=(ear_avg - tail_base) ----
        normalised_coordinates=[
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
        ],
        normalised_velocities=[
            "nose",
            "neck",
            "body_center",
            # "tail_base",
            "hip_left",
            "hip_right",
            "lateral_left",
            "lateral_right",
            "ear_left",
            "ear_right",
            "tail_tip",
        ],
        normalised_accelerations=[
            "nose",
            "neck",
            "body_center",
            # "tail_base",
            "hip_left",
            "hip_right",
            "tail_tip",
            "ear_left",
            "ear_right",
        ],
        # ---- raw arena axes, but scale-normalized by ||H|| ----
        scaled_velocities=[
            # "tail_base",
            "body_center",
            "nose",
        ],
        scaled_accelerations=[
            # "tail_base",
            "body_center",
        ],
        # ---- curvature of trajectories (rotation invariant) ----
        curvatures=["nose", "body_center"],
        # ---- pairwise distances (divided by ||H||) ----
        pairwise_distances=[
            ("tail_base", "nose"),
            ("tail_base", "neck"),
            ("tail_base", "body_center"),
            ("hip_left", "hip_right"),
            ("lateral_left", "lateral_right"),
            ("neck", "nose"),
            ("neck", "hip_left"),
            ("neck", "hip_right"),
            ("nose", "lateral_left"),
            ("nose", "lateral_right"),
            ("tail_base", "tail_tip"),
            ("body_center", "nose"),
            ("body_center", "neck"),
        ],
        include_normalised_body_length=True,
        # ---- relative motions: (dot, cross) of velocity unit vectors ----
        relative_motions=[
            ("nose", "body_center"),
            ("nose", "neck"),
            ("nose", "tail_base"),
            ("neck", "tail_base"),
            ("hip_left", "hip_right"),
            ("lateral_left", "lateral_right"),
            ("tail_tip", "tail_base"),
            ("ear_left", "ear_right"),
        ],
        # ---- heading & angles ----
        include_heading_angle=True,
        include_heading_angle_velocity=True,
        include_heading_angle_acceleration=True,
        # angle between segments; second can be "H" to mean heading (O->H)
        angles=[
            (("tail_base", "nose"), "H"),
            (("tail_base", "neck"), "H"),
            (("neck", "nose"), "H"),
            (("tail_base", "tail_tip"), "H"),
            (("hip_left", "hip_right"), "H"),
            (("lateral_left", "lateral_right"), "H"),
            (("ear_left", "ear_right"), "H"),
            (("nose", "ear_left"), ("nose", "ear_right")),  # ear splay
            (("neck", "hip_left"), ("neck", "hip_right")),  # torso bend symmetry
        ],
        include_angles_velocities=True,
    )


def make_default_mice_pair_features_config() -> MicePairFeaturesConfig:
    return MicePairFeaturesConfig(
        # Per-frame normalised distances for key pairs
        pairwise_distance_normalised=[
            ("nose", "nose"),
            ("nose", "body_center"),
            ("nose", "tail_base"),
            ("body_center", "body_center"),
            ("tail_base", "tail_base"),
        ],
        # Approach rate on nose–nose with ~10 frames at 30fps
        pairwise_approach_rate=[("nose", "nose")],
        pairwise_approach_rate_lags_msec=[333],
        # Rolling stats of center–center distance^2 over ~[5,15,30,60] frames
        distance_stats_pairs=[("body_center", "body_center")],
        distance_stats_windows_msec=[167, 500, 1000, 2000],
        # Interaction continuity over ~30 frames
        include_interaction_continuity=True,
        interaction_continuity_window_msec=1000,
        # Velocity alignment with signed offsets ~[-20,-10,0,10,20] frames
        velocity_alignment_offsets_msec=[-667, -333, 0, 333, 667],
        # Windowed dot-product stats over ~[5,15,30,60] frames
        velocity_dot_windows_msec=[167, 500, 1000, 2000],
        # Pursuit alignment windows ~[30,60] frames
        pursuit_alignment_windows_msec=[1000, 2000],
        # Chase window ~30 frames
        chase_window_msec=1000,
        # Speed correlation windows ~[60,120] frames
        speed_correlation_windows_msec=[2000, 4000],
        # Nose–nose temporal dynamics with lags ~[10,20,40] frames
        nose_nose_lags_msec=[333, 667, 1333],
        include_nose_nose_close_proportion=True,
        # Close threshold in body-length units (approximate mapping of ~10 cm)
        nose_nose_close_threshold_norm=0.35,
        # Body-axis (nose→tail) alignment between mice
        include_body_axis_alignment=True,
    )


def make_single_mouse_features_config_v2() -> SingleMouseFeaturesConfig:
    """
    Superset config: includes everything from the *old* single-mouse config
    (normalized coords/derivatives, heading & rich angles, relative motions, etc.)
    **and** all FPS-aware single-mouse features from the advanced notebook
    (curvature means, turning rate, multiscale speeds, speed ratio, center stats,
    EWM, speed percentiles, lagged displacements, ear offsets/consistency).

    30 fps → ms mapping (for intuition):
      - Center stats windows:      [5, 15, 30, 60]      → [167, 500, 1000, 2000]
      - Curvature mean windows:    [30, 60]             → [1000, 2000]
      - Turning rate window:       [30]                 → [1000]
      - Multiscale speed windows:  [10, 40, 160]        → [333, 1333, 5333]
      - Speed ratio:               10 vs 160 frames     → (333, 5333)
      - EWM spans:                 [60, 120]            → [2000, 4000]
      - Speed percentiles:         [60, 120]            → [2000, 4000]
      - Lagged displacements lags: [10, 20, 40]         → [333, 667, 1333]
      - Ear signed offsets:        [-20, -10, +10, +20] → [-667, -333, 333, 667]
    """

    # ---- Pairwise distances (union of "old" + notebook-needed) ----
    pairwise = [
        ("tail_base", "nose"),
        ("tail_base", "neck"),
        ("tail_base", "body_center"),
        ("hip_left", "hip_right"),
        ("lateral_left", "lateral_right"),
        ("neck", "nose"),
        ("neck", "hip_left"),
        ("neck", "hip_right"),
        ("nose", "lateral_left"),
        ("nose", "lateral_right"),
        ("tail_base", "tail_tip"),
        ("body_center", "nose"),
        ("body_center", "neck"),
        ("ear_left", "ear_right"),  # from notebook usage (elong denominator)
    ]

    # ---- Lagged displacement sets (notebook speed-like and NT dynamics) ----
    lagged_self = ["ear_left", "ear_right"]
    lagged_cross = [
        ("ear_left", "tail_base"),
        ("ear_right", "tail_base"),
        ("nose", "tail_base"),
    ]

    # ---- Angles (everything from old + notebook's body orientation) ----
    angles = [
        (("tail_base", "nose"), "H"),
        (("tail_base", "neck"), "H"),
        (("neck", "nose"), "H"),
        (("tail_base", "tail_tip"), "H"),
        (("hip_left", "hip_right"), "H"),
        (("lateral_left", "lateral_right"), "H"),
        (("ear_left", "ear_right"), "H"),
        (("nose", "ear_left"), ("nose", "ear_right")),  # ear splay
        (("neck", "hip_left"), ("neck", "hip_right")),  # torso bend symmetry
        # Notebook-style body orientation:
        (("body_center", "nose"), ("body_center", "tail_base")),
    ]

    return SingleMouseFeaturesConfig(
        # ---- Scalar toggles ----
        include_is_missing_keypoint=True,
        include_norm_in_derivatives=True,  # from old function
        include_categorical=True,
        # ---- Derivative lags (keep old, broadly useful) ----
        lag_windows_ms=[100, 200, 400, 800],
        # ---- Normalized (O/H) coordinates/derivatives (from old) ----
        normalised_coordinates=[
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
        ],
        normalised_velocities=[
            "nose",
            "neck",
            "body_center",
            "hip_left",
            "hip_right",
            "lateral_left",
            "lateral_right",
            "ear_left",
            "ear_right",
            "tail_tip",
        ],
        normalised_accelerations=[
            "nose",
            "neck",
            "body_center",
            "hip_left",
            "hip_right",
            "tail_tip",
            "ear_left",
            "ear_right",
        ],
        # ---- Raw arena axes but scale-normalized (from old) ----
        scaled_velocities=["body_center", "nose"],
        scaled_accelerations=["body_center"],
        # ---- Curvature per-keypoint (from old) ----
        curvatures=["nose", "body_center"],
        # ---- Pairwise distances normalized by ||H|| (union) ----
        pairwise_distances=pairwise,
        include_normalised_body_length=True,  # from old
        # ---- Relative motions (from old) ----
        relative_motions=[
            ("nose", "body_center"),
            ("nose", "neck"),
            ("nose", "tail_base"),
            ("neck", "tail_base"),
            ("hip_left", "hip_right"),
            ("lateral_left", "lateral_right"),
            ("tail_tip", "tail_base"),
            ("ear_left", "ear_right"),
        ],
        # ---- Heading & angles (from old) + notebook body orientation angle above ----
        include_heading_angle=True,
        include_heading_angle_velocity=True,
        include_heading_angle_acceleration=True,
        angles=angles,
        include_angles_velocities=True,
        # ---- Notebook curvature means & turning rate ----
        curvature_keypoints=["body_center"],
        curvature_mean_windows_msec=[1000, 2000],
        turning_rate_windows_msec=[1000],
        # ---- Notebook multiscale speeds (+ ratio) ----
        speed_keypoints=["body_center"],
        speed_windows_msec=[333, 1333, 5333],
        speed_ratio_pairs_msec=[(333, 5333)],
        # ---- Notebook center rolling stats ----
        center_stats_windows_msec=[167, 500, 1000, 2000],
        # ---- Notebook EWM spans ----
        ewm_spans_msec=[2000, 4000],
        # ---- Notebook speed percentile ranks ----
        speed_percentile_windows_msec=[2000, 4000],
        # ---- Notebook lagged displacements (self/cross) ----
        lagged_displacement_self_keypoints=lagged_self,
        lagged_displacement_cross_pairs=lagged_cross,
        lagged_displacement_lags_msec=[333, 667, 1333],
        # ---- Notebook ear separation temporal offsets & consistency ----
        ear_distance_signed_offsets_msec=[-667, -333, 333, 667],
        ear_distance_consistency_window_msec=1000,
    )


def make_mice_pair_features_config_v2() -> MicePairFeaturesConfig:
    """
    Importance-driven pair config.
    Focuses on: pairwise distances (nose↔tail_base, nose↔nose, tail_base↔tail_base, nose↔body_center, body_center↔body_center),
    nose–nose temporal dynamics (333/667/1333 ms + close proportion), velocity dot-product stats
    (167/500/1000/2000 ms), pursuit (1000/2000 ms), chase (1000 ms), speed correlation (2000/4000 ms),
    velocity alignment offsets (-667/-333/0/333/667 ms), distance^2 windows for center–center, and
    interaction continuity (1000 ms).
    """
    return MicePairFeaturesConfig(
        # Distances that ranked highest
        pairwise_distance_normalised=[
            ("nose", "tail_base"),
            ("nose", "nose"),
            ("tail_base", "tail_base"),
            ("nose", "body_center"),
            ("body_center", "body_center"),
        ],
        # Approach rate on nose–nose @ 333 ms (top signal)
        pairwise_approach_rate=[("nose", "nose")],
        pairwise_approach_rate_lags_msec=[333],
        # Center–center distance^2 stats windows seen in importances
        distance_stats_pairs=[("body_center", "body_center")],
        distance_stats_windows_msec=[167, 500, 1000, 2000],
        # Interaction continuity over ~1 s
        include_interaction_continuity=True,
        interaction_continuity_window_msec=1000,
        # Velocity alignment with signed offsets used in the rankings
        velocity_alignment_offsets_msec=[-667, -333, 0, 333, 667],
        # Velocity dot-product (co-movement) windows that appeared (mean/std)
        velocity_dot_windows_msec=[167, 500, 1000, 2000],
        # Pursuit alignment windows that scored well
        pursuit_alignment_windows_msec=[1000, 2000],
        # Chase score over ~1 s
        chase_window_msec=1000,
        # Speed correlation windows (notably 4000 ms showed up)
        speed_correlation_windows_msec=[2000, 4000],
        # Nose–nose temporal dynamics: lags + close-proportion
        nose_nose_lags_msec=[333, 667, 1333],
        include_nose_nose_close_proportion=True,
        nose_nose_close_threshold_norm=0.35,
        # Body-axis alignment between mice (present in importances)
        include_body_axis_alignment=True,
    )


def make_single_mouse_features_config_v3() -> SingleMouseFeaturesConfig:
    """
    Importance-driven single-mouse config.
    Keeps all coordinates/derivatives for keypoints that repeatedly appeared (nose, neck, body_center, tail_base,
    hips, laterals, ears, tail_tip), categorical encodings (strain/sex/age/color/condition), the high-signal
    within-mouse distances, relative motions, and rich angles incl. velocities. Lags match 100/200/400/800 ms.
    """
    # Within-mouse distances that repeatedly mattered
    pairwise = [
        ("tail_base", "nose"),
        ("tail_base", "neck"),
        ("tail_base", "body_center"),
        ("hip_left", "hip_right"),
        ("lateral_left", "lateral_right"),
        ("neck", "nose"),
        ("neck", "hip_left"),
        ("neck", "hip_right"),
        ("nose", "lateral_left"),
        ("nose", "lateral_right"),
        ("tail_base", "tail_tip"),
        ("body_center", "nose"),
        ("body_center", "neck"),
        ("ear_left", "ear_right"),
    ]

    # Relative-motion pairs that showed up often
    relmot = [
        ("nose", "body_center"),
        ("nose", "neck"),
        ("nose", "tail_base"),
        ("neck", "tail_base"),
        ("hip_left", "hip_right"),
        ("lateral_left", "lateral_right"),
        ("tail_tip", "tail_base"),
        ("ear_left", "ear_right"),
    ]

    # Angles that map to the top angle features in the list
    angles = [
        (("tail_base", "nose"), "H"),
        (("tail_base", "neck"), "H"),
        (("neck", "nose"), "H"),
        (("tail_base", "tail_tip"), "H"),
        (("hip_left", "hip_right"), "H"),
        (("lateral_left", "lateral_right"), "H"),
        (("ear_left", "ear_right"), "H"),
        (("nose", "ear_left"), ("nose", "ear_right")),  # ear splay
        (("neck", "hip_left"), ("neck", "hip_right")),  # torso symmetry
        (
            ("body_center", "nose"),
            ("body_center", "tail_base"),
        ),  # body axis orientation
    ]

    return SingleMouseFeaturesConfig(
        include_is_missing_keypoint=True,
        include_norm_in_derivatives=True,
        include_categorical=True,
        # Lags that dominate the importances
        lag_windows_ms=[100, 200, 400, 800],
        # Normalised coordinates/derivatives for all salient keypoints
        normalised_coordinates=[
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
        ],
        normalised_velocities=[
            "nose",
            "neck",
            "body_center",
            "hip_left",
            "hip_right",
            "lateral_left",
            "lateral_right",
            "ear_left",
            "ear_right",
            "tail_tip",
        ],
        normalised_accelerations=[
            "nose",
            "neck",
            "body_center",
            "hip_left",
            "hip_right",
            "tail_tip",
            "ear_left",
            "ear_right",
        ],
        # Scaled (arena) kinematics that ranked high for nose/body_center
        scaled_velocities=["body_center", "nose"],
        scaled_accelerations=["body_center"],
        # Curvatures (nose/body_center featured)
        curvatures=["nose", "body_center"],
        # Within-mouse distances + body length
        pairwise_distances=pairwise,
        include_normalised_body_length=True,
        # Relative motions that were important
        relative_motions=relmot,
        # Heading & angles (+ velocities/accelerations)
        include_heading_angle=True,
        include_heading_angle_velocity=True,
        include_heading_angle_acceleration=True,
        angles=angles,
        include_angles_velocities=True,
        # Rolling curvature means & turning rate around ~1–2 s
        curvature_keypoints=["body_center"],
        curvature_mean_windows_msec=[1000, 2000],
        turning_rate_windows_msec=[1000],
        # Multiscale speeds (+ ratio) used elsewhere in your stack
        speed_keypoints=["body_center"],
        speed_windows_msec=[333, 1333, 5333],
        speed_ratio_pairs_msec=[(333, 5333)],
        # Center rolling stats (167/500/1000/2000 ms)
        center_stats_windows_msec=[167, 500, 1000, 2000],
        # EWM & speed percentiles (2–4 s)
        ewm_spans_msec=[2000, 4000],
        speed_percentile_windows_msec=[2000, 4000],
        # Lagged displacement lags (333/667/1333 ms) for ears + cross with tail_base/nose
        lagged_displacement_self_keypoints=["ear_left", "ear_right"],
        lagged_displacement_cross_pairs=[
            ("ear_left", "tail_base"),
            ("ear_right", "tail_base"),
            ("nose", "tail_base"),
        ],
        lagged_displacement_lags_msec=[333, 667, 1333],
        # Ear separation temporal offsets & consistency (1 s)
        ear_distance_signed_offsets_msec=[-667, -333, 333, 667],
        ear_distance_consistency_window_msec=1000,
    )


def make_mice_pair_features_config_v3() -> MicePairFeaturesConfig:
    """
    Importance-driven pair config (v3).

    Kept (very strong):
      • Pair distances: nose↔tail_base, nose↔nose, tail_base↔tail_base, nose↔body_center, body_center↔body_center
      • Nose–nose temporal dynamics: lags 333/667/1333 ms + close-proportion
      • Velocity dot-product windows: 167/500/1000/2000 ms (mean/std)
      • Pursuit (Alead/Blead) windows: 1000/2000 ms; Chase: 1000 ms
      • Speed correlation: 2000/4000 ms
      • Velocity alignment offsets: −667/−333/0/+333/+667 ms
      • Center–center d² stats windows: 167/500/1000/2000 ms
      • Interaction continuity (≈1 s)
      • Body-axis alignment

    Added (likely helpful, cheap):
      • Time-to-contact (center–center)
      • Facing angles (nose & body_center, both directions)
    """
    return MicePairFeaturesConfig(
        # Top pairwise separations
        pairwise_distance_normalised=[
            ("nose", "tail_base"),
            ("nose", "nose"),
            ("tail_base", "tail_base"),
            ("nose", "body_center"),
            ("body_center", "body_center"),
        ],
        # Approach on nose–nose; 333 ms was strongest but 667 also showed signal
        pairwise_approach_rate=[("nose", "nose")],
        pairwise_approach_rate_lags_msec=[333, 667],
        # Distance^2 rolling stats for center–center (min/mean/std/max/int picked up in importances)
        distance_stats_pairs=[("body_center", "body_center")],
        distance_stats_windows_msec=[167, 500, 1000, 2000],
        # Interaction continuity over ~1 s
        include_interaction_continuity=True,
        interaction_continuity_window_msec=1000,
        # Signed velocity alignment offsets (all five offsets appeared)
        velocity_alignment_offsets_msec=[-667, -333, 0, 333, 667],
        # Co-movement (dot) windows (mean/std variants were ranked)
        velocity_dot_windows_msec=[167, 500, 1000, 2000],
        # Pursuit alignment and chase
        pursuit_alignment_windows_msec=[1000, 2000],
        chase_window_msec=1000,
        # Speed correlation (2–4 s)
        speed_correlation_windows_msec=[2000, 4000],
        # Nose–nose dynamics (lags & close-proportion)
        nose_nose_lags_msec=[333, 667, 1333],
        include_nose_nose_close_proportion=True,
        nose_nose_close_threshold_norm=0.35,
        # Orientation coupling: simple and cheap
        time_to_contact_sec=[("body_center", "body_center")],
        facing_angles_target_to_agent=["nose", "body_center"],
        facing_angles_agent_to_target=["nose", "body_center"],
        # Body-axis alignment was explicitly important
        include_body_axis_alignment=True,
    )


def make_single_mouse_features_config_v4() -> SingleMouseFeaturesConfig:
    """
    Importance-driven single-mouse config (v4).

    Kept (strong):
      • Categorical (condition/sex/strain/age/color)
      • Normalised coords with emphasis on ears, nose, neck, hips, tail_base, body_center
      • Normalised v/a for nose, neck, body_center, ears, tail_tip, hips
      • Scaled v/a for body_center & nose
      • Pairwise distances: tail_base↔{nose,neck,body_center,tail_tip}, hip_left↔hip_right, lateral_* pairs,
        neck↔{nose,hip_left,hip_right}, body_center↔{nose,neck}
      • Relative motions (dot/cross): nose/body_center/neck/tail_base; hips; ears; tail_tip↔tail_base
      • Heading angle, d/dt, d²/dt²
      • Angles vs H and ear-splay; include angle velocities

    Trimmed:
      • Lateral_* derivatives (kept their coordinates/distances but removed their v/a to reduce noise/width)

    Added (targeted, cheap):
      • Nose curvature + short-window curvature means (400/800 ms)
      • Turning-rate summaries (1000/2000 ms)
      • Simple multiscale speed stats (nose/body_center; 1–2 s)
      • Ear-distance stability (consistency over 1 s; signed offsets)
    """
    return SingleMouseFeaturesConfig(
        include_is_missing_keypoint=True,
        include_norm_in_derivatives=True,
        include_categorical=True,
        lag_windows_ms=[100, 200, 400, 800],
        # ---- O=tail_base; rotate/scale by H=(ear_avg - tail_base) ----
        normalised_coordinates=[
            "nose",
            "neck",
            "body_center",
            "tail_base",
            "hip_left",
            "hip_right",
            "lateral_left",
            "lateral_right",  # keep coords (x features had signal)
            "ear_left",
            "ear_right",
            "tail_tip",
        ],
        # Derivatives: drop lateral_* to trim low-yield families
        normalised_velocities=[
            "nose",
            "neck",
            "body_center",
            "hip_left",
            "hip_right",
            "ear_left",
            "ear_right",
            "tail_tip",
        ],
        normalised_accelerations=[
            "nose",
            "neck",
            "body_center",
            "hip_left",
            "hip_right",
            "tail_tip",
            "ear_left",
            "ear_right",
        ],
        # Arena-axis but scale-normalised
        scaled_velocities=["body_center", "nose"],
        scaled_accelerations=["body_center"],
        # Curvature: nose dominated; also add windowed means
        curvatures=["nose"],
        # Pairwise distances (all showed signal)
        pairwise_distances=[
            ("tail_base", "nose"),
            ("tail_base", "neck"),
            ("tail_base", "body_center"),
            ("hip_left", "hip_right"),
            ("lateral_left", "lateral_right"),
            ("neck", "nose"),
            ("neck", "hip_left"),
            ("neck", "hip_right"),
            ("nose", "lateral_left"),
            ("nose", "lateral_right"),
            ("tail_base", "tail_tip"),
            ("body_center", "nose"),
            ("body_center", "neck"),
        ],
        include_normalised_body_length=True,
        # Relative motions (dot/cross of unit velocities)
        relative_motions=[
            ("nose", "body_center"),
            ("nose", "neck"),
            ("nose", "tail_base"),
            ("neck", "tail_base"),
            ("hip_left", "hip_right"),
            ("lateral_left", "lateral_right"),
            ("tail_tip", "tail_base"),
            ("ear_left", "ear_right"),
        ],
        # Heading & angles
        include_heading_angle=True,
        include_heading_angle_velocity=True,
        include_heading_angle_acceleration=True,
        angles=[
            (("tail_base", "nose"), "H"),
            (("tail_base", "neck"), "H"),
            (("neck", "nose"), "H"),
            (("tail_base", "tail_tip"), "H"),
            (("hip_left", "hip_right"), "H"),
            (("lateral_left", "lateral_right"), "H"),
            (("ear_left", "ear_right"), "H"),
            (("nose", "ear_left"), ("nose", "ear_right")),  # ear splay
            (("neck", "hip_left"), ("neck", "hip_right")),  # torso bend symmetry
        ],
        include_angles_velocities=True,
        # —— Additions informed by importances ——
        curvature_keypoints=["nose"],
        curvature_mean_windows_msec=[400, 800],
        turning_rate_windows_msec=[1000, 2000],
        speed_keypoints=["nose", "body_center"],
        speed_windows_msec=[1000, 2000],
        speed_ratio_pairs_msec=[(1000, 2000)],  # simple stability ratio
        # Ear separation stability & signed offsets (ear-splay was important)
        ear_distance_signed_offsets_msec=[-400, 0, 400],
        ear_distance_consistency_window_msec=1000,
    )


def make_mice_pair_features_config_v4() -> MicePairFeaturesConfig:
    """
    Importance-driven pair config (v4).

    Highlights kept:
      • Distances: nose↔tail_base, tail_base↔tail_base, nose↔nose, nose↔body_center, body_center↔body_center
      • Nose–nose temporal: lags 333/667/1333 ms + close-proportion (333 & 1333 especially)
      • Co-movement (dot) windows: 167/500/1000/2000 ms (std/mean)
      • Pursuit: Alead/Blead @ 2000 ms (also 1000 ms); Chase @ 1000 ms
      • Speed correlation: 2000/4000 ms
      • Velocity alignment offsets: −667/−333/0/+333/+667 ms
      • Center–center d² stats windows: 167/500/1000/2000 ms (min/mean/std/max/int)
      • Interaction continuity (~1 s)
      • Body-axis alignment

    Trim/Focus changes (from v3):
      • Facing angles narrowed to the two strongest directions:
          - a2t: nose
          - t2a: body_center
      • Approach-rate lags focused to 333 & 667 ms (top in importances)
      • Kept TTC(center–center) due to positive signal.
    """
    return MicePairFeaturesConfig(
        # Per-frame normalised separations (very strong)
        pairwise_distance_normalised=[
            ("nose", "tail_base"),
            ("tail_base", "tail_base"),
            ("nose", "nose"),
            ("nose", "body_center"),
            ("body_center", "body_center"),
        ],
        # Approach rate on nose–nose (333/667 best)
        pairwise_approach_rate=[("nose", "nose")],
        pairwise_approach_rate_lags_msec=[333, 667],
        # Center–center d^2 rolling stats (min/mean/std/max/int)
        distance_stats_pairs=[("body_center", "body_center")],
        distance_stats_windows_msec=[167, 500, 1000, 2000],
        # Interaction continuity (~1 s)
        include_interaction_continuity=True,
        interaction_continuity_window_msec=1000,
        # Signed velocity alignment offsets (all five had signal)
        velocity_alignment_offsets_msec=[-667, -333, 0, 333, 667],
        # Raw velocity dot-product windows (std/mean)
        velocity_dot_windows_msec=[167, 500, 1000, 2000],
        # Pursuit/escape alignment + chase
        pursuit_alignment_windows_msec=[1000, 2000],
        chase_window_msec=1000,
        # Speed correlation
        speed_correlation_windows_msec=[2000, 4000],
        # Nose–nose lags (also exposes nn_change) + close proportion
        nose_nose_lags_msec=[333, 667, 1333],
        include_nose_nose_close_proportion=True,
        nose_nose_close_threshold_norm=0.35,
        # TTC was helpful (center–center)
        time_to_contact_sec=[("body_center", "body_center")],
        # Facing angles: strongest directions only
        facing_angles_agent_to_target=["nose"],  # a2t_nose_{cos/sin/θ}
        facing_angles_target_to_agent=["body_center"],  # t2a_body_center_{cos/sin/θ}
        # Orientation alignment between body axes
        include_body_axis_alignment=True,
    )


def make_single_mouse_features_config_v5() -> SingleMouseFeaturesConfig:
    """
    Importance-driven single-mouse config (v5).

    Kept:
      • Categorical (condition/sex/strain/age/color), is_missing flags
      • Normalised coordinates for tail_base, ears, nose, neck, hips, body_center, tail_tip, laterals
      • Normalised v/a for nose/neck/body_center/ears/hips/tail_tip
      • Scaled v/a: body_center (a), body_center & nose (v)
      • Pairwise distances: tail_base↔{nose,neck,body_center,tail_tip}, hip_left↔hip_right,
        lateral_left↔lateral_right, neck↔{nose,hip_left,hip_right}, body_center↔{nose,neck}
      • Relative motions (dot/cross) for core pairs including ears and hips
      • Heading angle (+ velocity & acceleration)
      • Angles to H and ear-splay (+ angle velocities)

    Focused additions (all showed strong signal):
      • Nose curvature means @ 400/800 ms
      • Turning rate windows @ 1000/2000 ms
      • Speed stats for nose & body_center @ 1000/2000 ms + ratio(1000/2000)
      • Ear-distance: signed offsets [−400, 0, +400] and 1 s consistency
    """
    return SingleMouseFeaturesConfig(
        include_is_missing_keypoint=True,
        include_norm_in_derivatives=True,
        include_categorical=True,
        lag_windows_ms=[100, 200, 400, 800],
        # ---- normalised by O=tail_base, rotated/scaled by H=(ear_avg - tail_base) ----
        normalised_coordinates=[
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
        ],
        # Derivatives: keep high-signal parts; still omit lateral_* derivatives to trim width
        normalised_velocities=[
            "nose",
            "neck",
            "body_center",
            "hip_left",
            "hip_right",
            "ear_left",
            "ear_right",
            "tail_tip",
        ],
        normalised_accelerations=[
            "nose",
            "neck",
            "body_center",
            "hip_left",
            "hip_right",
            "tail_tip",
            "ear_left",
            "ear_right",
        ],
        # Arena-axis but scale-normalised
        scaled_velocities=["body_center", "nose"],
        scaled_accelerations=["body_center"],
        # Curvature (instantaneous) + windowed means (nose dominates)
        curvatures=["nose"],
        curvature_keypoints=["nose"],
        curvature_mean_windows_msec=[400, 800],
        # Pairwise distances (normalised by ||H||)
        pairwise_distances=[
            ("tail_base", "nose"),
            ("tail_base", "neck"),
            ("tail_base", "body_center"),
            ("tail_base", "tail_tip"),
            ("hip_left", "hip_right"),
            ("lateral_left", "lateral_right"),
            ("neck", "nose"),
            ("neck", "hip_left"),
            ("neck", "hip_right"),
            ("body_center", "nose"),
            ("body_center", "neck"),
        ],
        include_normalised_body_length=True,
        # Relative motions (dot/cross of unit velocities)
        relative_motions=[
            ("nose", "body_center"),
            ("nose", "neck"),
            ("nose", "tail_base"),
            ("neck", "tail_base"),
            ("hip_left", "hip_right"),
            ("lateral_left", "lateral_right"),
            ("tail_tip", "tail_base"),
            ("ear_left", "ear_right"),
        ],
        # Heading & angles (and their velocities)
        include_heading_angle=True,
        include_heading_angle_velocity=True,
        include_heading_angle_acceleration=True,
        angles=[
            (("tail_base", "nose"), "H"),
            (("tail_base", "neck"), "H"),
            (("neck", "nose"), "H"),
            (("tail_base", "tail_tip"), "H"),
            (("hip_left", "hip_right"), "H"),
            (("lateral_left", "lateral_right"), "H"),
            (("ear_left", "ear_right"), "H"),
            (("nose", "ear_left"), ("nose", "ear_right")),  # ear splay
            (("neck", "hip_left"), ("neck", "hip_right")),  # torso bend symmetry
        ],
        include_angles_velocities=True,
        # Turning rate summaries (nose path)
        turning_rate_windows_msec=[1000, 2000],
        # Multiscale speed (per second; body-length normalised)
        speed_keypoints=["nose", "body_center"],
        speed_windows_msec=[1000, 2000],
        speed_ratio_pairs_msec=[(1000, 2000)],
        # Ear separation stability & signed offsets (strong signal)
        ear_distance_signed_offsets_msec=[-400, 0, 400],
        ear_distance_consistency_window_msec=1000,
    )


def make_mice_pair_features_config_v5() -> MicePairFeaturesConfig:
    """
    Competition-tuned pair config (v5).

    Design choices:
      • Multi-scale temporal: keep 167/500/1000/2000ms for co-movement; add 1000ms to speedcorr.
      • Approach-rate on nose→nose *and* nose→body_center (short lags 333/667ms).
      • Facing symmetry: use both nose and body_center for a2t and t2a to capture reciprocal bouts.
      • Keep strong distances (nose↔tail_base, tail_base↔tail_base, nose↔nose, nose↔center, center↔center).
      • Keep nn_closeprop at 333/667/1333ms; retain TTC(center–center), body-axis alignment, interaction continuity.
    """
    return MicePairFeaturesConfig(
        # Strong per-frame separations
        pairwise_distance_normalised=[
            ("nose", "tail_base"),
            ("tail_base", "tail_base"),
            ("nose", "nose"),
            ("nose", "body_center"),
            ("body_center", "body_center"),
        ],
        # Approach rate channels (short-lag, social proximity)
        pairwise_approach_rate=[("nose", "nose"), ("nose", "body_center")],
        pairwise_approach_rate_lags_msec=[333, 667],
        # Center–center d^2 stats (broad context)
        distance_stats_pairs=[("body_center", "body_center")],
        distance_stats_windows_msec=[167, 500, 1000, 2000],
        # Bout continuity
        include_interaction_continuity=True,
        interaction_continuity_window_msec=1000,
        # Velocity alignment offsets (keep full 5-point ring)
        velocity_alignment_offsets_msec=[-667, -333, 0, 333, 667],
        # Co-movement windows (std/mean)
        velocity_dot_windows_msec=[167, 500, 1000, 2000],
        # Pursuit / chase
        pursuit_alignment_windows_msec=[1000, 2000],
        chase_window_msec=1000,
        # Speed correlation (add 1000ms)
        speed_correlation_windows_msec=[1000, 2000, 4000],
        # Nose–nose temporal + close proportion (multi-scale)
        nose_nose_lags_msec=[333, 667, 1333],
        include_nose_nose_close_proportion=True,
        # nose_nose_close_windows_msec=[333, 667, 1333],
        nose_nose_close_threshold_norm=0.38,  # a tad stricter than v4
        # TTC on center–center
        time_to_contact_sec=[("body_center", "body_center")],
        # Facing symmetry (helps reciprocal actions)
        facing_angles_agent_to_target=["nose", "body_center"],
        facing_angles_target_to_agent=["nose", "body_center"],
        # Body-axis alignment
        include_body_axis_alignment=True,
    )


def make_single_mouse_features_config_v6() -> SingleMouseFeaturesConfig:
    """
    Competition-tuned single-mouse config (v6).

    Design choices:
      • Keep rich normalised kinematics; re-introduce lateral_* derivatives (sideways posture helps social acts).
      • Add short-scale (400ms) speed for nose/body_center; keep 1000/2000 and the 1000/2000 ratio.
      • Expand turning summaries to [400, 1000, 2000] to catch head flicks vs sustained turns.
      • Keep ear-distance stability (1s) with signed offsets; retain strong angle families and relative motions.
      • Keep nose curvature means (400/800ms); add curvature for neck (posture cue).
    """
    return SingleMouseFeaturesConfig(
        include_is_missing_keypoint=True,
        include_norm_in_derivatives=True,
        include_categorical=True,
        lag_windows_ms=[100, 200, 400, 800],
        # Normalised coords (H-frame) — broad set
        normalised_coordinates=[
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
        ],
        # Derivatives: include laterals for sideways posture/maneuvers
        normalised_velocities=[
            "nose",
            "neck",
            "body_center",
            "hip_left",
            "hip_right",
            "ear_left",
            "ear_right",
            "tail_tip",
            "lateral_left",
            "lateral_right",
        ],
        normalised_accelerations=[
            "nose",
            "neck",
            "body_center",
            "hip_left",
            "hip_right",
            "tail_tip",
            "ear_left",
            "ear_right",
            "lateral_left",
            "lateral_right",
        ],
        # Arena-axis but length-normalised
        scaled_velocities=["body_center", "nose"],
        scaled_accelerations=["body_center"],
        # Curvature cues
        curvatures=["nose", "neck"],
        curvature_keypoints=["nose", "neck"],
        curvature_mean_windows_msec=[400, 800],
        # Pairwise distances (H-normalised)
        pairwise_distances=[
            ("tail_base", "nose"),
            ("tail_base", "neck"),
            ("tail_base", "body_center"),
            ("tail_base", "tail_tip"),
            ("hip_left", "hip_right"),
            ("lateral_left", "lateral_right"),
            ("neck", "nose"),
            ("neck", "hip_left"),
            ("neck", "hip_right"),
            ("body_center", "nose"),
            ("body_center", "neck"),
        ],
        include_normalised_body_length=True,
        # Relative motions (unit-vel dot/cross)
        relative_motions=[
            ("nose", "body_center"),
            ("nose", "neck"),
            ("nose", "tail_base"),
            ("neck", "tail_base"),
            ("hip_left", "hip_right"),
            ("lateral_left", "lateral_right"),
            ("tail_tip", "tail_base"),
            ("ear_left", "ear_right"),
        ],
        # Heading & angles (with velocities)
        include_heading_angle=True,
        include_heading_angle_velocity=True,
        include_heading_angle_acceleration=True,
        angles=[
            (("tail_base", "nose"), "H"),
            (("tail_base", "neck"), "H"),
            (("neck", "nose"), "H"),
            (("tail_base", "tail_tip"), "H"),
            (("hip_left", "hip_right"), "H"),
            (("lateral_left", "lateral_right"), "H"),
            (("ear_left", "ear_right"), "H"),
            (("nose", "ear_left"), ("nose", "ear_right")),  # ear splay
            (("neck", "hip_left"), ("neck", "hip_right")),  # torso bend symmetry
        ],
        include_angles_velocities=True,
        # Turning summaries: short + medium + long
        turning_rate_windows_msec=[400, 1000, 2000],
        # Multi-scale speeds (per-second, body-length norm)
        speed_keypoints=["nose", "body_center"],
        speed_windows_msec=[400, 1000, 2000],
        speed_ratio_pairs_msec=[(1000, 2000)],
        # Ear separation stability & signed offsets
        ear_distance_signed_offsets_msec=[-400, 0, 400],
        ear_distance_consistency_window_msec=1000,
    )


def make_single_mouse_features_config_v7() -> SingleMouseFeaturesConfig:
    """
    Action-oriented, lightweight add-on to your latest v3 single-mouse config.

    Keeps all v3 features and ADDS:
      - Turning-rate @ 500 ms (sharper head movement: selfgroom/rear/avoid).
      - Short-horizon speed @ 167 ms (sprints/escape/run onsets).
      - Extra speed ratios without extra window cost (ratios reuse existing means).
      - Extra EWM @ 1000 ms (faster trend for short actions).
    """
    # --- Base: v3 selections (kept) ---
    normalised_coordinates = [
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
    normalised_velocities = [
        "nose",
        "neck",
        "body_center",
        "ear_left",
        "ear_right",
        "tail_tip",
    ]
    normalised_accelerations = [
        "nose",
        "neck",
        "body_center",
        "ear_left",
        "ear_right",
        "tail_tip",
    ]
    pairwise = [
        ("tail_base", "nose"),
        ("tail_base", "neck"),
        ("tail_base", "body_center"),
        ("hip_left", "hip_right"),
        ("lateral_left", "lateral_right"),
        ("neck", "nose"),
        ("neck", "hip_left"),
        ("neck", "hip_right"),
        ("nose", "lateral_left"),
        ("nose", "lateral_right"),
        ("tail_base", "tail_tip"),
        ("body_center", "nose"),
        ("body_center", "neck"),
        ("ear_left", "ear_right"),
    ]
    relmot = [
        ("nose", "tail_base"),
        ("nose", "neck"),
        ("nose", "body_center"),
        ("neck", "tail_base"),
        ("hip_left", "hip_right"),
        ("lateral_left", "lateral_right"),
        ("tail_tip", "tail_base"),
        ("ear_left", "ear_right"),
    ]
    angles = [
        (("tail_base", "nose"), "H"),
        (("tail_base", "neck"), "H"),
        (("neck", "nose"), "H"),
        (("tail_base", "tail_tip"), "H"),
        (("hip_left", "hip_right"), "H"),
        (("lateral_left", "lateral_right"), "H"),
        (("ear_left", "ear_right"), "H"),
        (("nose", "ear_left"), ("nose", "ear_right")),  # ear splay
        (("neck", "hip_left"), ("neck", "hip_right")),  # torso symmetry
        (
            ("body_center", "nose"),
            ("body_center", "tail_base"),
        ),  # body axis orientation
    ]

    # --- Add-ons (lightweight) ---
    # Turning rate: add 500 ms (keeps 1000 ms too)
    turning_windows = [500, 1000]
    # Speed: add 167 ms (keeps 333/1333/5333)
    speed_windows = [167, 333, 1333, 5333]
    # Reuse existing speed means for extra ratios (no extra rolling cost)
    speed_ratio_pairs = [(167, 1333), (333, 1333), (333, 5333)]
    # EWM: add 1000 ms (keeps 2000/4000)
    ewm_spans = [1000, 2000, 4000]

    return SingleMouseFeaturesConfig(
        include_is_missing_keypoint=True,
        include_norm_in_derivatives=True,
        include_categorical=True,
        lag_windows_ms=[100, 200, 400, 800],
        normalised_coordinates=normalised_coordinates,
        normalised_velocities=normalised_velocities,
        normalised_accelerations=normalised_accelerations,
        scaled_velocities=["body_center", "nose"],
        scaled_accelerations=["body_center"],
        curvatures=["nose", "body_center"],
        pairwise_distances=pairwise,
        include_normalised_body_length=True,
        relative_motions=relmot,
        include_heading_angle=True,
        include_heading_angle_velocity=True,
        include_heading_angle_acceleration=False,  # keep off for speed
        angles=angles,
        include_angles_velocities=False,  # keep off for speed
        curvature_keypoints=["body_center"],
        curvature_mean_windows_msec=[1000, 2000],  # add back 1000 (cheap via cumsum)
        turning_rate_windows_msec=turning_windows,
        speed_keypoints=["body_center"],
        speed_windows_msec=speed_windows,
        speed_ratio_pairs_msec=speed_ratio_pairs,
        center_stats_windows_msec=[2000],  # keep single long window (cheap enough)
        # ewm_spans_msec=ewm_spans,
        speed_percentile_windows_msec=[2000],  # keep single (cheaper)
        lagged_displacement_self_keypoints=["ear_left", "ear_right"],
        lagged_displacement_cross_pairs=[
            ("ear_left", "tail_base"),
            ("ear_right", "tail_base"),
            ("nose", "tail_base"),
        ],
        lagged_displacement_lags_msec=[333, 1333],  # keep 2 lags only
        ear_distance_signed_offsets_msec=[-667, -333, 333],
        ear_distance_consistency_window_msec=1000,
    )


def make_mice_pair_features_config_v6() -> MicePairFeaturesConfig:
    """
    Action-oriented pair config (lightweight).

    Adds signals for:
      - Mount/intromit: agent nose ↔ target tail_base distance & approach rate; facing angles using nose and tail_base.
      - Follow/approach/avoid/escape: center↔center distance, approach rate, time-to-contact, velocity alignment (0, +333 ms).
      - Chase/attack/defend: pursuit line-of-sight alignment (1000 ms), co-movement stats (500/1000 ms), speed-corr (1000/2000).
      - Sniff variants: keep nose↔nose (lags) + close proportion.

    Heavy ops (rolling min/max) are limited to ONE pair (center↔center) at ONE window (1000 ms).
    """

    # Distances that matter across actions (normalized by agent ||H||)
    pairwise_distance_normalised = [
        ("nose", "nose"),  # sniff/sniffface
        ("nose", "tail_base"),  # genital sniff / mount / intromit
        ("nose", "body_center"),  # sniffbody / approach
        ("tail_base", "tail_base"),  # mounting alignment/back-to-back checks
        ("body_center", "body_center"),  # follow/approach/avoid & social spacing
    ]

    # Approach rates (keep very light: one lag)
    pairwise_approach_rate = [
        ("nose", "nose"),
        ("nose", "tail_base"),
        ("body_center", "body_center"),
    ]
    pairwise_approach_rate_lags_msec = [333]

    # Time-to-contact (in seconds) derived from center distance dynamics
    time_to_contact_pairs = [("body_center", "body_center"), ("nose", "tail_base")]

    # Facing: use nose & tail_base to resolve front/back approaches
    facing_t2a = [
        "nose",
        "tail_base",
    ]  # target→agent (how agent is oriented relative to target points)
    facing_a2t = ["nose", "tail_base"]  # agent→target

    # Keep min/max cost tiny: just center↔center at 1000 ms
    distance_stats_pairs = [("body_center", "body_center")]
    distance_stats_windows_msec = [1000]

    return MicePairFeaturesConfig(
        # --- Distances & approach/closure ---
        pairwise_distance_normalised=pairwise_distance_normalised,
        pairwise_approach_rate=pairwise_approach_rate,
        pairwise_approach_rate_lags_msec=pairwise_approach_rate_lags_msec,
        time_to_contact_sec=time_to_contact_pairs,
        # --- Orientation / facing ---
        facing_angles_target_to_agent=facing_t2a,
        facing_angles_agent_to_target=facing_a2t,
        include_body_axis_alignment=True,
        # --- Simple windowed separation stats (minimal heavy work) ---
        distance_stats_pairs=distance_stats_pairs,
        distance_stats_windows_msec=distance_stats_windows_msec,
        # --- Interaction continuity (variance/mean of center distance^2) ---
        include_interaction_continuity=True,
        interaction_continuity_window_msec=1000,
        # --- Alignment & co-movement ---
        velocity_alignment_offsets_msec=[0, 333],  # follow/chase vs delayed reactions
        velocity_dot_windows_msec=[500, 1000],  # short co-movement stats (cheap)
        pursuit_alignment_windows_msec=[1000],  # pursuit along LOS (cheap via cumsums)
        speed_correlation_windows_msec=[1000, 2000],  # speed coupling for chase/follow
        # --- Chase indicator (approach * target lead) ---
        chase_window_msec=1000,
        # --- Nose–nose dynamics for sniff classes (light) ---
        nose_nose_lags_msec=[333, 1333],
        include_nose_nose_close_proportion=True,
        nose_nose_close_threshold_norm=0.20,  # within 0.2 body-lengths ≈ "very close"
    )


def make_mice_pair_features_config_v7() -> MicePairFeaturesConfig:
    """
    Pair features v7 (lightweight, importance-guided, multi-action ready).

    - Keeps existing rolling stats ONLY:
        * co-movement windows: 500ms, 1000ms  (mean & std)  → pair_co_win{500,1000}ms_{mean,std}
        * speed correlation:   1000ms, 2000ms               → pair_speedcorr_win{1000,2000}ms
        * distance^2 stats for center-center @ 1000ms       → pair_d2_body_center__body_center_win1000ms_{mean,min,max,std}
        * interaction continuity @ 1000ms                    → pair_interaction_continuity
        * pursuit & chase @ 1000ms                           → pair_pursuit_{Alead,Blead}_win1000ms_mean, pair_chase_win1000ms_mean

    - Adds/keeps only lightweight features (no new rolling min/max/std):
        * Key distances: nose↔tail_base, nose↔nose, tail_base↔tail_base, nose↔body_center, center↔center
        * Approach rates at 333ms for: nose↔tail_base, nose↔nose, center↔center
        * Time-to-contact: center↔center, nose↔tail_base
        * Facing angles both directions (sin/cos/rad) for nose and tail_base
        * Velocity alignment with offsets 0ms and 333ms
        * Nose–nose temporal dynamics at 333ms & 1333ms, plus close-proportion
        * Body-axis alignment (nose→tail vs nose→tail)
    """
    # --- Distances that dominate across actions (normalized by ||H_agent||) ---
    pairwise_distance_normalised = [
        ("nose", "tail_base"),  # sniffgenital, mount/intromit
        ("nose", "nose"),  # sniffface, close contact
        ("tail_base", "tail_base"),  # mounting/back alignment
        ("nose", "body_center"),  # sniffbody, approach
        ("body_center", "body_center"),  # spacing, follow/avoid/huddle
    ]

    # --- Approach/closure rates (light; centered finite diff on distance) ---
    pairwise_approach_rate = [
        ("nose", "tail_base"),
        ("nose", "nose"),
        ("body_center", "body_center"),
    ]
    pairwise_approach_rate_lags_msec = [333]

    # --- Time-to-contact (no rolling stats) ---
    time_to_contact_sec = [
        ("body_center", "body_center"),  # general approach/follow/avoid
        ("nose", "tail_base"),  # genital approach / mounting
    ]

    # --- Facing (we emit radians, sin, cos) ---
    facing_angles_target_to_agent = ["nose", "tail_base"]
    facing_angles_agent_to_target = ["nose", "tail_base"]

    # --- Keep ONLY existing rolling stats (no new ones added) ---
    distance_stats_pairs = [("body_center", "body_center")]
    distance_stats_windows_msec = [1000]

    include_interaction_continuity = True
    interaction_continuity_window_msec = 1000

    # Velocity alignment (instantaneous; with signed offsets)
    velocity_alignment_offsets_msec = [0, 333]

    # Co-movement (windowed dot-product stats) → covers pair_co_win500/1000 mean/std
    velocity_dot_windows_msec = [500, 1000]

    # Pursuit/escape along LOS (window mean) + chase score
    pursuit_alignment_windows_msec = [1000]
    chase_window_msec = 1000

    # Speed correlation windows kept as in your stack
    speed_correlation_windows_msec = [1000, 2000]

    # Nose–nose temporal dynamics (lags, changes, close proportion)
    nose_nose_lags_msec = [333, 1333]
    nose_nose_close_threshold_norm = 0.20
    include_nose_nose_close_proportion = True

    # Body-axis alignment (nose→tail vectors cosine)
    include_body_axis_alignment = True

    return MicePairFeaturesConfig(
        pairwise_distance_normalised=pairwise_distance_normalised,
        pairwise_approach_rate=pairwise_approach_rate,
        pairwise_approach_rate_lags_msec=pairwise_approach_rate_lags_msec,
        time_to_contact_sec=time_to_contact_sec,
        facing_angles_target_to_agent=facing_angles_target_to_agent,
        facing_angles_agent_to_target=facing_angles_agent_to_target,
        distance_stats_pairs=distance_stats_pairs,
        distance_stats_windows_msec=distance_stats_windows_msec,
        include_interaction_continuity=include_interaction_continuity,
        interaction_continuity_window_msec=interaction_continuity_window_msec,
        velocity_alignment_offsets_msec=velocity_alignment_offsets_msec,
        velocity_dot_windows_msec=velocity_dot_windows_msec,
        pursuit_alignment_windows_msec=pursuit_alignment_windows_msec,
        chase_window_msec=chase_window_msec,
        speed_correlation_windows_msec=speed_correlation_windows_msec,
        nose_nose_lags_msec=nose_nose_lags_msec,
        nose_nose_close_threshold_norm=nose_nose_close_threshold_norm,
        include_nose_nose_close_proportion=include_nose_nose_close_proportion,
        include_body_axis_alignment=include_body_axis_alignment,
    )


def make_single_mouse_features_config_v8() -> SingleMouseFeaturesConfig:
    """
    Single-mouse v7 (lightweight, importance-guided, multi-action ready).

    Keeps existing rolling families already in your stack:
      - curvature means @ [1000, 2000]
      - turning rate @ [1000]
      - multiscale speeds (mean/std) @ [333, 1333, 5333] + speed ratio (333/5333)
      - center stats @ [167, 500, 1000, 2000]
      - EWM @ [2000, 4000]
      - speed percentiles @ [2000, 4000]
      - lagged displacements (τ ∈ {333, 667, 1333})
      - ear offsets ±{333, 667} and 1s ear-distance consistency

    Adds only lightweight features that help other actions:
      - extra within-mouse distances & relative motions incl. nose↔tail_tip (selfgroom/intromit/mount),
        neck/body chain segments (approach/avoid), ear/lateral/hip symmetry (dominance, rear),
        and a body-axis coupling angle between (tail_base→nose) and (tail_base→tail_tip).
      - lagged displacement of the nose (fast head bobs; sniffbody/selfgroom/run).
    """
    # ---- Within-mouse distances (normalised by ||H||) ----
    # (Union of your high-signal set + a few light additions for other actions)
    pairwise = [
        ("tail_base", "nose"),
        ("tail_base", "neck"),
        ("tail_base", "body_center"),
        ("tail_base", "tail_tip"),
        ("hip_left", "hip_right"),
        ("lateral_left", "lateral_right"),
        ("neck", "nose"),
        ("neck", "hip_left"),
        ("neck", "hip_right"),
        ("nose", "lateral_left"),
        ("nose", "lateral_right"),
        ("body_center", "nose"),
        ("body_center", "neck"),
        ("ear_left", "ear_right"),
        # lightweight additions:
        ("nose", "tail_tip"),  # selfgroom/genital focus, mounting posture
        ("neck", "body_center"),  # approach/avoid chain compactness
    ]

    # ---- Relative motions (unit-velocity dot/cross; lightweight) ----
    relmot = [
        ("nose", "body_center"),
        ("nose", "neck"),
        ("nose", "tail_base"),
        ("neck", "tail_base"),
        ("hip_left", "hip_right"),
        ("lateral_left", "lateral_right"),
        ("tail_tip", "tail_base"),
        ("ear_left", "ear_right"),
        # additions:
        ("nose", "tail_tip"),  # grooming, mounting, tight turns when fleeing
    ]

    # ---- Angles (radians/sin/cos) ----
    angles = [
        (("tail_base", "nose"), "H"),
        (("tail_base", "neck"), "H"),
        (("neck", "nose"), "H"),
        (("tail_base", "tail_tip"), "H"),
        (("hip_left", "hip_right"), "H"),  # pelvis orientation (mount/dominance)
        (("lateral_left", "lateral_right"), "H"),  # lateral splay (avoid/defend/rear)
        (("ear_left", "ear_right"), "H"),  # ear splay/alertness
        (("nose", "ear_left"), ("nose", "ear_right")),  # ear splay from nose
        (("neck", "hip_left"), ("neck", "hip_right")),  # torso symmetry
        (
            ("body_center", "nose"),
            ("body_center", "tail_base"),
        ),  # body axis orientation
        # new lightweight coupling: head vs tail orientation (mount/intromit posture)
        (("tail_base", "nose"), ("tail_base", "tail_tip")),
    ]

    return SingleMouseFeaturesConfig(
        # Scalars/toggles
        include_is_missing_keypoint=True,
        include_norm_in_derivatives=True,
        include_categorical=True,
        lag_windows_ms=[100, 200, 400, 800],
        # Normalised coordinates/derivatives for salient keypoints
        normalised_coordinates=[
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
        ],
        normalised_velocities=[
            "nose",
            "neck",
            "body_center",
            "hip_left",
            "hip_right",
            "lateral_left",
            "lateral_right",
            "ear_left",
            "ear_right",
            "tail_tip",
        ],
        normalised_accelerations=[
            "nose",
            "neck",
            "body_center",
            "hip_left",
            "hip_right",
            "tail_tip",
            "ear_left",
            "ear_right",
        ],
        # Arena-scale (but length-normalised) kinematics
        scaled_velocities=["body_center", "nose"],
        scaled_accelerations=["body_center"],
        # Curvatures
        curvatures=["nose", "body_center"],
        # Within-mouse distances + body length
        pairwise_distances=pairwise,
        include_normalised_body_length=True,
        # Relative motions (lightweight)
        relative_motions=relmot,
        # Heading & angles (+ velocities/accelerations)
        include_heading_angle=True,
        include_heading_angle_velocity=True,
        include_heading_angle_acceleration=True,
        angles=angles,
        include_angles_velocities=True,
        # (Keep existing rolling families as-is; no new heavy stats added)
        curvature_keypoints=["body_center"],
        curvature_mean_windows_msec=[1000, 2000],
        turning_rate_windows_msec=[1000],
        speed_keypoints=["body_center"],
        speed_windows_msec=[333, 1333, 5333],
        speed_ratio_pairs_msec=[(333, 5333)],
        center_stats_windows_msec=[167, 500, 1000, 2000],
        ewm_spans_msec=[2000, 4000],
        speed_percentile_windows_msec=[2000, 4000],
        # Lagged displacements (duration-aware): keep ears; add nose (light, useful)
        lagged_displacement_self_keypoints=["ear_left", "ear_right", "nose"],
        lagged_displacement_cross_pairs=[
            ("ear_left", "tail_base"),
            ("ear_right", "tail_base"),
            ("nose", "tail_base"),
            ("nose", "neck"),  # quick head bob vs neck (sniffbody/selfgroom)
        ],
        lagged_displacement_lags_msec=[333, 667, 1333],
        # Ear separation temporal offsets & consistency (existing)
        ear_distance_signed_offsets_msec=[-667, -333, 333, 667],
        ear_distance_consistency_window_msec=1000,
    )
