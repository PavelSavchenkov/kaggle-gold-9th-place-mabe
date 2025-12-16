from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.feats_common import NamedCoords, NamedFeatures, interpolate_nans_per_column
from gbdt.configs import (
    FeaturesConfig,
    MicePairFeaturesConfig,
    SingleMouseFeaturesConfig,
)


class InterpCache:
    def __init__(self):
        self.cache = {}

    def get(self, n, dt, lag):
        key = (n, dt, float(lag))
        if key in self.cache:
            return self.cache[key]
        k = lag / dt
        t = np.arange(n, dtype=np.float32)
        pos = t - k
        valid = (pos >= 0.0) & (pos <= n - 1)
        pv = pos[valid].astype(np.int64, copy=False)  # keep float for 'a' below
        i0 = np.floor(pos[valid]).astype(np.int64)
        i1 = np.minimum(i0 + 1, n - 1)
        a = (pos[valid] - i0).astype(np.float32)[:, None]
        self.cache[key] = (valid, i0, i1, a)
        return self.cache[key]


interp_cache = InterpCache()


# ---- Numeric helpers (module-level, shared) ----
QUANTIZE_LAGS = True


def frames_and_dt(lag_ms: float, dt_ms: float):
    """Quantize lag in ms to frames and effective ms when enabled.

    Returns (k_frames, eff_ms) when QUANTIZE_LAGS is True, else (None, lag_ms).
    """
    if QUANTIZE_LAGS:
        k = max(1, int(round(float(lag_ms) / float(dt_ms))))
        eff = float(k) * float(dt_ms)
        return k, eff
    return None, float(lag_ms)


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise division with NaN where denom is invalid.

    Uses a single output allocation and `where=` to avoid unnecessary
    temporaries. Keeps dtype of inputs (typically float32).
    """
    out = np.empty_like(a)
    out.fill(np.nan)
    denom_ok = np.isfinite(b) & (b != 0)
    # ignore warnings; invalid positions remain NaN in `out`
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(a, b, out=out, where=denom_ok)
    return out


def cumsums_1d(x1: np.ndarray):
    """Prefix sums for 1D float array and its squares.

    Computes cumulative sums in float64 to prevent overflow on long
    sequences, then returns the prefix sums with a leading zero so window
    sums are csum[r] - csum[l].
    """
    assert x1.ndim == 1
    x1c = np.ascontiguousarray(x1, dtype=np.float64)
    zero = 0.0
    csum = np.cumsum(np.r_[zero, x1c], dtype=np.float64)
    csum2 = np.cumsum(np.r_[zero, x1c * x1c], dtype=np.float64)
    return csum, csum2


def rolling_mean_std_from_cumsums(csum: np.ndarray, csum2: np.ndarray, w: int):
    """Trailing rolling mean/std/var using precomputed cumsums for (N,) series.

    For the first w-1 rows, window grows from 1..w. Uses float64 math for
    stability, returns float32 outputs shaped (N,1).
    """
    n = csum.size - 1
    idx = np.arange(n, dtype=np.int64)
    l = np.maximum(0, idx - w + 1)
    r = idx + 1
    win = (r - l).astype(np.float64)
    s = csum[r] - csum[l]
    s2 = csum2[r] - csum2[l]
    mean64 = (s / win)[:, None]
    var64 = np.maximum(0.0, s2 / win - (s / win) ** 2)[:, None]
    std64 = np.sqrt(var64)
    # Cast back to float32 to keep overall feature dtype/memory consistent
    mean = mean64.astype(np.float32, copy=False)
    std = std64.astype(np.float32, copy=False)
    var = var64.astype(np.float32, copy=False)
    return mean, std, var


def rolling_min_max(x: np.ndarray, w: int):
    """Vectorized trailing min/max over window w for (N,1) input.

    For i < w-1, the effective window grows as [0..i]. Returns (N,1) arrays.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    a = x[:, 0]
    n = a.size
    mins = np.empty(n, dtype=a.dtype)
    maxs = np.empty(n, dtype=a.dtype)
    if w <= 1:
        mins[:] = a
        maxs[:] = a
        return mins[:, None], maxs[:, None]
    mins[: w - 1] = np.minimum.accumulate(a[: w - 1])
    maxs[: w - 1] = np.maximum.accumulate(a[: w - 1])
    win = sliding_window_view(a, w)  # (n-w+1, w)
    mins[w - 1 :] = win.min(axis=1)
    maxs[w - 1 :] = win.max(axis=1)
    return mins[:, None], maxs[:, None]


def interpolate_with_lag(X, lag, dt):
    X = np.asarray(X)
    n, m = X.shape
    Y = np.empty_like(X)
    Y[...] = 0
    valid, i0, i1, a = interp_cache.get(n, dt, lag)
    if valid.any():
        X0 = X[i0, :]
        X1 = X[i1, :]
        one = a.dtype.type(1.0)
        Y[valid, :] = (one - a) * X0 + a * X1
    return Y


def calc_O_H(p: NamedCoords, config: FeaturesConfig) -> tuple[NamedCoords, NamedCoords]:
    """
    Returns:
        O - origin (tail_base)
        H - heading vector = (ear_avg - origin), fallback to neck if ear avg is NaN
    """
    col = {name: i for i, name in enumerate(p.columns)}

    def x_idx(i):
        # use slice view to avoid copies
        return p.x[:, i : i + 1]

    def y_idx(i):
        # use slice view to avoid copies
        return p.y[:, i : i + 1]

    i_tail = col["tail_base"]
    i_el, i_er = col["ear_left"], col["ear_right"]
    i_neck = col["neck"]

    O = NamedCoords(columns=["O"], x=x_idx(i_tail), y=y_idx(i_tail))

    half = p.x.dtype.type(0.5)
    Hx = (x_idx(i_el) + x_idx(i_er)) * half
    Hy = (y_idx(i_el) + y_idx(i_er)) * half

    # fallback to neck where ear avg is NaN
    mask_nan = np.isnan(Hx) | np.isnan(Hy)
    if np.any(mask_nan):
        nx, ny = x_idx(i_neck), y_idx(i_neck)
        Hx = np.where(mask_nan, nx, Hx)
        Hy = np.where(mask_nan, ny, Hy)

    Hx = Hx - O.x
    Hy = Hy - O.y
    H = NamedCoords(columns=["H"], x=Hx, y=Hy)
    return (O, H)


def precompute_derivatives(
    p_x: np.ndarray, p_y: np.ndarray, lags_ms: list[float], dt: float
):
    """
    Vectorized derivatives for all columns and lags:
      D1* : central difference    (N,M)
      D2* : second difference     (N,M)
    """
    D1x, D1y, D2x, D2y = {}, {}, {}, {}
    dtype = p_x.dtype
    n = p_x.shape[0]
    for lag in lags_ms:
        k, eff_ms = frames_and_dt(lag, dt)
        if k is not None:
            # integer frame shifts with edge replication
            fwd_x = np.empty_like(p_x)
            fwd_x[:-k, :] = p_x[k:, :]
            fwd_x[-k:, :] = p_x[n - 1 : n, :]
            bwd_x = np.empty_like(p_x)
            bwd_x[k:, :] = p_x[:-k, :]
            bwd_x[:k, :] = p_x[0:1, :]

            fwd_y = np.empty_like(p_y)
            fwd_y[:-k, :] = p_y[k:, :]
            fwd_y[-k:, :] = p_y[n - 1 : n, :]
            bwd_y = np.empty_like(p_y)
            bwd_y[k:, :] = p_y[:-k, :]
            bwd_y[:k, :] = p_y[0:1, :]
        else:
            fwd_x = interpolate_with_lag(p_x, -lag, dt)
            bwd_x = interpolate_with_lag(p_x, +lag, dt)
            fwd_y = interpolate_with_lag(p_y, -lag, dt)
            bwd_y = interpolate_with_lag(p_y, +lag, dt)

        inv_2lag = dtype.type(1.0) / dtype.type(2.0 * eff_ms)
        inv_lag2 = dtype.type(1.0) / dtype.type(eff_ms * eff_ms)

        D1x[lag] = (fwd_x - bwd_x) * inv_2lag
        D1y[lag] = (fwd_y - bwd_y) * inv_2lag
        D2x[lag] = (fwd_x - dtype.type(2.0) * p_x + bwd_x) * inv_lag2
        D2y[lag] = (fwd_y - dtype.type(2.0) * p_y + bwd_y) * inv_lag2
    return D1x, D1y, D2x, D2y


def precompute_derivatives_single(A: np.ndarray, lags_ms: list[float], dt: float):
    """
    Same as above but for a single (N,1) column matrix.
    Returns dicts of (N,1) arrays.
    """
    D1, D2 = {}, {}
    dtype = A.dtype
    n = A.shape[0]
    for lag in lags_ms:
        k, eff_ms = frames_and_dt(lag, dt)
        if k is not None:
            fwd = np.empty_like(A)
            fwd[:-k, :] = A[k:, :]
            fwd[-k:, :] = A[n - 1 : n, :]
            bwd = np.empty_like(A)
            bwd[k:, :] = A[:-k, :]
            bwd[:k, :] = A[0:1, :]
        else:
            fwd = interpolate_with_lag(A, -lag, dt)
            bwd = interpolate_with_lag(A, +lag, dt)
        inv_2lag = dtype.type(1.0) / dtype.type(2.0 * eff_ms)
        inv_lag2 = dtype.type(1.0) / dtype.type(eff_ms * eff_ms)
        D1[lag] = (fwd - bwd) * inv_2lag
        D2[lag] = (fwd - dtype.type(2.0) * A + bwd) * inv_lag2
    return D1, D2


def calc_numeric_single_mouse(
    p: NamedCoords,
    O: NamedCoords,
    H: NamedCoords,
    config: SingleMouseFeaturesConfig,
    fps: float,
) -> NamedFeatures:
    # work on a copy and move to O; keep float32, contiguous
    px = np.asarray(p.x, dtype=np.float32, order="C").copy()
    py = np.asarray(p.y, dtype=np.float32, order="C").copy()
    px -= O.x
    py -= O.y
    p = NamedCoords(columns=p.columns, x=px, y=py)

    n, _ = p.x.shape
    cols: list[str] = []
    feats: list[np.ndarray] = []

    # --- Time params ---
    dt = 1000.0 / fps  # ms
    lag_ms_list = list(config.lag_windows_ms)
    cfg = config

    # --- Fast column index map ---
    col = {name: j for j, name in enumerate(p.columns)}

    # --- Heading / scale ---
    Hx, Hy = H.x, H.y
    hnorm = np.sqrt(Hx * Hx + Hy * Hy)  # true body length
    eps = 1e-6
    hscale = hnorm + eps
    inv_hscale = 1.0 / hscale
    inv_hscale2 = inv_hscale * inv_hscale  # 1/(||H||^2)

    # --- Helpers to collect outputs ---
    def lag_suffix(ms: float) -> str:
        return f"_lag{int(round(ms))}ms"

    def add(name: str, arr: np.ndarray):
        cols.append(name)
        feats.append(arr)

    def add_vec(prefix: str, x: np.ndarray, y: np.ndarray, suffix: str = ""):
        add(f"{prefix}{suffix}_x", x)
        add(f"{prefix}{suffix}_y", y)

    def to_heading_scaled(dx: np.ndarray, dy: np.ndarray):
        # rotate to heading and divide by ||H||: use H components directly
        # x' = (dx*Hx + dy*Hy) / ||H||^2 ; y' = (dy*Hx - dx*Hy) / ||H||^2
        xh = (dx * Hx + dy * Hy) * inv_hscale2
        yh = (dy * Hx - dx * Hy) * inv_hscale2
        return xh, yh

    def get_xy_idx(j: int):
        # slice views avoid extra column copies
        return p.x[:, j : j + 1], p.y[:, j : j + 1]

    # --- Cached ms->frames ---
    _ms_to_frames_cache: dict[float, int] = {}

    def ms_to_frames(ms: int | float) -> int:
        key = float(ms)
        k = _ms_to_frames_cache.get(key)
        if k is None:
            k = int(round(key / dt))
            k = max(1, k)
            _ms_to_frames_cache[key] = k
        return k

    def shift_backfill(x: np.ndarray, k: int) -> np.ndarray:
        """Shift by k frames to the past with edge backfill (x[0]).
        For k < 0 (future), backfill with last value x[-1].
        """
        if k == 0:
            return x
        n = x.shape[0]
        y = np.empty_like(x)
        if k > 0:
            y[:k] = x[0]
            y[k:] = x[:-k]
        else:
            kk = -k
            y[:-kk] = x[kk:]
            y[-kk:] = x[-1]
        return y

    # Precompute heading-frame transforms for base coords and derivatives
    PX_h, PY_h = to_heading_scaled(p.x, p.y)

    # Precompute derivative heading-frame projections per lag
    D1x, D1y, D2x, D2y = precompute_derivatives(p.x, p.y, lag_ms_list, dt)
    D1x_h, D1y_h, D2x_h, D2y_h = {}, {}, {}, {}
    for lag_ms in lag_ms_list:
        hx, hy = to_heading_scaled(D1x[lag_ms], D1y[lag_ms])
        D1x_h[lag_ms], D1y_h[lag_ms] = hx, hy
        axh, ayh = to_heading_scaled(D2x[lag_ms], D2y[lag_ms])
        D2x_h[lag_ms], D2y_h[lag_ms] = axh, ayh

    # --- Normalised coordinates (body-frame, rotated & scaled) ---
    for kp in cfg.normalised_coordinates:
        j = col[kp]
        add_vec(f"normcoord_{kp}", PX_h[:, j : j + 1], PY_h[:, j : j + 1])

    # --- Derivatives of arena origin (O) once per lag ---
    Ox, Oy = O.x, O.y
    dOx, ddOx = precompute_derivatives_single(Ox, lag_ms_list, dt)
    dOy, ddOy = precompute_derivatives_single(Oy, lag_ms_list, dt)

    # --- Derivative-based features per lag ---
    for lag_ms in lag_ms_list:
        suffix = lag_suffix(lag_ms)

        # Normalised velocities (body-frame)
        if cfg.normalised_velocities:
            for kp in cfg.normalised_velocities:
                j = col[kp]
                vx_h = D1x_h[lag_ms][:, j : j + 1]
                vy_h = D1y_h[lag_ms][:, j : j + 1]
                add_vec(f"normvel_{kp}", vx_h, vy_h, suffix)
                if cfg.include_norm_in_derivatives:
                    add(
                        f"normvel_{kp}{suffix}_norm", np.sqrt(vx_h * vx_h + vy_h * vy_h)
                    )

        # Normalised accelerations (body-frame)
        if cfg.normalised_accelerations:
            for kp in cfg.normalised_accelerations:
                j = col[kp]
                ax_h = D2x_h[lag_ms][:, j : j + 1]
                ay_h = D2y_h[lag_ms][:, j : j + 1]
                add_vec(f"normaccel_{kp}", ax_h, ay_h, suffix)
                if cfg.include_norm_in_derivatives:
                    add(
                        f"normaccel_{kp}{suffix}_norm",
                        np.sqrt(ax_h * ax_h + ay_h * ay_h),
                    )

        # Scaled velocities (arena-frame / ||H||)
        if cfg.scaled_velocities:
            for kp in cfg.scaled_velocities:
                j = col[kp]
                vx_ar = D1x[lag_ms][:, j : j + 1] + dOx[lag_ms]
                vy_ar = D1y[lag_ms][:, j : j + 1] + dOy[lag_ms]
                add_vec(
                    f"scaledvel_{kp}",
                    vx_ar * inv_hscale,
                    vy_ar * inv_hscale,
                    suffix,
                )
                if cfg.include_norm_in_derivatives:
                    vnorm = np.sqrt(vx_ar * vx_ar + vy_ar * vy_ar)
                    add(f"scaledvel_{kp}{suffix}_norm", vnorm * inv_hscale)

        # Scaled accelerations (arena-frame / ||H||)
        if cfg.scaled_accelerations:
            for kp in cfg.scaled_accelerations:
                j = col[kp]
                ax_ar = D2x[lag_ms][:, j : j + 1] + ddOx[lag_ms]
                ay_ar = D2y[lag_ms][:, j : j + 1] + ddOy[lag_ms]
                add_vec(
                    f"scaledaccel_{kp}",
                    ax_ar * inv_hscale,
                    ay_ar * inv_hscale,
                    suffix,
                )
                if cfg.include_norm_in_derivatives:
                    anorm = np.sqrt(ax_ar * ax_ar + ay_ar * ay_ar)
                    add(f"scaledaccel_{kp}{suffix}_norm", anorm * inv_hscale)

        # Curvature (dimensionless)
        if cfg.curvatures:
            for kp in cfg.curvatures:
                j = col[kp]
                vx = D1x[lag_ms][:, j : j + 1]
                vy = D1y[lag_ms][:, j : j + 1]
                ax = D2x[lag_ms][:, j : j + 1]
                ay = D2y[lag_ms][:, j : j + 1]
                cross = vx * ay - vy * ax
                vnorm3 = (vx * vx + vy * vy) ** 1.5
                kappa = safe_div(cross, vnorm3)  # 1/length
                kappa_dimless = kappa * hscale  # dimensionless
                add(f"curvature_{kp}{suffix}", kappa_dimless)

        # Relative motions
        if cfg.relative_motions:
            for a, b in cfg.relative_motions:
                ja, jb = col[a], col[b]
                avx = D1x[lag_ms][:, ja : ja + 1] + dOx[lag_ms]
                avy = D1y[lag_ms][:, ja : ja + 1] + dOy[lag_ms]
                bvx = D1x[lag_ms][:, jb : jb + 1] + dOx[lag_ms]
                bvy = D1y[lag_ms][:, jb : jb + 1] + dOy[lag_ms]
                adotb = avx * bvx + avy * bvy
                acrossb = avx * bvy - avy * bvx
                an = np.sqrt(avx * avx + avy * avy)
                bn = np.sqrt(bvx * bvx + bvy * bvy)
                denom = an * bn
                add(f"relmot_{a}__{b}{suffix}_dot", safe_div(adotb, denom))
                add(f"relmot_{a}__{b}{suffix}_cross", safe_div(acrossb, denom))

    # Pairwise distances (scaled by ||H||)
    if cfg.pairwise_distances:
        for a, b in cfg.pairwise_distances:
            ja, jb = col[a], col[b]
            axp, ayp = p.x[:, ja : ja + 1], p.y[:, ja : ja + 1]
            bxp, byp = p.x[:, jb : jb + 1], p.y[:, jb : jb + 1]
            dx = axp - bxp
            dy = ayp - byp
            dist = np.sqrt(dx * dx + dy * dy)
            add(f"dist_{a}__{b}", dist * inv_hscale)

    # Normalised body length: ||H|| / median(||H||) (use true hnorm, no ε)
    if cfg.include_normalised_body_length:
        med = np.nanmedian(hnorm) if np.any(~np.isnan(hnorm)) else np.nan
        add(
            "body_length_norm",
            (
                (hnorm / med)
                if (med is not None and med > 0)
                else np.full_like(hnorm, np.nan)
            ),
        )

    # Heading angle & derivatives
    if (
        cfg.include_heading_angle
        or cfg.include_heading_angle_velocity
        or cfg.include_heading_angle_acceleration
    ):
        theta = np.arctan2(Hy, Hx)
        theta_unwrapped = np.unwrap(theta[:, 0]).astype(theta.dtype)[:, None]
        if cfg.include_heading_angle:
            add("heading_angle", theta)
            add("sin_heading_angle", np.sin(theta))
            add("cos_heading_angle", np.cos(theta))

        # precompute theta derivs if needed
        if cfg.include_heading_angle_velocity or cfg.include_heading_angle_acceleration:
            dTh, ddTh = precompute_derivatives_single(theta_unwrapped, lag_ms_list, dt)
            for lag_ms in lag_ms_list:
                suffix = lag_suffix(lag_ms)
                if cfg.include_heading_angle_velocity:
                    add(f"heading_angle_velocity{suffix}", dTh[lag_ms])
                if cfg.include_heading_angle_acceleration:
                    add(f"heading_angle_acceleration{suffix}", ddTh[lag_ms])

    # --- Segment angles (+ optional angular velocities using first lag) ---
    if cfg.angles:
        for seg1, seg2 in cfg.angles:
            p0, p1 = seg1
            j0, j1 = col[p0], col[p1]
            p0x, p0y = p.x[:, j0 : j0 + 1], p.y[:, j0 : j0 + 1]
            p1x, p1y = p.x[:, j1 : j1 + 1], p.y[:, j1 : j1 + 1]

            v1x = p1x - p0x
            v1y = p1y - p0y

            if isinstance(seg2, str) and seg2 == "H":
                v2x, v2y = Hx, Hy
                seg2_name = "H"
            else:
                q0, q1 = seg2  # type: ignore
                jq0, jq1 = col[q0], col[q1]
                q0x, q0y = p.x[:, jq0 : jq0 + 1], p.y[:, jq0 : jq0 + 1]
                q1x, q1y = p.x[:, jq1 : jq1 + 1], p.y[:, jq1 : jq1 + 1]
                v2x = q1x - q0x
                v2y = q1y - q0y
                seg2_name = f"{q0}__{q1}"

            dot = v1x * v2x + v1y * v2y
            cross = v1x * v2y - v1y * v2x
            ang = np.arctan2(cross, dot)  # (n,1)

            base = f"angle_{p0}__{p1}__{seg2_name}"
            add(base, ang)
            add(f"{base}_sin", np.sin(ang))
            add(f"{base}_cos", np.cos(ang))

            if cfg.include_angles_velocities and lag_ms_list:
                # velocity at the first lag using wrapped finite differences
                lag0 = lag_ms_list[0]
                sfx0 = f"_lag{int(round(lag0))}ms"
                k, eff_ms = frames_and_dt(lag0, dt)
                if k is not None:
                    th = ang[:, 0]
                    prev = np.empty_like(th)
                    prev[:k] = th[0]
                    prev[k:] = th[:-k]
                    pi = th.dtype.type(np.pi)
                    two_pi = th.dtype.type(2.0 * np.pi)
                    d = ((th - prev) + pi) % two_pi - pi
                    vel = (d / th.dtype.type(eff_ms))[:, None]
                    add(f"{base}_velocity{sfx0}", vel)
                else:
                    ang_unwrapped = np.unwrap(ang[:, 0]).astype(ang.dtype)[:, None]
                    dTh0, _ = precompute_derivatives_single(ang_unwrapped, [lag0], dt)
                    add(f"{base}_velocity{sfx0}", dTh0[lag0])

    # --- Curvature mean over windows (use first derivative lag) ---
    if cfg.curvature_keypoints and cfg.curvature_mean_windows_msec:
        lag0 = lag_ms_list[0] if lag_ms_list else 100.0
        for kp in cfg.curvature_keypoints:
            j = col[kp]
            vx = D1x[lag0][:, j : j + 1]
            vy = D1y[lag0][:, j : j + 1]
            ax = D2x[lag0][:, j : j + 1]
            ay = D2y[lag0][:, j : j + 1]
            cross = vx * ay - vy * ax
            vnorm3 = (vx * vx + vy * vy) ** 1.5
            kappa = safe_div(cross, vnorm3) * hscale
            kappa_abs = np.abs(kappa)
            x1 = kappa_abs[:, 0]
            csum, csum2 = cumsums_1d(x1)
            for ms in cfg.curvature_mean_windows_msec:
                w = ms_to_frames(ms)
                m, _, _ = rolling_mean_std_from_cumsums(csum, csum2, w)
                add(f"curv_mean_{kp}_win{int(ms)}ms", m)

    # --- Turning rate over windows (use velocity angle of reference keypoints) ---
    if cfg.turning_rate_windows_msec and cfg.curvature_keypoints:
        lag0 = lag_ms_list[0] if lag_ms_list else 100.0
        for kp in cfg.curvature_keypoints:
            j = col[kp]
            vx = D1x[lag0][:, j : j + 1]
            vy = D1y[lag0][:, j : j + 1]
            theta = np.arctan2(vy, vx)[:, 0]
            dth = theta - np.r_[theta[0], theta[:-1]]
            # wrap to [-pi, pi]
            pi = theta.dtype.type(np.pi)
            two_pi = theta.dtype.type(2.0 * np.pi)
            dth = ((dth + pi) % two_pi) - pi
            step = np.abs(dth).astype(theta.dtype)[:, None]
            zero = step.dtype.type(0.0)
            csum = np.cumsum(np.r_[zero, step[:, 0]], dtype=step.dtype)
            for ms in cfg.turning_rate_windows_msec:
                w = ms_to_frames(ms)
                idx = np.arange(n)
                l = np.maximum(0, idx - w + 1)
                r = idx + 1
                val = (csum[r] - csum[l])[:, None]
                add(f"turn_rate_{kp}_win{int(ms)}ms", val)

    # --- Multiscale speed statistics (normalised, per second) ---
    speed_means: dict[tuple[str, int], np.ndarray] = {}
    if cfg.speed_keypoints and cfg.speed_windows_msec:
        # finite-difference speed per second
        dt_sec = dt / 1000.0
        for kp in cfg.speed_keypoints:
            j = col[kp]
            px, py = get_xy_idx(j)
            dx = np.empty_like(px)
            dy = np.empty_like(py)
            dx[0] = 0
            dy[0] = 0
            dx[1:] = px[1:] - px[:-1]
            dy[1:] = py[1:] - py[:-1]
            sp = np.sqrt(dx * dx + dy * dy) / (dt_sec + 1e-12)
            sp_norm = sp * inv_hscale
            x1 = sp_norm[:, 0]
            csum, csum2 = cumsums_1d(x1)
            for ms in cfg.speed_windows_msec:
                w = ms_to_frames(ms)
                m, s, _ = rolling_mean_std_from_cumsums(csum, csum2, w)
                add(f"speed_{kp}_win{int(ms)}ms_mean", m)
                add(f"speed_{kp}_win{int(ms)}ms_std", s)
                speed_means[(kp, int(ms))] = m

    # --- Speed ratios between two scales ---
    if cfg.speed_ratio_pairs_msec and speed_means:
        for kp in cfg.speed_keypoints:
            for wf, ws in cfg.speed_ratio_pairs_msec:
                keyf = (kp, int(wf))
                keys = (kp, int(ws))
                if keyf in speed_means and keys in speed_means:
                    num = speed_means[keyf]
                    den = speed_means[keys]
                    add(
                        f"speed_ratio_{kp}_win{int(wf)}ms_over_win{int(ws)}ms",
                        safe_div(num, den + 1e-12),
                    )

    # --- Center-based long-range stats and EWM ---
    # Center fallback: body_center or O + 0.5*H
    if "body_center" in col:
        Cx, Cy = get_xy_idx(col["body_center"])
    else:
        half = Hx.dtype.type(0.5)
        Cx, Cy = O.x + half * Hx, O.y + half * Hy

    if cfg.center_stats_windows_msec:
        # precompute increments of center
        dCx = np.empty_like(Cx)
        dCy = np.empty_like(Cy)
        dCx[0] = 0
        dCy[0] = 0
        dCx[1:] = Cx[1:] - Cx[:-1]
        dCy[1:] = Cy[1:] - Cy[:-1]
        cx1 = Cx[:, 0]
        cy1 = Cy[:, 0]
        dcx1 = dCx[:, 0]
        dcy1 = dCy[:, 0]
        csum_x, csum2_x = cumsums_1d(cx1)
        csum_y, csum2_y = cumsums_1d(cy1)
        csum_dcx, csum2_dcx = cumsums_1d(dcx1)
        csum_dcy, csum2_dcy = cumsums_1d(dcy1)
        for ms in cfg.center_stats_windows_msec:
            w = ms_to_frames(ms)
            mx, sx, _ = rolling_mean_std_from_cumsums(csum_x, csum2_x, w)
            my, sy, _ = rolling_mean_std_from_cumsums(csum_y, csum2_y, w)
            add(f"center_mean_x_win{int(ms)}ms", mx)
            add(f"center_mean_y_win{int(ms)}ms", my)
            add(f"center_std_x_win{int(ms)}ms", sx * inv_hscale)
            add(f"center_std_y_win{int(ms)}ms", sy * inv_hscale)
            mnx, mxx = rolling_min_max(Cx, w)
            mny, mxy = rolling_min_max(Cy, w)
            add(f"center_range_x_win{int(ms)}ms", (mxx - mnx) * inv_hscale)
            add(f"center_range_y_win{int(ms)}ms", (mxy - mny) * inv_hscale)
            # net displacement over window (vector sum of increments)
            net_dx = Cx - shift_backfill(Cx, w - 1)
            net_dy = Cy - shift_backfill(Cy, w - 1)
            disp = np.sqrt(net_dx * net_dx + net_dy * net_dy)
            add(f"center_disp_win{int(ms)}ms", disp * inv_hscale)
            # activity = sqrt(var(dCx) + var(dCy)) over window
            _, _, varx = rolling_mean_std_from_cumsums(csum_dcx, csum2_dcx, w)
            _, _, vary = rolling_mean_std_from_cumsums(csum_dcy, csum2_dcy, w)
            activity = np.sqrt(varx + vary)
            add(f"center_activity_win{int(ms)}ms", activity * inv_hscale)

    if cfg.ewm_spans_msec:
        for ms in cfg.ewm_spans_msec:
            span_frames = ms_to_frames(ms)
            alpha = Cx.dtype.type(2.0) / Cx.dtype.type(span_frames + 1.0)
            one = Cx.dtype.type(1.0)
            ex = np.empty_like(Cx)
            ey = np.empty_like(Cy)
            ex[0] = Cx[0]
            ey[0] = Cy[0]
            for i in range(1, n):
                ex[i] = alpha * Cx[i] + (one - alpha) * ex[i - 1]
                ey[i] = alpha * Cy[i] + (one - alpha) * ey[i - 1]
            add(f"center_ewm_x_span{int(ms)}ms", ex)
            add(f"center_ewm_y_span{int(ms)}ms", ey)

    # --- Speed percentile ranks within windows (per selected speed_keypoints) ---
    if cfg.speed_percentile_windows_msec and cfg.speed_keypoints:
        dt_sec = dt / 1000.0
        from numpy.lib.stride_tricks import sliding_window_view

        for kp in cfg.speed_keypoints:
            j = col[kp]
            px, py = get_xy_idx(j)
            dx = np.empty_like(px)
            dy = np.empty_like(py)
            dx[0] = 0
            dy[0] = 0
            dx[1:] = px[1:] - px[:-1]
            dy[1:] = py[1:] - py[:-1]
            sp = np.sqrt(dx * dx + dy * dy) / (dt_sec + 1e-12)
            sp_norm = (sp * inv_hscale)[:, 0]
            for ms in cfg.speed_percentile_windows_msec:
                w = ms_to_frames(ms)
                if w <= 1:
                    pct = np.zeros(n, dtype=sp_norm.dtype)
                else:
                    pct = np.zeros(n, dtype=sp_norm.dtype)
                    if n >= w:
                        wins = sliding_window_view(sp_norm, w)
                        last = wins[:, -1][:, None]
                        ranks = (wins <= last).sum(axis=1) - 1
                        pct[w - 1 :] = ranks / (w - 1)
                    up_to = min(w - 1, n)
                    for t in range(up_to):
                        window = sp_norm[: t + 1]
                        if window.size <= 1:
                            pct[t] = 0.0
                        else:
                            pct[t] = (np.sum(window <= sp_norm[t]) - 1) / (
                                window.size - 1
                            )
                add(f"speed_pct_{kp}_win{int(ms)}ms", pct[:, None])

    # --- Lagged displacements (self and cross) via interpolation at fractional lags ---
    if cfg.lagged_displacement_lags_msec and (
        cfg.lagged_displacement_self_keypoints or cfg.lagged_displacement_cross_pairs
    ):
        # Precompute lagged positions for all needed keypoints and lags once
        needed_kps = set(cfg.lagged_displacement_self_keypoints)
        needed_kps.update(k2 for _, k2 in cfg.lagged_displacement_cross_pairs)
        lagged_xy: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for ms in cfg.lagged_displacement_lags_msec:
            lagged_xy[ms] = {}
            for kp in needed_kps:
                j = col[kp]
                px_k, py_k = get_xy_idx(j)
                lagged_xy[ms][kp] = (
                    interpolate_with_lag(px_k, ms, dt),
                    interpolate_with_lag(py_k, ms, dt),
                )

        for ms in cfg.lagged_displacement_lags_msec:
            sfx = f"lag{int(ms)}ms"
            # self: ||P_k(t) - P_k(t-τ)|| / ||H||, squared
            for kp in cfg.lagged_displacement_self_keypoints:
                j = col[kp]
                px_k, py_k = get_xy_idx(j)
                px_l, py_l = lagged_xy[ms][kp]
                dx = px_k - px_l
                dy = py_k - py_l
                d2 = (dx * dx + dy * dy) * inv_hscale2
                add(f"disp2_self_{kp}_{sfx}", d2)
            # cross: ||P_k1(t) - P_k2(t-τ)|| / ||H||, squared
            for k1, k2 in cfg.lagged_displacement_cross_pairs:
                j1, j2 = col[k1], col[k2]
                p1x, p1y = get_xy_idx(j1)
                p2x_l, p2y_l = lagged_xy[ms][k2]
                dx = p1x - p2x_l
                dy = p1y - p2y_l
                d2 = (dx * dx + dy * dy) * inv_hscale2
                add(f"disp2_cross_{k1}__{k2}_{sfx}", d2)

    # --- Ear distance signed offsets and consistency ---
    if cfg.ear_distance_signed_offsets_msec or cfg.ear_distance_consistency_window_msec:
        jl, jr = col["ear_left"], col["ear_right"]
        ex, ey = get_xy_idx(jl)
        rx, ry = get_xy_idx(jr)
        ear_d = np.sqrt((ex - rx) ** 2 + (ey - ry) ** 2) * inv_hscale
        if cfg.ear_distance_signed_offsets_msec:
            # precompute shifted ear distances for configured offsets
            ear_shifted = {
                float(off): interpolate_with_lag(ear_d, float(off), dt)
                for off in cfg.ear_distance_signed_offsets_msec
            }
            for off, shifted in ear_shifted.items():
                add(f"ear_distance_offset_{int(off)}ms", shifted)
        if cfg.ear_distance_consistency_window_msec is not None:
            w = ms_to_frames(cfg.ear_distance_consistency_window_msec)
            csum, csum2 = cumsums_1d(ear_d[:, 0])
            m, s, _ = rolling_mean_std_from_cumsums(csum, csum2, w)
            add(
                f"ear_distance_consistency_win{int(cfg.ear_distance_consistency_window_msec)}ms",
                safe_div(s, m + eps),
            )

    data = np.concatenate(feats, axis=1)
    return NamedFeatures(columns=cols, data=data)


def calc_numeric_mice_pair(
    agent: NamedCoords,
    target: NamedCoords,
    config: MicePairFeaturesConfig,
    fps: float,
) -> NamedFeatures:
    """Compute pairwise numeric features for a mice pair.

    All distances are normalised by the agent's heading length ||H_agent||.
    Centers use body_center when available, with fallback to O + 0.5·H.
    Velocities for alignment/co-movement are computed from centers.
    Rolling statistics use trailing windows of the requested duration.
    """
    assert agent.is_aligned(target)

    n, _ = agent.x.shape
    cols: list[str] = []
    feats: list[np.ndarray] = []

    # Time units
    dt_ms = 1000.0 / float(fps)

    # Index maps
    col_a = {name: j for j, name in enumerate(agent.columns)}
    col_b = {name: j for j, name in enumerate(target.columns)}

    # Origins and headings
    Oa, Ha = calc_O_H(agent, FeaturesConfig())  # using default; only names matter
    Ob, Hb = calc_O_H(target, FeaturesConfig())

    Hax, Hay = Ha.x, Ha.y
    hnorm = np.sqrt(Hax * Hax + Hay * Hay)
    eps = 1e-6
    hscale = hnorm + eps
    inv_hscale = 1.0 / hscale

    # Helpers
    def add(name: str, arr: np.ndarray):
        cols.append(name)
        feats.append(arr)

    # Cached ms->frames
    _ms_to_frames_cache: dict[float, int] = {}

    def ms_to_frames(ms: int | float) -> int:
        key = float(ms)
        k = _ms_to_frames_cache.get(key)
        if k is None:
            k = int(round(key / dt_ms))
            k = max(1, k)
            _ms_to_frames_cache[key] = k
        return k

    def get_xy(src: NamedCoords, mapping: dict[str, int], name: str):
        j = mapping[name]
        return src.x[:, j : j + 1], src.y[:, j : j + 1]

    # Centers with per-frame fallback: body_center present but may be NaN; fallback to O + 0.5·H
    Cax, Cay = get_xy(agent, col_a, "body_center")
    Cbx, Cby = get_xy(target, col_b, "body_center")

    # Apply fallback where body_center is missing (either x or y is NaN)
    half = Hax.dtype.type(0.5)
    Afx, Afy = Oa.x + half * Hax, Oa.y + half * Hay
    Bfx, Bfy = Ob.x + half * Hb.x, Ob.y + half * Hb.y
    maskA = np.isnan(Cax) | np.isnan(Cay)
    maskB = np.isnan(Cbx) | np.isnan(Cby)
    if maskA.any():
        Cax = np.where(maskA, Afx, Cax)
        Cay = np.where(maskA, Afy, Cay)
    if maskB.any():
        Cbx = np.where(maskB, Bfx, Cbx)
        Cby = np.where(maskB, Bfy, Cby)

    # Utility: pair separation and normalised distance
    def sep_and_norm(Ax, Ay, Bx, By):
        dx, dy = Ax - Bx, Ay - By
        dist = np.sqrt(dx * dx + dy * dy)
        return dist, dist * inv_hscale

    # Utility: finite difference velocities v(t) = P(t) - P(t-1)
    def velocities(Px: np.ndarray, Py: np.ndarray):
        vx = np.empty_like(Px)
        vy = np.empty_like(Py)
        vx[0, :] = 0
        vy[0, :] = 0
        vx[1:, :] = Px[1:, :] - Px[:-1, :]
        vy[1:, :] = Py[1:, :] - Py[:-1, :]
        return vx, vy

    # (rolling helpers moved to module-level)

    # Utility: shift by k frames (positive = look back)
    def shift_trailing(x: np.ndarray, k: int):
        if k <= 0:
            if k == 0:
                return x
            # future shift
            kk = -k
            out = np.empty_like(x)
            out[...] = 0
            out[:-kk] = x[kk:]
            return out
        out = np.empty_like(x)
        out[...] = 0
        out[k:] = x[:-k]
        return out

    # 1) normalised pairwise distances for configured keypoint pairs
    if config.pairwise_distance_normalised:
        for ka, kb in config.pairwise_distance_normalised:
            Ax, Ay = get_xy(agent, col_a, ka)
            Bx, By = get_xy(target, col_b, kb)
            _, d_norm = sep_and_norm(Ax, Ay, Bx, By)
            add(f"pair_dist_{ka}__{kb}", d_norm)

    # 2) normalised approach rate for pairs and lags
    if config.pairwise_approach_rate and config.pairwise_approach_rate_lags_msec:
        lags = list(config.pairwise_approach_rate_lags_msec)
        for ka, kb in config.pairwise_approach_rate:
            Ax, Ay = get_xy(agent, col_a, ka)
            Bx, By = get_xy(target, col_b, kb)
            _, d_norm = sep_and_norm(Ax, Ay, Bx, By)
            D1, _ = precompute_derivatives_single(d_norm, lags, dt_ms)
            for ms in lags:
                add(f"pair_approachrate_{ka}__{kb}_lag{int(ms)}ms", D1[ms])

    # Centers and velocities used across multiple features
    d_c, d_c_norm = sep_and_norm(Cax, Cay, Cbx, Cby)
    vAx, vAy = velocities(Cax, Cay)
    vBx, vBy = velocities(Cbx, Cby)
    vAn = np.sqrt(vAx * vAx + vAy * vAy) + eps
    vBn = np.sqrt(vBx * vBx + vBy * vBy) + eps

    # 3) time to contact (seconds)
    if config.time_to_contact_sec:
        # derivative of normalised center distance per second
        # central difference with 1-frame step (approx): d'(t) ≈ (d(t) - d(t-1)) / dt
        dd = np.empty_like(d_c_norm)
        dd[0] = 0
        dd[1:] = d_c_norm[1:] - d_c_norm[:-1]
        ddt = dd / (dt_ms / 1000.0)  # 1/s
        closure_rate = ddt  # positive when separating, negative when closing
        ttc = np.minimum(10.0, safe_div(d_c_norm, np.maximum(eps, -closure_rate)))
        for ka, kb in config.time_to_contact_sec:
            add(f"pair_ttc_sec_{ka}__{kb}", ttc)

    # 4) facing angles (target→agent, agent→target)
    if config.facing_angles_target_to_agent:
        # Agent heading segment from Oa to Oa + Ha
        for kb in config.facing_angles_target_to_agent:
            Bx, By = get_xy(target, col_b, kb)
            seg2x, seg2y = Bx - Oa.x, By - Oa.y
            dot = (Hax * seg2x) + (Hay * seg2y)
            cross = (Hax * seg2y) - (Hay * seg2x)
            ang = np.arctan2(cross, dot)
            base = f"pair_facing_t2a_{kb}"
            add(base, ang)
            add(f"{base}_sin", np.sin(ang))
            add(f"{base}_cos", np.cos(ang))

    if (
        hasattr(config, "facing_angles_agent_to_target")
        and config.facing_angles_agent_to_target
    ):
        for ka in config.facing_angles_agent_to_target:
            Ax, Ay = get_xy(agent, col_a, ka)
            seg2x, seg2y = Ax - Ob.x, Ay - Ob.y
            dot = (Hb.x * seg2x) + (Hb.y * seg2y)
            cross = (Hb.x * seg2y) - (Hb.y * seg2x)
            ang = np.arctan2(cross, dot)
            base = f"pair_facing_a2t_{ka}"
            add(base, ang)
            add(f"{base}_sin", np.sin(ang))
            add(f"{base}_cos", np.cos(ang))

    # 5) rolling stats for distance^2
    if config.distance_stats_pairs and config.distance_stats_windows_msec:
        for ka, kb in config.distance_stats_pairs:
            Ax, Ay = get_xy(agent, col_a, ka)
            Bx, By = get_xy(target, col_b, kb)
            _, d_norm = sep_and_norm(Ax, Ay, Bx, By)
            d2 = d_norm * d_norm
            x1 = d2[:, 0]
            csum, csum2 = cumsums_1d(x1)
            for ms in config.distance_stats_windows_msec:
                w = ms_to_frames(ms)
                mean, std, var = rolling_mean_std_from_cumsums(csum, csum2, w)
                mn, mx = rolling_min_max(d2, w)
                add(f"pair_d2_{ka}__{kb}_win{int(ms)}ms_mean", mean)
                add(f"pair_d2_{ka}__{kb}_win{int(ms)}ms_std", std)
                add(f"pair_d2_{ka}__{kb}_win{int(ms)}ms_min", mn)
                add(f"pair_d2_{ka}__{kb}_win{int(ms)}ms_max", mx)
                one = var.dtype.type(1.0)
                add(
                    f"pair_d2_{ka}__{kb}_win{int(ms)}ms_int",
                    one / (one + var),
                )

    # 6) interaction continuity
    if (
        config.include_interaction_continuity
        and config.interaction_continuity_window_msec
    ):
        w = ms_to_frames(config.interaction_continuity_window_msec)
        d2 = d_c_norm * d_c_norm
        csum, csum2 = cumsums_1d(d2[:, 0])
        mean, std, _ = rolling_mean_std_from_cumsums(csum, csum2, w)
        add("pair_interaction_continuity", safe_div(std, mean + eps))

    # 7) velocity alignment with signed offsets (centers) via interpolation
    if config.velocity_alignment_offsets_msec:
        dot = vAx * vBx + vAy * vBy
        denom = vAn * vBn
        val = safe_div(dot, denom)
        for ms in config.velocity_alignment_offsets_msec:
            add(f"pair_valign_offset_{int(ms)}ms", interpolate_with_lag(val, ms, dt_ms))

    # 8) co-movement: windowed stats of dot product
    if config.velocity_dot_windows_msec:
        dot = vAx * vBx + vAy * vBy
        x1 = dot[:, 0]
        csum, csum2 = cumsums_1d(x1)
        for ms in config.velocity_dot_windows_msec:
            w = ms_to_frames(ms)
            mean, std, _ = rolling_mean_std_from_cumsums(csum, csum2, w)
            add(f"pair_co_win{int(ms)}ms_mean", mean)
            add(f"pair_co_win{int(ms)}ms_std", std)

    # 9) pursuit alignment along LOS
    if config.pursuit_alignment_windows_msec:
        relx, rely = Cax - Cbx, Cay - Cby
        r = np.sqrt(relx * relx + rely * rely)
        ux = safe_div(relx, r + eps)
        uy = safe_div(rely, r + eps)
        A_lead = safe_div(vAx * ux + vAy * uy, vAn)
        B_lead = safe_div(vBx * (-ux) + vBy * (-uy), vBn)
        csum_A, csum2_A = cumsums_1d(A_lead[:, 0])
        csum_B, csum2_B = cumsums_1d(B_lead[:, 0])
        for ms in config.pursuit_alignment_windows_msec:
            w = ms_to_frames(ms)
            mA, _, _ = rolling_mean_std_from_cumsums(csum_A, csum2_A, w)
            mB, _, _ = rolling_mean_std_from_cumsums(csum_B, csum2_B, w)
            add(f"pair_pursuit_Alead_win{int(ms)}ms_mean", mA)
            add(f"pair_pursuit_Blead_win{int(ms)}ms_mean", mB)

    # 10) chase score
    if config.chase_window_msec is not None:
        # approach_norm = -(r_norm(t) - r_norm(t-1))
        r_norm = d_c * inv_hscale
        dr = np.empty_like(r_norm)
        dr[0] = 0
        dr[1:] = r_norm[1:] - r_norm[:-1]
        approach_norm = -dr
        # reuse B_lead from pursuit alignment definition
        relx, rely = Cax - Cbx, Cay - Cby
        r = np.sqrt(relx * relx + rely * rely)
        ux = safe_div(relx, r + eps)
        uy = safe_div(rely, r + eps)
        B_lead = safe_div(vBx * (-ux) + vBy * (-uy), vBn)
        chase_inst = approach_norm * B_lead
        w = ms_to_frames(config.chase_window_msec)
        csum, csum2 = cumsums_1d(chase_inst[:, 0])
        m, _, _ = rolling_mean_std_from_cumsums(csum, csum2, w)
        add(f"pair_chase_win{int(config.chase_window_msec)}ms_mean", m)

    # 11) speed correlation windows
    if config.speed_correlation_windows_msec:
        sA = np.sqrt(vAx * vAx + vAy * vAy)
        sB = np.sqrt(vBx * vBx + vBy * vBy)
        sAB = sA * sB
        csum_A, csum2_A = cumsums_1d(sA[:, 0])
        csum_B, csum2_B = cumsums_1d(sB[:, 0])
        csum_AB, csum2_AB = cumsums_1d(sAB[:, 0])
        for ms in config.speed_correlation_windows_msec:
            w = ms_to_frames(ms)
            mA, _, varA = rolling_mean_std_from_cumsums(csum_A, csum2_A, w)
            mB, _, varB = rolling_mean_std_from_cumsums(csum_B, csum2_B, w)
            mAB, _, _ = rolling_mean_std_from_cumsums(csum_AB, csum2_AB, w)
            denom = np.sqrt(varA) * np.sqrt(varB) + eps
            cov = mAB - mA * mB
            add(f"pair_speedcorr_win{int(ms)}ms", safe_div(cov, denom))

    # 12) nose–nose temporal dynamics
    if config.nose_nose_lags_msec:
        Nax, Nay = get_xy(agent, col_a, "nose")
        Nbx, Nby = get_xy(target, col_b, "nose")
        _, nn_norm = sep_and_norm(Nax, Nay, Nbx, Nby)
        nn_lagged = {
            float(ms): interpolate_with_lag(nn_norm, float(ms), dt_ms)
            for ms in config.nose_nose_lags_msec
        }
        for ms, lagged in nn_lagged.items():
            add(f"pair_nn_lag{int(ms)}ms", lagged)
            add(f"pair_nn_change{int(ms)}ms", nn_norm - lagged)
            if (
                config.include_nose_nose_close_proportion
                and config.nose_nose_close_threshold_norm is not None
            ):
                thr = float(config.nose_nose_close_threshold_norm)
                mask = (nn_norm[:, 0] < thr).astype(np.float32)[:, None]
                # rolling mean of mask over window corresponding to this lag (in frames)
                w = ms_to_frames(ms)
                csum, csum2 = cumsums_1d(mask[:, 0])
                mean, _, _ = rolling_mean_std_from_cumsums(csum, csum2, w)
                add(f"pair_nn_closeprop{int(ms)}ms", mean)

    # 13) body-axis alignment
    if config.include_body_axis_alignment:
        Anx, Any = get_xy(agent, col_a, "nose")
        Atx, Aty = get_xy(agent, col_a, "tail_base")
        Bnx, Bny = get_xy(target, col_b, "nose")
        Btx, Bty = get_xy(target, col_b, "tail_base")
        aAx, aAy = Anx - Atx, Any - Aty
        aBx, aBy = Bnx - Btx, Bny - Bty
        dot = aAx * aBx + aAy * aBy
        nA = np.sqrt(aAx * aAx + aAy * aAy) + eps
        nB = np.sqrt(aBx * aBx + aBy * aBy) + eps
        add("pair_body_axis_alignment", safe_div(dot, nA * nB))

    data = np.concatenate(feats, axis=1)
    return NamedFeatures(columns=cols, data=data)
