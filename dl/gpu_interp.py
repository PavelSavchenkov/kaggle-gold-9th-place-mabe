import torch


def _prev_valid_index(valid_mask: torch.Tensor) -> torch.Tensor:
    """
    valid_mask: [N, T] bool
    Returns prev_idx: [N, T] with the index of the last True at or before t,
    or -1 if none exists.
    """
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must be [N, T]")

    N, T = valid_mask.shape
    device = valid_mask.device

    idx = torch.arange(T, device=device).view(1, T).expand(N, T)
    valid_idx = idx.masked_fill(~valid_mask, -1)  # -1 where invalid
    prev_idx, _ = valid_idx.cummax(dim=1)
    return prev_idx


def _next_valid_index(valid_mask: torch.Tensor) -> torch.Tensor:
    """
    valid_mask: [N, T] bool
    Returns next_idx: [N, T] with the index of the first True at or after t,
    or -1 if none exists.
    """
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must be [N, T]")

    # Compute "previous valid" in reversed time, then map back.
    valid_rev = valid_mask.flip(1)  # [N, T]
    prev_rev = _prev_valid_index(valid_rev)  # [N, T] in reversed coords
    T = valid_mask.size(1)

    # prev_rev is index in reversed coordinates; map back:
    # -1 stays -1, others map via (T-1 - j')
    j_prime = prev_rev.flip(1)
    next_idx = torch.where(
        j_prime >= 0,
        (T - 1) - j_prime,
        torch.full_like(j_prime, -1),
    )
    return next_idx


def interpolate_missing_along_time_(
    coords: torch.Tensor,  # [B, T, K]
    missing_mask: torch.Tensor,  # [B, T, K], True where we want to replace with interp
):
    """
    In-place linear interpolation over time for `coords` where `missing_mask` is True.

    For each (b, k) independently:
      - Treat coords[b, :, k] as a 1D signal of length T.
      - Frames with missing_mask[b, t, k] == True are replaced by:
          * linear interpolation between nearest non-missing frames if they exist on
            both sides, OR
          * the closest non-missing value if only on one side.
      - If all frames are missing for that (b, k), masked positions are set to 0.

    This is fully vectorized over (B * K) and uses no Python loops over B/T/K.
    """
    if coords.shape != missing_mask.shape:
        raise ValueError("coords and missing_mask must have the same shape [B, T, K]")

    if coords.ndim != 3:
        raise ValueError("coords must be 3D [B, T, K]")

    B, T, K = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Flatten (B, K) -> N
    # x, m are views into coords, so writes into x modify coords.
    x = coords.permute(0, 2, 1).reshape(-1, T)  # [N, T]
    m = missing_mask.permute(0, 2, 1).reshape(-1, T)  # [N, T] bool

    if not m.any():
        return

    N = x.size(0)
    valid = ~m  # [N, T]
    any_valid = valid.any(dim=1)  # [N]

    rows_with_anchor = any_valid  # rows with at least one non-missing frame
    rows_no_anchor = ~any_valid  # rows where everything is missing

    # --- Case 1: rows with no anchors at all -> fill masked positions with 0.0 ---
    if rows_no_anchor.any():
        x[rows_no_anchor] = torch.where(
            m[rows_no_anchor],
            torch.zeros_like(x[rows_no_anchor]),
            x[rows_no_anchor],
        )

    # --- Case 2: rows that have at least one valid frame -> real interpolation ---
    if rows_with_anchor.any():
        rw = rows_with_anchor.nonzero(as_tuple=False).squeeze(-1)  # [M]
        xv = x[rw]  # [M, T]
        mv = m[rw]  # [M, T]
        validv = ~mv  # [M, T]

        prev_idx = _prev_valid_index(validv)  # [M, T], -1 or index
        next_idx = _next_valid_index(validv)  # [M, T], -1 or index

        # Time coordinate (as float for interpolation)
        t = torch.arange(T, device=device, dtype=dtype).view(1, T).expand_as(xv)

        # For gather, clamp indices into [0, T-1]; we’ll ignore invalid ones via masks
        left_clamped = prev_idx.clamp(min=0)
        # If there is no "next" (next_idx < 0), we reuse prev_idx for the right side
        right_clamped = torch.where(next_idx >= 0, next_idx, prev_idx.clamp(min=0))

        left_vals = torch.gather(xv, 1, left_clamped)  # [M, T]
        right_vals = torch.gather(xv, 1, right_clamped)  # [M, T]

        # Region masks:
        #  - left_extrap: no previous good point, but some next one
        #  - right_extrap: no next good point, but some previous one
        #  - interior: we have both, and they are different indices
        left_extrap = (prev_idx < 0) & (next_idx >= 0)
        right_extrap = (next_idx < 0) & (prev_idx >= 0)
        interior = (prev_idx >= 0) & (next_idx >= 0) & (next_idx != prev_idx)

        outv = xv.clone()

        # Left extrapolation: use right_vals
        if left_extrap.any():
            outv[left_extrap] = right_vals[left_extrap]

        # Right extrapolation: use left_vals
        if right_extrap.any():
            outv[right_extrap] = left_vals[right_extrap]

        # Interior: linear interpolation
        if interior.any():
            li = prev_idx.to(dtype)[interior]  # left indices as float
            ri = next_idx.to(dtype)[interior]  # right indices as float
            ti = t[interior]  # time indices at those positions
            w = (ti - li) / (ri - li)
            outv[interior] = (
                left_vals[interior] + (right_vals[interior] - left_vals[interior]) * w
            )

        # Only overwrite coordinates at positions marked missing
        fill_mask = mv  # [M, T]
        xv = torch.where(fill_mask, outv, xv)
        x[rw] = xv

    # coords is already updated through the view `x`
