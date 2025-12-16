from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch  # type: ignore
import torch.nn as nn  # type: ignore

from common.constants import get_opposite_bodypart
from dl.configs import AugmentationsConfig, FeaturesConfigDL
from dl.gpu_interp import interpolate_missing_along_time_


@dataclass
class BodyCS:
    """
    Body-centered coordinate system defined by origin O and heading H.

    All tensors are [B, 1, 1] so they broadcast cleanly against [B, T, K].

    - Origin O: keypoint `origin_name` at the *center frame* of the window
    - Heading H: mean(heading_left, heading_right) - origin at center frame

    This CS is then broadcast over the whole window [0..T-1], so we don't
    normalise each frame independently (we keep true movement across time).
    """

    O_x: torch.Tensor  # [B, 1, 1]
    O_y: torch.Tensor  # [B, 1, 1]
    H_x: torch.Tensor  # [B, 1, 1]
    H_y: torch.Tensor  # [B, 1, 1]
    H_n: torch.Tensor  # [B, 1, 1]  (||H||, clamped with eps)

    @staticmethod
    def from_coords(
        x: torch.Tensor,  # [B, T, K]
        y: torch.Tensor,  # [B, T, K]
        keypoint_names: list[str],
        origin_name: str,
        heading_left: str,
        heading_right: str,
        eps: float = 1e-6,
    ) -> BodyCS:
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError("x and y must be [B, T, K]")
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")

        B, T, K = x.shape
        if len(keypoint_names) != K:
            raise ValueError("len(keypoint_names) must match K dimension")

        idx = {name: i for i, name in enumerate(keypoint_names)}

        o_idx = idx[origin_name]
        hl_idx = idx[heading_left]
        hr_idx = idx[heading_right]

        # center frame index
        assert T % 2 == 1
        t0 = T // 2

        # origin at center frame
        O_x_c = x[:, t0, o_idx]  # [B]
        O_y_c = y[:, t0, o_idx]

        # heading from ears at center frame
        hl_x_c = x[:, t0, hl_idx]
        hl_y_c = y[:, t0, hl_idx]
        hr_x_c = x[:, t0, hr_idx]
        hr_y_c = y[:, t0, hr_idx]

        ear_x_c = 0.5 * (hl_x_c + hr_x_c)
        ear_y_c = 0.5 * (hl_y_c + hr_y_c)

        H_x_c = ear_x_c - O_x_c
        H_y_c = ear_y_c - O_y_c
        H_n_c = torch.sqrt(H_x_c * H_x_c + H_y_c * H_y_c + eps)

        # reshape to [B, 1, 1] so they broadcast over T and K
        O_x = O_x_c.view(B, 1, 1)
        O_y = O_y_c.view(B, 1, 1)
        H_x = H_x_c.view(B, 1, 1)
        H_y = H_y_c.view(B, 1, 1)
        H_n = H_n_c.view(B, 1, 1)

        return BodyCS(O_x=O_x, O_y=O_y, H_x=H_x, H_y=H_y, H_n=H_n)

    def project(
        self,
        x: torch.Tensor,  # [B, T, K]
        y: torch.Tensor,  # [B, T, K]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project arena coords (x, y) into this body CS.

        x_body: forward
        y_body: left
        """
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")

        rel_x = x - self.O_x  # [B, T, K]
        rel_y = y - self.O_y

        # rotate by heading, normalised by ||H||
        x_b = (rel_x * self.H_x + rel_y * self.H_y) / self.H_n
        y_b = (-rel_x * self.H_y + rel_y * self.H_x) / self.H_n
        return x_b, y_b


def central_diff(x: torch.Tensor) -> torch.Tensor:
    """
    Central difference along time dimension (dim=1).

    x: [B, T, ...]
    Returns: same shape, with zeros at the boundaries.
    """
    if x.ndim < 2:
        raise ValueError("central_diff expects at least 2D tensor [B, T, ...]")

    dx = torch.zeros_like(x)
    dx[:, 1:-1] = 0.5 * (x[:, 2:] - x[:, :-2])
    return dx


def first_diff(x: torch.Tensor) -> torch.Tensor:
    """
    Forward difference along time (dim=1).

    x: [B, T, ...]
    Returns: same shape, with zeros at t=0.
    """
    if x.ndim < 2:
        raise ValueError("first_diff expects at least 2D tensor [B, T, ...]")

    dx = torch.zeros_like(x)
    dx[:, 1:] = x[:, 1:] - x[:, :-1]
    return dx


def angle_from_xy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Angle (radians) for vectors (x, y).
    """
    return torch.atan2(y, x)


def angle_trig(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (angle, sin(angle), cos(angle)) for vectors (x, y).
    """
    angle = angle_from_xy(x, y)
    sin_a = torch.sin(angle)
    cos_a = torch.cos(angle)
    return angle, sin_a, cos_a


def angle_between(
    u_x: torch.Tensor,
    u_y: torch.Tensor,
    v_x: torch.Tensor,
    v_y: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Angle between u and v in 2D, using atan2(cross, dot).

    Returns (angle, sin(angle), cos(angle)).
    All inputs shaped [B, T].
    """
    dot = u_x * v_x + u_y * v_y
    cross = u_x * v_y - u_y * v_x
    denom = torch.sqrt((u_x * u_x + u_y * u_y + eps) * (v_x * v_x + v_y * v_y + eps))

    cos_a = dot / (denom + eps)
    sin_a = cross / (denom + eps)
    angle = torch.atan2(sin_a, cos_a)
    return angle, sin_a, cos_a


class LandmarkFeatureExtractor(nn.Module):
    """
    Converts raw windowed landmarks to handcrafted features
    """

    def __init__(self, feats_config: FeaturesConfigDL, aug_config: AugmentationsConfig):
        super().__init__()
        self.feats_config = feats_config
        self.aug_config = aug_config
        self.eps = 1e-6
        self.must_force_hflip = False

        # assume: K dimension order == config.bodyparts, by construction
        self.kp_index = {name: i for i, name in enumerate(self.feats_config.bodyparts)}

        K = len(feats_config.bodyparts)
        perm = list(range(K))
        for bodypart in self.feats_config.bodyparts:
            opposite = get_opposite_bodypart(bodypart)
            assert opposite in self.kp_index
            idx = self.kp_index[bodypart]
            new_idx = self.kp_index[opposite]
            perm[idx] = new_idx
        assert set(perm) == set(list(range(K)))

        self.register_buffer("flip_sides_perm", torch.tensor(perm, dtype=torch.long))

        body_width_indices: list[int] = []
        for name in self.feats_config.bodyparts:
            opposite = get_opposite_bodypart(name)
            assert opposite in self.kp_index
            if opposite != name:
                body_width_indices.append(self.kp_index[name])
        self.body_width_indices = sorted(set(body_width_indices))

        if self.aug_config.bodypart_dropout_config is not None:
            self.bodypart_group_names = []
            masks = []
            for (
                group_name,
                list_names,
            ) in self.aug_config.bodypart_dropout_config.groups_mapping.items():
                mask = torch.zeros(K, dtype=torch.bool)
                for name in list_names:
                    if name in self.kp_index:
                        mask[self.kp_index[name]] = True
                if mask.any():
                    self.bodypart_group_names.append(group_name)
                    masks.append(mask)
            assert (
                masks
            ), f"either remove bodypart dropout or add bodyparts which could be dropped out"
            masks = torch.stack(masks, dim=0)  # [N_GROUPS, K]
            self.register_buffer("bodypart_group_masks", masks, persistent=False)

    def _single_mouse_feats(
        self,
        x_b: torch.Tensor,  # [B,T,K]
        y_b: torch.Tensor,  # [B,T,K]
        body_cs: BodyCS,  # for heading angle in arena coords
        is_agent: bool = False,
    ) -> torch.Tensor:  # returns [B,T,D_single]
        single_cfg = self.feats_config.single_mouse_numeric
        idx = self.kp_index
        B, T, K = x_b.shape

        chunks: list[torch.Tensor] = []

        # normalised coords
        norm_indices = [
            idx[name] for name in single_cfg.normalised_keypoints if name in idx
        ]
        x_norm = x_b[:, :, norm_indices]
        y_norm = y_b[:, :, norm_indices]
        chunks.append(x_norm)
        chunks.append(y_norm)

        # velocities / speed
        v_x = central_diff(x_norm)
        v_y = central_diff(y_norm)
        if single_cfg.include_normalised_velocity:
            chunks.append(v_x)
            chunks.append(v_y)
        if single_cfg.include_normalised_speed:
            speed = torch.sqrt(v_x * v_x + v_y * v_y + self.eps)
            chunks.append(speed)

        # intra-mouse distances (body CS)
        if single_cfg.intra_distance_pairs:
            for name_a, name_b in single_cfg.intra_distance_pairs:
                ia = idx[name_a]
                ib = idx[name_b]
                ax = x_b[:, :, ia]  # [B, T]
                ay = y_b[:, :, ia]
                bx = x_b[:, :, ib]
                by = y_b[:, :, ib]
                dx = bx - ax
                dy = by - ay
                dist = torch.sqrt(dx * dx + dy * dy + self.eps)  # [B, T]
                chunks.append(dist.unsqueeze(-1))

        # ear distance + temporal change
        if single_cfg.include_ear_distance:
            ileft = idx["ear_left"]
            iright = idx["ear_right"]

            ex = x_b[:, :, ileft]
            ey = y_b[:, :, ileft]
            rx = x_b[:, :, iright]
            ry = y_b[:, :, iright]

            ear_d = torch.sqrt((ex - rx) * (ex - rx) + (ey - ry) * (ey - ry) + self.eps)
            chunks.append(ear_d.unsqueeze(-1))  # [B, T, 1]

            if single_cfg.include_ear_distance_velocity:
                ear_d_vel = central_diff(ear_d.unsqueeze(-1))  # [B, T, 1]
                chunks.append(ear_d_vel)

        # simple shape angles: ear splay
        if single_cfg.include_ear_splay:
            inose = idx["nose"]
            ileft = idx["ear_left"]
            iright = idx["ear_right"]

            # vectors: nose -> ear_left / ear_right
            vx_l = x_b[:, :, ileft] - x_b[:, :, inose]  # [B, T]
            vy_l = y_b[:, :, ileft] - y_b[:, :, inose]
            vx_r = x_b[:, :, iright] - x_b[:, :, inose]
            vy_r = y_b[:, :, iright] - y_b[:, :, inose]

            angle_es, sin_es, cos_es = angle_between(
                vx_l, vy_l, vx_r, vy_r, eps=self.eps
            )  # [B, T] each
            chunks.append(angle_es.unsqueeze(-1))
            chunks.append(sin_es.unsqueeze(-1))
            chunks.append(cos_es.unsqueeze(-1))

        # simple shape angles: torso bend
        if single_cfg.include_torso_bend:
            ineck = idx["neck"]
            il = idx["hip_left"]
            ir = idx["hip_right"]

            vx_l = x_b[:, :, il] - x_b[:, :, ineck]
            vy_l = y_b[:, :, il] - y_b[:, :, ineck]
            vx_r = x_b[:, :, ir] - x_b[:, :, ineck]
            vy_r = y_b[:, :, ir] - y_b[:, :, ineck]

            angle_tb, sin_tb, cos_tb = angle_between(
                vx_l, vy_l, vx_r, vy_r, eps=self.eps
            )
            chunks.append(angle_tb.unsqueeze(-1))
            chunks.append(sin_tb.unsqueeze(-1))
            chunks.append(cos_tb.unsqueeze(-1))

        if is_agent:
            # curvature of selected keypoints in body CS
            if single_cfg.curvature_keypoints:
                H_n = body_cs.H_n.view(B, 1, 1)  # [B, 1, 1]
                for name in single_cfg.curvature_keypoints:
                    j = idx[name]
                    xk = x_b[:, :, j : j + 1]  # [B, T, 1]
                    yk = y_b[:, :, j : j + 1]
                    vx = central_diff(xk)  # [B, T, 1]
                    vy = central_diff(yk)
                    ax = central_diff(vx)
                    ay = central_diff(vy)
                    vnorm3 = (vx * vx + vy * vy + self.eps) ** 1.5
                    cross = vx * ay - vy * ax
                    kappa = cross / (vnorm3 + self.eps)
                    kappa_dimless = kappa * H_n  # dimensionless
                    chunks.append(kappa_dimless)  # [B, T, 1]

            # head vs tail-axis coupling
            if single_cfg.include_body_axis_coupling:
                itail = idx["tail_base"]
                inose = idx["nose"]
                itip = idx["tail_tip"]

                v_head_x = x_b[:, :, inose] - x_b[:, :, itail]  # [B, T]
                v_head_y = y_b[:, :, inose] - y_b[:, :, itail]
                v_tail_x = x_b[:, :, itip] - x_b[:, :, itail]
                v_tail_y = y_b[:, :, itip] - y_b[:, :, itail]

                ang_ba, sin_ba, cos_ba = angle_between(
                    v_head_x, v_head_y, v_tail_x, v_tail_y, eps=self.eps
                )
                chunks.append(ang_ba.unsqueeze(-1))
                chunks.append(sin_ba.unsqueeze(-1))
                chunks.append(cos_ba.unsqueeze(-1))

            # scalar body length (||H||), broadcast over time
            if single_cfg.include_body_length:
                bl = body_cs.H_n.view(B, 1, 1).expand(B, T, 1)  # [B, T, 1]
                chunks.append(bl)
            if (
                single_cfg.include_heading_angle
                or single_cfg.include_heading_angle_trig
            ):
                Ha_x = body_cs.H_x.view(B)
                Ha_y = body_cs.H_y.view(B)
                theta0 = angle_from_xy(Ha_x, Ha_y)
                theta = theta0.view(B, 1, 1).expand(B, T, 1)
                if single_cfg.include_heading_angle:
                    chunks.append(theta)
                if single_cfg.include_heading_angle_trig:
                    chunks.append(torch.sin(theta))
                    chunks.append(torch.cos(theta))

            # lagged nose displacement(s)
            if (
                single_cfg.include_nose_lag_displacements
                and single_cfg.nose_lag_displacement_frames
            ):
                inose = idx["nose"]
                nose_x = x_b[:, :, inose : inose + 1]  # [B, T, 1]
                nose_y = y_b[:, :, inose : inose + 1]
                H_n = body_cs.H_n.view(B, 1, 1)  # [B, 1, 1]
                H_n2 = H_n * H_n

                for lag in single_cfg.nose_lag_displacement_frames:
                    if lag <= 0 or lag >= T:
                        continue
                    # simple forward displacement over 'lag' frames
                    dx = torch.zeros_like(nose_x)
                    dy = torch.zeros_like(nose_y)
                    dx[:, lag:] = nose_x[:, lag:] - nose_x[:, :-lag]
                    dy[:, lag:] = nose_y[:, lag:] - nose_y[:, :-lag]

                    disp2 = dx * dx + dy * dy
                    chunks.append(disp2)  # [B, T, 1]

        return torch.cat(chunks, dim=-1)

    def _body_width_augment_single(
        self,
        x: torch.Tensor,  # [B, T, K]
        y: torch.Tensor,  # [B, T, K]
        scales: torch.Tensor,  # [B], per-sample body-width scale
    ):
        """
        In-place: for each sample b and each time t, rescale distance of
        selected keypoints to the body axis (nose -> tail_base) by scales[b].
        """
        B = x.size(0)

        inose = self.kp_index["nose"]
        itail = self.kp_index["tail_base"]

        # body axis for each (b,t): tail_base -> nose
        x_head = x[:, :, inose]  # [B, T]
        y_head = y[:, :, inose]
        x_tail = x[:, :, itail]
        y_tail = y[:, :, itail]

        ax = x_head - x_tail  # [B, T]
        ay = y_head - y_tail
        denom = torch.sqrt(ax * ax + ay * ay + self.eps)
        ux = ax / (denom + self.eps)  # unit axis vector
        uy = ay / (denom + self.eps)

        # [B] -> [B, 1] so it broadcasts along time
        scale_bt = scales.view(B, 1)  # [B, 1]

        for j in self.body_width_indices:
            px = x[:, :, j]  # [B, T]
            py = y[:, :, j]

            # vector from tail_base to point
            APx = px - x_tail  # [B, T]
            APy = py - y_tail

            # projection length onto axis and projected point on axis
            proj_len = APx * ux + APy * uy  # [B, T]
            proj_x = x_tail + proj_len * ux  # [B, T]
            proj_y = y_tail + proj_len * uy

            # offset from axis
            dx = px - proj_x  # [B, T]
            dy = py - proj_y

            # rescale distance to axis by per-sample factor
            px_new = proj_x + dx * scale_bt  # [B, T]
            py_new = proj_y + dy * scale_bt

            x[:, :, j] = px_new
            y[:, :, j] = py_new

    def _body_width_augment_agent_target(
        self,
        agent_x: torch.Tensor,
        agent_y: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
    ):
        cfg = self.aug_config.body_width_config
        assert cfg

        B = agent_x.size(0)
        dev = agent_x.device

        scales = torch.ones(B, device=dev)  # [B]

        apply_mask = torch.rand(B, device=dev) < cfg.apply_prob  # [B]
        scales_aug = torch.empty(B, device=dev).uniform_(cfg.L, cfg.R)  # [B]
        scales[apply_mask] = scales_aug[apply_mask]

        self._body_width_augment_single(agent_x, agent_y, scales)
        self._body_width_augment_single(target_x, target_y, scales)

    def _bodypart_dropout_mouse_single(
        self,
        x,
        y,
        who: str,  # "agent" or "target"
        batch: dict[str, torch.Tensor],
    ) -> None:
        """
        In-place body-part dropout for a single mouse.

        For each sample b in the batch:
          - with probability cfg.apply_prob:
              * choose n_groups[b] in [cfg.min_groups, cfg.max_groups]
              * randomly pick that many groups (without replacement)
              * zero x[b, :, k], y[b, :, k] for all keypoints k in those groups
                across the entire window
              * if cfg.update_masks is True, also set {who}_x_mask and {who}_y_mask
                to 1.0 at the same positions.
        """
        cfg = self.aug_config.bodypart_dropout_config
        assert cfg is not None

        B, T, K = x.shape
        G = self.bodypart_group_masks.size(0)
        dev = x.device

        # clamp min/max groups to what we actually have
        max_g = int(min(cfg.max_groups, G))
        min_g = int(max(1, min(cfg.min_groups, max_g)))
        assert min_g <= max_g, f"min_g: {min_g}, max_g: {max_g}"

        # which samples to apply dropout to
        apply = torch.rand(B, device=dev) < cfg.apply_prob  # [B]
        if not apply.any():
            return

        # random scores for each group in each sample -> top max_g groups
        scores = torch.rand(B, G, device=dev)  # [B, G]
        top_idx = scores.topk(k=max_g, dim=1).indices  # [B, max_g]

        # how many groups per sample (between min_g and max_g)
        n_groups = torch.randint(
            low=min_g,
            high=max_g + 1,
            size=(B,),
            device=dev,
        )  # [B]

        # build [B, G] boolean mask of selected groups using scatter
        group_sel = torch.zeros(B, G, dtype=torch.bool, device=dev)  # [B, G]

        if max_g == 1:
            # simple case: only one candidate group per sample
            group_sel[apply, top_idx[apply, 0]] = True
        else:
            # rank positions 0..max_g-1
            rank = (
                torch.arange(max_g, device=dev).unsqueeze(0).expand(B, max_g)
            )  # [B, max_g]
            use = rank < n_groups.unsqueeze(1)  # [B, max_g] bool
            use = use & apply.unsqueeze(1)  # mask non-applied samples

            # scatter 'use' flags to group positions
            group_sel.scatter_(1, top_idx, use)  # [B, G]

        if not group_sel.any():
            return

        # Combine selected groups with group->keypoint mask
        # group_kp_mask: [G, K] -> [1, G, K]
        # group_sel:     [B, G] -> [B, G, 1]
        group_kp_mask = self.bodypart_group_masks.to(device=dev)
        drop_b_g_k = group_sel.unsqueeze(-1) & group_kp_mask.unsqueeze(0)  # [B, G, K]
        drop_bk = drop_b_g_k.any(dim=1)  # [B, K]

        if not drop_bk.any():
            return

        # expand over time: [B, 1, K] -> [B, T, K]
        drop_mask = drop_bk.unsqueeze(1).expand(B, T, K)  # [B, T, K]

        # zero coords in-place
        x[drop_mask] = 0.0
        y[drop_mask] = 0.0

        # optionally set masks to 1.0 where we dropped
        if cfg.update_masks:
            x_mask_name = f"{who}_x_mask"
            y_mask_name = f"{who}_y_mask"

            xm = batch[x_mask_name]
            ym = batch[y_mask_name]

            # ensure on same device
            if xm.device != dev:
                xm = xm.to(dev)
                batch[x_mask_name] = xm
            if ym.device != dev:
                ym = ym.to(dev)
                batch[y_mask_name] = ym

            xm[drop_mask] = 1.0
            ym[drop_mask] = 1.0

    def _spatial_cutout(self, agent_x, agent_y, target_x, target_y, batch):
        cfg = self.aug_config.spatial_cutout_config
        assert cfg is not None

        B, T, K = agent_x.shape
        dev = agent_x.device

        apply_mask = torch.rand(B, 1, 1, device=dev) < cfg.apply_prob
        if not apply_mask.any():
            return

        all_x = torch.cat([agent_x, target_x], dim=2)
        all_y = torch.cat([agent_y, target_y], dim=2)

        flat_x = all_x.view(B, -1)
        flat_y = all_y.view(B, -1)

        min_x = flat_x.amin(dim=1, keepdim=True).view(B, 1, 1)
        max_x = flat_x.amax(dim=1, keepdim=True).view(B, 1, 1)
        min_y = flat_y.amin(dim=1, keepdim=True).view(B, 1, 1)
        max_y = flat_y.amax(dim=1, keepdim=True).view(B, 1, 1)

        eps = 1e-6
        range_x = torch.clamp(max_x - min_x, min=eps)
        range_y = torch.clamp(max_y - min_y, min=eps)

        # === sample rectangles ===
        u_w = torch.rand(B, 1, 1, device=dev)
        u_h = torch.rand(B, 1, 1, device=dev)

        frac_min, frac_max = cfg.min_size_ratio, cfg.max_size_ratio
        frac_w = frac_min + (frac_max - frac_min) * u_w
        frac_h = frac_min + (frac_max - frac_min) * u_h

        width = range_x * frac_w  # [B, 1, 1]
        height = range_y * frac_h  # [B, 1, 1]

        # centers inside bbox
        u_cx = torch.rand(B, 1, 1, device=dev)
        u_cy = torch.rand(B, 1, 1, device=dev)
        cx = min_x + range_x * u_cx
        cy = min_y + range_y * u_cy

        half_w = 0.5 * width
        half_h = 0.5 * height

        # === build masks for cutout ===
        x_agent_inside = (agent_x >= (cx - half_w)) & (agent_x <= (cx + half_w))
        y_agent_inside = (agent_y >= (cy - half_h)) & (agent_y <= (cy + half_h))
        x_target_inside = (target_x >= (cx - half_w)) & (target_x <= (cx + half_w))
        y_target_inside = (target_y >= (cy - half_h)) & (target_y <= (cy + half_h))

        mask_cutout_agent = x_agent_inside & y_agent_inside & apply_mask
        mask_cutout_target = x_target_inside & y_target_inside & apply_mask

        if not (mask_cutout_agent.any() or mask_cutout_target.any()):
            return

        # === apply cutout ===
        if not cfg.use_interp:
            agent_x[mask_cutout_agent] = 0.0
            agent_y[mask_cutout_agent] = 0.0
            target_x[mask_cutout_target] = 0.0
            target_y[mask_cutout_target] = 0.0
        else:
            interpolate_missing_along_time_(agent_x, mask_cutout_agent)
            interpolate_missing_along_time_(agent_y, mask_cutout_agent)
            interpolate_missing_along_time_(target_x, mask_cutout_target)
            interpolate_missing_along_time_(target_y, mask_cutout_target)

        if cfg.update_masks:
            batch["agent_x_mask"][mask_cutout_agent] = 1.0
            batch["agent_y_mask"][mask_cutout_agent] = 1.0

            batch["target_x_mask"][mask_cutout_target] = 1.0
            batch["target_y_mask"][mask_cutout_target] = 1.0

    def force_hflip(self):
        self.must_force_hflip = True

    def remove_force_hflip(self):
        assert self.must_force_hflip
        self.must_force_hflip = False

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        agent_x = batch["agent_x"]
        agent_y = batch["agent_y"]
        target_x = batch["target_x"]
        target_y = batch["target_y"]

        B, T, K = agent_x.shape
        if (
            agent_y.shape != (B, T, K)
            or target_x.shape != (B, T, K)
            or target_y.shape != (B, T, K)
        ):
            raise ValueError("agent/target x,y must all be [B, T, K] and match")

        if K != len(self.feats_config.bodyparts):
            raise ValueError(
                f"K={K} must match len(config.bodyparts)={len(self.feats_config.bodyparts)}"
            )

        def do_flip_sides(tensors, mask):
            for arr in tensors:
                arr_flipped = arr[mask]
                arr[mask] = arr_flipped[:, :, self.flip_sides_perm]

        agent_mask_names = ["agent_x_mask", "agent_y_mask"]
        target_mask_names = ["target_x_mask", "target_y_mask"]

        if self.must_force_hflip:
            assert not self.training
            agent_x = -agent_x
            target_x = -target_x
            agent_y = agent_y.clone()
            target_y = target_y.clone()
            tensors = [agent_x, agent_y, target_x, target_y]
            if self.aug_config.flip_masks_if_flip_sides:
                for name in agent_mask_names + target_mask_names:
                    batch[name] = batch[name].clone()
                    tensors.append(batch[name])
            do_flip_sides(
                tensors, mask=torch.ones(B, device=agent_x.device, dtype=torch.bool)
            )

        # ===== Augmentations =====
        if self.training:
            # --- hflip ---
            if self.aug_config.hflip:
                flip_mask = torch.rand(B, device=agent_x.device) < 0.5
                sign = (1.0 - 2.0 * flip_mask.to(agent_x.dtype)).view(B, 1, 1)
                agent_x = agent_x * sign
                target_x = target_x * sign
                if self.aug_config.flip_sides_if_flip_coords:
                    tensors = [agent_x, agent_y, target_x, target_y]
                    if self.aug_config.flip_masks_if_flip_sides:
                        for name in agent_mask_names + target_mask_names:
                            tensors.append(batch[name])
                    do_flip_sides(tensors, mask=flip_mask)
            # random flip sides in coords
            if self.aug_config.random_flip_sides_agent:
                flip_mask_agent = torch.rand(B, device=agent_x.device) < 0.5
                tensors = [agent_x, agent_y]
                if self.aug_config.flip_masks_if_flip_sides:
                    for name in agent_mask_names:
                        tensors.append(batch[name])
                do_flip_sides(tensors, mask=flip_mask_agent)
            if self.aug_config.random_flip_sides_target:
                flip_mask_target = torch.rand(B, device=target_x.device) < 0.5
                tensors = [target_x, target_y]
                if self.aug_config.flip_masks_if_flip_sides:
                    for name in target_mask_names:
                        tensors.append(batch[name])
                do_flip_sides(tensors, mask=flip_mask_target)
            # random flip sides in masks
            if self.aug_config.random_flip_sides_agent_masks:
                flip_mask_agent = torch.rand(B, device=agent_x.device) < 0.5
                do_flip_sides(
                    [batch[name] for name in agent_mask_names], mask=flip_mask_agent
                )
            if self.aug_config.random_flip_sides_target_masks:
                flip_mask_target = torch.rand(B, device=agent_x.device) < 0.5
                do_flip_sides(
                    [batch[name] for name in target_mask_names], mask=flip_mask_target
                )

            # --- widening ---
            if self.aug_config.body_width_config is not None:
                self._body_width_augment_agent_target(
                    agent_x=agent_x,
                    agent_y=agent_y,
                    target_x=target_x,
                    target_y=target_y,
                )

            # --- jitter ---
            if self.aug_config.coord_jitter_std > 0.0:
                std = self.aug_config.coord_jitter_std

                def apply(arr):
                    arr += torch.randn_like(arr) * std

                apply(agent_x)
                apply(agent_y)
                apply(target_x)
                apply(target_y)

        single_cfg = self.feats_config.single_mouse_numeric
        pair_cfg = self.feats_config.pair_numeric
        idx = self.kp_index

        # ===== Build window-centered body CS from agent and project =====
        def get_body_cs(x, y):
            return BodyCS.from_coords(
                x=x,
                y=y,
                keypoint_names=self.feats_config.bodyparts,
                origin_name=single_cfg.origin,
                heading_left=single_cfg.heading_left,
                heading_right=single_cfg.heading_right,
                eps=self.eps,
            )

        body_cs = get_body_cs(agent_x, agent_y)

        agent_x_b, agent_y_b = body_cs.project(agent_x, agent_y)  # [B, T, K]
        target_x_b, target_y_b = body_cs.project(target_x, target_y)  # [B, T, K]

        # ===== Bodypart Dropout Augmentation =====
        if self.training:
            if self.aug_config.bodypart_dropout_config is not None:
                self._bodypart_dropout_mouse_single(
                    x=agent_x_b, y=agent_y_b, who="agent", batch=batch
                )
                self._bodypart_dropout_mouse_single(
                    x=target_x_b, y=target_y_b, who="target", batch=batch
                )

        # ===== Spatial Cutout Augmentation =====
        if self.training:
            if self.aug_config.spatial_cutout_config is not None:
                self._spatial_cutout(
                    agent_x=agent_x_b,
                    agent_y=agent_y_b,
                    target_x=target_x_b,
                    target_y=target_y_b,
                    batch=batch,
                )

        agent_single = self._single_mouse_feats(
            agent_x_b, agent_y_b, body_cs, is_agent=True
        )
        target_single = self._single_mouse_feats(
            target_x_b, target_y_b, body_cs, is_agent=False
        )

        pair_chunks: list[torch.Tensor] = []

        # ===== Pair features (agent–target) =====

        # distance pairs in agent body CS
        if pair_cfg.distance_pairs:
            dp_chunks = []
            for ag_name, tg_name in pair_cfg.distance_pairs:
                ia = idx[ag_name]
                it = idx[tg_name]
                ax = agent_x_b[:, :, ia]
                ay = agent_y_b[:, :, ia]
                tx = target_x_b[:, :, it]
                ty = target_y_b[:, :, it]
                dx = tx - ax
                dy = ty - ay
                dist = torch.sqrt(dx * dx + dy * dy + self.eps)
                dp_chunks.append(dist.unsqueeze(-1))
            if dp_chunks:
                pair_chunks.append(torch.cat(dp_chunks, dim=-1))  # [B, T, n_pairs]

        # nose–nose distance / delta / "close" flag (agent CS)
        if (
            pair_cfg.include_nose_nose_delta
            or pair_cfg.include_nose_nose_distance
            or pair_cfg.include_nose_nose_close_flag
        ):
            inose = idx["nose"]
            ax = agent_x_b[:, :, inose]
            ay = agent_y_b[:, :, inose]
            tx = target_x_b[:, :, inose]
            ty = target_y_b[:, :, inose]

            dx = tx - ax
            dy = ty - ay
            dist_nn = torch.sqrt(dx * dx + dy * dy + self.eps)  # [B, T]

            if pair_cfg.include_nose_nose_distance:
                pair_chunks.append(dist_nn.unsqueeze(-1))

            if pair_cfg.include_nose_nose_delta:
                delta_nn = central_diff(dist_nn)  # [B, T]
                pair_chunks.append(delta_nn.unsqueeze(-1))

            if pair_cfg.include_nose_nose_close_flag:
                thr = pair_cfg.nose_nose_close_threshold
                close_flag = (dist_nn < thr).float()  # [B, T]
                pair_chunks.append(close_flag.unsqueeze(-1))

        # facing target→agent: angle of target keypoints in agent body CS
        if pair_cfg.include_facing_target_to_agent and pair_cfg.facing_keypoints:
            for name in pair_cfg.facing_keypoints:
                it = idx[name]
                vx = target_x_b[:, :, it]  # [B, T]
                vy = target_y_b[:, :, it]
                ang, sin_a, cos_a = angle_trig(vx, vy)
                pair_chunks.append(ang.unsqueeze(-1))
                pair_chunks.append(sin_a.unsqueeze(-1))
                pair_chunks.append(cos_a.unsqueeze(-1))

        # facing agent→target: angle of agent keypoints in *target's* body CS
        if pair_cfg.include_facing_agent_to_target and pair_cfg.facing_keypoints:
            # Build a window-centered body CS for the target as well
            target_cs = BodyCS.from_coords(
                x=target_x,
                y=target_y,
                keypoint_names=self.feats_config.bodyparts,
                origin_name=single_cfg.origin,
                heading_left=single_cfg.heading_left,
                heading_right=single_cfg.heading_right,
                eps=self.eps,
            )

            # constant heading vector for target, broadcast over time
            Ht_x_2d = target_cs.H_x.view(B, 1).expand(B, T)  # [B, T]
            Ht_y_2d = target_cs.H_y.view(B, 1).expand(B, T)  # [B, T]

            for name in pair_cfg.facing_keypoints:
                ia = idx[name]
                Ax = agent_x[:, :, ia]  # [B, T]
                Ay = agent_y[:, :, ia]

                # vector from target origin (at center frame) to agent keypoint
                rx = Ax - target_cs.O_x.view(B, 1)  # broadcasts to [B, T]
                ry = Ay - target_cs.O_y.view(B, 1)

                ang, sin_a, cos_a = angle_between(
                    Ht_x_2d, Ht_y_2d, rx, ry, eps=self.eps
                )
                pair_chunks.append(ang.unsqueeze(-1))
                pair_chunks.append(sin_a.unsqueeze(-1))
                pair_chunks.append(cos_a.unsqueeze(-1))

        # body-axis alignment (nose→tail) for agent & target in arena coords
        if pair_cfg.include_body_axis_alignment:
            inose = idx["nose"]
            itail = idx["tail_base"]

            a_ax = agent_x[:, :, inose] - agent_x[:, :, itail]
            a_ay = agent_y[:, :, inose] - agent_y[:, :, itail]
            t_ax = target_x[:, :, inose] - target_x[:, :, itail]
            t_ay = target_y[:, :, inose] - target_y[:, :, itail]

            dot = a_ax * t_ax + a_ay * t_ay
            norm_a = torch.sqrt(a_ax * a_ax + a_ay * a_ay + self.eps)
            norm_t = torch.sqrt(t_ax * t_ax + t_ay * t_ay + self.eps)
            cos_align = dot / (norm_a * norm_t + self.eps)  # [B, T]
            pair_chunks.append(cos_align.unsqueeze(-1))

        # center velocity alignment (co-movement)
        if pair_cfg.include_center_velocity_alignment:
            ic = idx["body_center"]

            # velocities in agent body CS
            avx = central_diff(agent_x_b[:, :, ic : ic + 1])  # [B, T, 1]
            avy = central_diff(agent_y_b[:, :, ic : ic + 1])
            tvx = central_diff(target_x_b[:, :, ic : ic + 1])
            tvy = central_diff(target_y_b[:, :, ic : ic + 1])

            avx2 = avx[:, :, 0]
            avy2 = avy[:, :, 0]
            tvx2 = tvx[:, :, 0]
            tvy2 = tvy[:, :, 0]

            dot = avx2 * tvx2 + avy2 * tvy2
            norm_a = torch.sqrt(avx2 * avx2 + avy2 * avy2 + self.eps)
            norm_t = torch.sqrt(tvx2 * tvx2 + tvy2 * tvy2 + self.eps)
            cos_valign = dot / (norm_a * norm_t + self.eps)  # [B, T]

            pair_chunks.append(cos_valign.unsqueeze(-1))

        # pursuit-like feature: agent velocity along LOS to target (in agent CS)
        for name in pair_cfg.pursuit_keypoints:
            j = idx[name]
            ax = agent_x_b[:, :, j]  # [B, T]
            ay = agent_y_b[:, :, j]
            tx = target_x_b[:, :, j]
            ty = target_y_b[:, :, j]

            # LOS vector agent -> target in agent CS
            rx = tx - ax  # [B, T]
            ry = ty - ay
            r_norm = torch.sqrt(rx * rx + ry * ry + self.eps)
            ux = rx / r_norm  # [B, T]
            uy = ry / r_norm

            # agent velocity in body CS (central diff, 1 frame)
            ax_v = central_diff(ax.unsqueeze(-1))[:, :, 0]  # [B, T]
            ay_v = central_diff(ay.unsqueeze(-1))[:, :, 0]

            # projection onto LOS; >0 means moving toward target
            pursuit = (ax_v * ux + ay_v * uy).unsqueeze(-1)  # [B, T, 1]
            pair_chunks.append(pursuit)

        numeric_feats = torch.cat([agent_single, target_single] + pair_chunks, dim=-1)
        numeric_feats.clamp_(-10, 10)

        out = dict(batch)
        out["numeric_feats"] = numeric_feats
        return out
