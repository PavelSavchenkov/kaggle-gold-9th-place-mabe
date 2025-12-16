"""
Utilities for visualising tracked mice behaviour inside Jupyter notebooks.

The main entry point is ``play_mouse_animation`` which consumes three pandas
DataFrames describing the video metadata, tracking coordinates, and behavioural
annotations.  The function renders an inline animation using matplotlib that
overlays reconstructed mouse shapes, active behaviour labels, and arena bounds.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from IPython.display import HTML
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.text import Text


# Keywords that identify head-related keypoints.
_HEAD_KEYWORDS = ("head", "ear", "nose", "snout", "mouth", "muzzle")

# Tail bodyparts ordered from base to tip for prioritised lookups.
_DEFAULT_TAIL_PARTS = (
    "tail_base",
    "tail_midpoint",
    "tail_middle_1",
    "tail_middle_2",
    "tail_tip",
)

_COLOUR_CYCLE = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
]


@dataclass(frozen=True)
class _MouseArtists:
    head: patches.Polygon
    body: patches.Polygon
    tail: Line2D
    body_points: Line2D
    nose: Line2D


@dataclass(frozen=True)
class _FrameMouseGeometry:
    triangle: Optional[np.ndarray]
    body_polygon: Optional[np.ndarray]
    tail_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
    body_points: Optional[np.ndarray]
    nose: Optional[Tuple[float, float]]
    distances_cm: Dict[str, Optional[float]]
    keypoints: Tuple[str, ...]


def _categorise_bodypart(name: str) -> str:
    """
    Categorise a bodypart into ``head``, ``tail`` or ``body`` groups.
    """
    lowered = name.lower()
    if "tail" in lowered:
        return "tail"
    if any(keyword in lowered for keyword in _HEAD_KEYWORDS):
        return "head"
    return "body"


def _convex_hull(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Compute a 2D convex hull using Andrew's monotone chain algorithm.
    """
    if len(points) <= 3:
        return list(points)

    pts = sorted(set(points))
    if len(pts) <= 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Omit the last point of each list because it is repeated.
    return lower[:-1] + upper[:-1]


def _first_valid_point(
    points: Dict[str, Tuple[float, float]], candidates: Sequence[str]
) -> Optional[Tuple[float, float]]:
    for name in candidates:
        pt = points.get(name)
        if pt is not None:
            return pt
    return None


def _build_frame_lookup(
    tracking_df: pd.DataFrame,
) -> Tuple[Dict[int, Dict[int, Dict[str, Tuple[float, float]]]], List[int], List[int]]:
    """
    Materialise a nested lookup of tracking coordinates for fast access.
    """
    if not {"video_frame", "mouse_id", "bodypart", "x", "y"}.issubset(tracking_df.columns):
        missing = {"video_frame", "mouse_id", "bodypart", "x", "y"} - set(tracking_df.columns)
        raise ValueError(f"tracking_df missing required columns: {missing}")

    frames: Dict[int, Dict[int, Dict[str, Tuple[float, float]]]] = {}
    present_mouse_ids: set[int] = set()

    for row in tracking_df.itertuples(index=False):
        frame = int(getattr(row, "video_frame"))
        mouse_id = int(getattr(row, "mouse_id"))
        bodypart = str(getattr(row, "bodypart"))
        x = getattr(row, "x")
        y = getattr(row, "y")
        if pd.isna(x) or pd.isna(y):
            continue
        frames.setdefault(frame, {}).setdefault(mouse_id, {})[bodypart] = (float(x), float(y))
        present_mouse_ids.add(mouse_id)

    frame_numbers = sorted(frames.keys())
    mouse_ids = sorted(present_mouse_ids)
    if not frame_numbers:
        raise ValueError("No usable tracking coordinates were found in tracking_df.")

    return frames, frame_numbers, mouse_ids


def _build_action_lookup(annotations_df: Optional[pd.DataFrame]) -> Dict[int, str]:
    if annotations_df is None or annotations_df.empty:
        return {}

    lookup: Dict[int, List[str]] = defaultdict(list)
    for row in annotations_df.itertuples(index=False):
        start = getattr(row, "start_frame", None)
        stop = getattr(row, "stop_frame", None)
        if pd.isna(start) or pd.isna(stop):
            continue
        action = str(getattr(row, "action", ""))
        agent = getattr(row, "agent_id", None)
        target = getattr(row, "target_id", None)

        try:
            start_frame = int(start)
            stop_frame = int(stop)
        except (TypeError, ValueError):
            continue

        if stop_frame < start_frame:
            start_frame, stop_frame = stop_frame, start_frame

        agent_label = f"{int(agent)}" if pd.notna(agent) else "?"
        if pd.notna(target):
            label = f"{action} ({agent_label}->{int(target)})"
        else:
            label = f"{action} ({agent_label})"

        for frame in range(start_frame, stop_frame + 1):
            lookup[frame].append(label)

    formatted: Dict[int, str] = {}
    for frame, labels in lookup.items():
        if not labels:
            continue
        unique_labels = list(dict.fromkeys(labels))
        formatted[frame] = ", ".join(unique_labels)
    return formatted


def _ensure_video_metadata_row(metadata_df: pd.DataFrame, video_id: Optional[str]) -> pd.Series:
    if not isinstance(metadata_df, pd.DataFrame):
        raise TypeError("metadata_df must be a pandas DataFrame.")
    if "video_id" not in metadata_df.columns:
        if len(metadata_df) != 1:
            raise ValueError(
                "metadata_df must contain a 'video_id' column or refer to a single video."
            )
        return metadata_df.iloc[0]

    if video_id is None:
        unique_ids = metadata_df["video_id"].dropna().unique()
        if len(unique_ids) != 1:
            raise ValueError(
                "Multiple video_ids detected; please specify the desired video_id."
            )
        video_id = unique_ids[0]

    mask = metadata_df["video_id"] == video_id
    if not mask.any():
        raise ValueError(f"Video id '{video_id}' not found in metadata_df.")
    return metadata_df.loc[mask].iloc[0]


def play_mouse_animation(
    metadata_df: pd.DataFrame,
    tracking_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    video_id: Optional[str] = None,
    *,
    max_frames: Optional[int] = None,
    figsize: Tuple[float, float] = (6.5, 6.5),
    tail_bodyparts: Sequence[str] = _DEFAULT_TAIL_PARTS,
    body_alpha: float = 0.65,
    action_frames_only: bool = False,
) -> HTML:
    """
    Render tracked mice movement inline within a Jupyter notebook.

    Parameters
    ----------
    metadata_df:
        DataFrame describing videos. Must contain at least ``video_id``,
        ``video_width_pix``, ``video_height_pix`` and ``frames_per_second``.
    tracking_df:
        Long-form DataFrame containing tracked bodypart coordinates with columns
        ``video_frame``, ``mouse_id``, ``bodypart``, ``x`` and ``y``.
    annotations_df:
        DataFrame describing behavioural annotations per frame. Expected
        columns: ``agent_id``, ``target_id``, ``action``, ``start_frame``,
        ``stop_frame``.
    video_id:
        Identifier of the video to render. If omitted the function attempts to
        infer a single video from ``metadata_df`` (and will error if multiple
        entries are present).
    max_frames:
        Optional cap on the number of frames rendered from the beginning of the
        sequence.
    figsize:
        Matplotlib figure size in inches.
    tail_bodyparts:
        Ordered list of tail keypoints (from base to tip) used when connecting
        the head anchor to the tail.
    body_alpha:
        Fill transparency applied to the reconstructed body hull (higher values
        make the hull more opaque).
    action_frames_only:
        When ``True`` only frames that contain at least one annotated action are
        rendered.
    """

    metadata_row = _ensure_video_metadata_row(metadata_df, video_id)

    for required_col in ("video_width_pix", "video_height_pix"):
        if required_col not in metadata_row:
            raise ValueError(f"metadata_df missing required column '{required_col}'.")

    width = float(metadata_row["video_width_pix"])
    height = float(metadata_row["video_height_pix"])
    fps = float(metadata_row.get("frames_per_second", np.nan))
    pixels_per_cm_raw = metadata_row.get("pix_per_cm_approx", np.nan)
    if np.isfinite(pixels_per_cm_raw) and pixels_per_cm_raw > 0:
        pixels_per_cm: Optional[float] = float(pixels_per_cm_raw)
    else:
        pixels_per_cm = None

    frames_lookup, frame_numbers, mouse_ids = _build_frame_lookup(tracking_df)
    action_lookup = _build_action_lookup(annotations_df)

    if action_frames_only:
        frame_numbers = [frame for frame in frame_numbers if frame in action_lookup]

    if max_frames is not None and max_frames >= 0:
        frame_numbers = frame_numbers[:max_frames]

    if not frame_numbers:
        raise ValueError("No frames available after applying filters.")

    first_frame = frame_numbers[0]

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.22, right=0.78)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    ax.set_title(str(metadata_row.get("video_id", "Mouse Behaviour")), loc="left", pad=12)

    lab_value = metadata_row.get("lab_id", np.nan)
    tracking_value = metadata_row.get("tracking_method", np.nan)
    lab_label = "Unknown" if pd.isna(lab_value) else str(lab_value)
    tracking_label = "Unknown" if pd.isna(tracking_value) else str(tracking_value)
    metadata_texts = [
        fig.text(
            0.02,
            0.97,
            f"Lab: {lab_label}",
            ha="left",
            va="top",
            fontsize=10,
        ),
        fig.text(
            0.02,
            0.90,
            f"Tracking: {tracking_label}",
            ha="left",
            va="top",
            fontsize=10,
        ),
    ]

    arena = patches.Rectangle(
        (0, 0), width, height, linewidth=1.5, edgecolor="dimgray", facecolor="none"
    )
    ax.add_patch(arena)

    frame_text = ax.text(
        0.98,
        1.06,
        "",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
    )
    playback_fps = 30.0
    body_fallback = max(width, height) * 0.01
    if np.isfinite(fps):
        fps_label = f"Real FPS: {fps:.2f} | Playback: {playback_fps:.2f}"
    else:
        fps_label = f"Real FPS: N/A | Playback: {playback_fps:.2f}"
    fps_text = ax.text(
        0.02, 0.02, fps_label, ha="left", va="bottom", transform=ax.transAxes, fontsize=10
    )
    action_text = fig.text(
        0.46,
        0.04,
        "",
        ha="center",
        va="center",
        fontsize=11,
    )

    # Assign distinct colours per mouse via a high-contrast palette.
    mouse_artists: Dict[int, _MouseArtists] = {}
    for idx, mouse_id in enumerate(mouse_ids):
        colour = _COLOUR_CYCLE[idx % len(_COLOUR_CYCLE)]
        head_poly = patches.Polygon(
            np.zeros((3, 2)),
            closed=True,
            fill=False,
            linewidth=2.0,
            edgecolor=colour,
            visible=False,
        )
        body_poly = patches.Polygon(
            np.zeros((3, 2)),
            closed=True,
            fill=True,
            facecolor=colour,
            edgecolor=colour,
            alpha=body_alpha,
            visible=False,
        )
        tail_line = Line2D([], [], color=colour, linewidth=1.6, alpha=0.9, visible=False)
        body_points_line = Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=5,
            markerfacecolor=colour,
            markeredgecolor=colour,
            markeredgewidth=1.2,
            alpha=1.0,
            visible=False,
        )
        nose_point = Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=5,
            markerfacecolor=colour,
            markeredgecolor=colour,
            markeredgewidth=1.2,
            alpha=1.0,
            visible=False,
        )

        ax.add_patch(body_poly)
        ax.add_patch(head_poly)
        ax.add_line(tail_line)
        ax.add_line(body_points_line)
        ax.add_line(nose_point)

        mouse_artists[mouse_id] = _MouseArtists(
            head=head_poly,
            body=body_poly,
            tail=tail_line,
            body_points=body_points_line,
            nose=nose_point,
        )

    legend_texts: List[Text] = []
    keypoint_texts: Dict[int, Text] = {}
    if mouse_ids:
        legend_x = 0.82
        legend_start_y = 0.97
        legend_step = 0.05
        for idx, mouse_id in enumerate(mouse_ids):
            colour = _COLOUR_CYCLE[idx % len(_COLOUR_CYCLE)]
            legend_texts.append(
                fig.text(
                    legend_x,
                    legend_start_y - idx * legend_step,
                    f"\u2588 Mouse {mouse_id}",
                    ha="left",
                    va="top",
                    fontsize=10,
                    color=colour,
                    fontweight="bold",
                )
            )

        keypoint_base_y = legend_start_y - len(mouse_ids) * legend_step - 0.05
        min_y = 0.08
        available_space = max(keypoint_base_y - min_y, 0.25)
        keypoint_step = available_space / max(len(mouse_ids), 1)
        keypoint_step = max(0.18, min(keypoint_step, 0.3))
        for idx, mouse_id in enumerate(mouse_ids):
            colour = _COLOUR_CYCLE[idx % len(_COLOUR_CYCLE)]
            text_artist = fig.text(
                legend_x,
                keypoint_base_y - idx * keypoint_step,
                f"Mouse {mouse_id}:\n- none",
                ha="left",
                va="top",
                fontsize=9,
                color=colour,
            )
            keypoint_texts[mouse_id] = text_artist

    def _extract_groups(points: Dict[str, Tuple[float, float]]) -> Tuple[
        Dict[str, Tuple[float, float]],
        Dict[str, Tuple[float, float]],
        Dict[str, Tuple[float, float]],
    ]:
        head_pts: Dict[str, Tuple[float, float]] = {}
        body_pts: Dict[str, Tuple[float, float]] = {}
        tail_pts: Dict[str, Tuple[float, float]] = {}
        for name, coords in points.items():
            category = _categorise_bodypart(name)
            if category == "head":
                head_pts[name] = coords
            elif category == "tail":
                tail_pts[name] = coords
            else:
                body_pts[name] = coords
        return head_pts, body_pts, tail_pts

    def _head_triangle(head_points: Dict[str, Tuple[float, float]]) -> Optional[np.ndarray]:
        if not head_points:
            return None

        nose = head_points.get("nose")
        ear_left = head_points.get("ear_left")
        ear_right = head_points.get("ear_right")

        available = [pt for pt in (nose, ear_left, ear_right) if pt is not None]
        if len(available) == 3:
            return np.array([nose, ear_left, ear_right])

        if len(head_points) >= 3:
            # Fall back to the first three available head points.
            pts = list(head_points.values())[:3]
            return np.array(pts)

        if len(head_points) == 2 and nose is not None:
            # Duplicate nose to form a degenerate triangle.
            return np.array([nose, *head_points.values()])

        return None

    def _head_anchor(head_points: Dict[str, Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        priority = ("nose", "head", "head_center")
        anchor = _first_valid_point(head_points, priority)
        if anchor is not None:
            return anchor
        if head_points:
            pts = np.array(list(head_points.values()))
            return float(np.nanmean(pts[:, 0])), float(np.nanmean(pts[:, 1]))
        return None

    def _body_hull(body_points: Dict[str, Tuple[float, float]]) -> Optional[np.ndarray]:
        pts = [np.array(p, dtype=float) for p in body_points.values()]
        if len(pts) >= 3:
            hull = _convex_hull([tuple(p) for p in pts])
            return np.array(hull)
        if len(pts) == 2:
            p1, p2 = pts
            direction = p2 - p1
            if np.allclose(direction, 0.0):
                direction = np.array([1.0, 0.0])
            perp = np.array([-direction[1], direction[0]])
            norm = np.linalg.norm(perp)
            if norm == 0.0:
                perp = np.array([0.0, 1.0])
                norm = 1.0
            perp /= norm
            offset = body_fallback
            return np.array(
                [
                    p1 + perp * offset,
                    p2 + perp * offset,
                    p2 - perp * offset,
                    p1 - perp * offset,
                ]
            )
        if len(pts) == 1:
            center = pts[0]
            offset = body_fallback
            return np.array(
                [
                    center + np.array([offset, 0.0]),
                    center + np.array([0.0, offset]),
                    center - np.array([offset, 0.0]),
                    center - np.array([0.0, offset]),
                ]
            )
        return None

    def _tail_anchor(tail_points: Dict[str, Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        return _first_valid_point(tail_points, tail_bodyparts)

    def _precompute_geometry() -> Dict[int, Dict[int, _FrameMouseGeometry]]:
        geometry: Dict[int, Dict[int, _FrameMouseGeometry]] = {}
        prev_frame_points: Dict[int, Dict[str, Tuple[float, float]]] = {}
        for frame in frame_numbers:
            per_frame: Dict[int, _FrameMouseGeometry] = {}
            frame_points = frames_lookup.get(frame, {})
            for mouse_id, points in frame_points.items():
                head_pts, body_pts, tail_pts = _extract_groups(points)
                triangle = _head_triangle(head_pts)
                body_polygon = _body_hull(body_pts)
                head_anchor = _head_anchor(head_pts)
                tail_anchor = _tail_anchor(tail_pts)
                body_points_coords = (
                    np.array(list(body_pts.values()), dtype=float) if body_pts else None
                )
                nose_coord = head_pts.get("nose")
                tail_line = None
                if head_anchor is not None and tail_anchor is not None:
                    tail_line = (head_anchor, tail_anchor)
                prev_mouse_points = prev_frame_points.get(mouse_id, {})
                distances_cm: Dict[str, Optional[float]] = {}
                for name, coord in points.items():
                    prev_coord = prev_mouse_points.get(name)
                    if prev_coord is not None:
                        coord_arr = np.array(coord, dtype=float)
                        prev_arr = np.array(prev_coord, dtype=float)
                        dist_px = float(np.linalg.norm(coord_arr - prev_arr))
                        if pixels_per_cm is not None:
                            distances_cm[name] = dist_px / pixels_per_cm
                        else:
                            distances_cm[name] = None
                    else:
                        distances_cm[name] = None
                per_frame[mouse_id] = _FrameMouseGeometry(
                    triangle=triangle,
                    body_polygon=body_polygon,
                    tail_line=tail_line,
                    body_points=body_points_coords,
                    nose=tuple(nose_coord) if nose_coord is not None else None,
                    distances_cm=distances_cm,
                    keypoints=tuple(sorted(points.keys())),
                )
            geometry[frame] = per_frame
            prev_frame_points = {mid: dict(pts) for mid, pts in frame_points.items()}
        return geometry

    geometry_by_frame = _precompute_geometry()

    def init():
        frame_text.set_text(f"Frame {first_frame}")
        action_text.set_text(f"Actions: {action_lookup.get(first_frame, 'None')}")
        for mouse_id, text_artist in keypoint_texts.items():
            text_artist.set_text(f"Mouse {mouse_id}:\n- none")
        for artists in mouse_artists.values():
            artists.head.set_visible(False)
            artists.body.set_visible(False)
            artists.tail.set_visible(False)
            artists.body_points.set_visible(False)
            artists.nose.set_visible(False)
        return (
            [arena, frame_text, fps_text, action_text]
            + metadata_texts
            + legend_texts
            + list(keypoint_texts.values())
            + [
                artist
                for group in mouse_artists.values()
                for artist in (group.head, group.body, group.tail, group.body_points, group.nose)
            ]
        )

    def update(frame_idx: int):
        frame = frame_numbers[frame_idx]
        frame_text.set_text(f"Frame {frame}")
        action_text.set_text(f"Actions: {action_lookup.get(frame, 'None')}")

        geometry_map = geometry_by_frame.get(frame, {})
        for mouse_id, artists in mouse_artists.items():
            geometry = geometry_map.get(mouse_id)

            if geometry is None or geometry.triangle is None:
                artists.head.set_visible(False)
            else:
                artists.head.set_xy(geometry.triangle)
                artists.head.set_visible(True)

            if geometry is None or geometry.body_polygon is None:
                artists.body.set_visible(False)
            else:
                artists.body.set_xy(geometry.body_polygon)
                artists.body.set_visible(True)

            if geometry is None or geometry.tail_line is None:
                artists.tail.set_visible(False)
            else:
                head_anchor, tail_anchor = geometry.tail_line
                artists.tail.set_data(
                    (head_anchor[0], tail_anchor[0]),
                    (head_anchor[1], tail_anchor[1]),
                )
                artists.tail.set_visible(True)

            if geometry is None or geometry.body_points is None:
                artists.body_points.set_visible(False)
            else:
                artists.body_points.set_data(
                    geometry.body_points[:, 0], geometry.body_points[:, 1]
                )
                artists.body_points.set_visible(True)

            if geometry is None or geometry.nose is None:
                artists.nose.set_visible(False)
            else:
                artists.nose.set_data([geometry.nose[0]], [geometry.nose[1]])
                artists.nose.set_visible(True)

            text_artist = keypoint_texts.get(mouse_id)
            if text_artist is not None:
                if geometry is None or not geometry.keypoints:
                    text_artist.set_text(f"Mouse {mouse_id}:\n- none")
                else:
                    formatted_lines: List[str] = []
                    for name in geometry.keypoints:
                        distance_cm = geometry.distances_cm.get(name)
                        if distance_cm is not None:
                            formatted_lines.append(f"- {name} ({distance_cm:.2f} cm)")
                        else:
                            formatted_lines.append(f"- {name}")
                    lines = "\n".join(formatted_lines)
                    text_artist.set_text(f"Mouse {mouse_id}:\n{lines}")

        return (
            [arena, frame_text, fps_text, action_text]
            + metadata_texts
            + legend_texts
            + list(keypoint_texts.values())
            + [
                artist
                for group in mouse_artists.values()
                for artist in (group.head, group.body, group.tail, group.body_points, group.nose)
            ]
        )

    interval = 1000.0 / playback_fps
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frame_numbers),
        interval=interval,
        blit=False,
        repeat=False,
    )

    plt.close(fig)
    return HTML(anim.to_jshtml())


__all__ = ["play_mouse_animation"]
