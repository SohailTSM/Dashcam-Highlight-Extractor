"""
heuristics.py
-------------
All 8 heuristic scoring functions.
Each function:
  - receives the stable_tracks dict, the current EgoMotion, frame shape, and cfg
  - returns a raw score in [0, 1]
  - operates exclusively on ego-compensated centroids (centroid_comp)
  - is stride-invariant: velocities expressed in px/s using snap.dt_s

Heuristics:
  H1 — Rapid Approach (relative, ego-gated)
  H2 — Lateral Cut-In (trajectory angle change)
  H3 — Sudden Braking (deceleration via linear regression)
  H4 — Near-Miss Proximity (class-normalised size × vertical position)
  H5 — Scene Complexity + Motion Entropy
  H6 — Pedestrian / Cyclist in Road Zone
  H7 — Stable Track Birth/Death Rate + Spatial Spread
  H8 — Time-To-Collision (TTC via bbox area derivative)
"""

from __future__ import annotations
from collections import deque
from typing import Dict, List
import math

import numpy as np
from scipy.stats import linregress

from .yolo_tracker import TrackedObject, TrackSnapshot
from .ego_motion import EgoMotion

EPS = 1e-9


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _centroid_velocities_px_s(snaps: list[TrackSnapshot]) -> list[float]:
    """
    Compute per-step ego-compensated displacement magnitudes in px/s.
    Divides each step displacement by snap.dt_s (stride-invariant).
    Skips steps where dt_s == 0.
    """
    speeds = []
    for i in range(1, len(snaps)):
        dt = snaps[i].dt_s
        if dt < EPS:
            continue
        dx = snaps[i].centroid_comp[0] - snaps[i-1].centroid_comp[0]
        dy = snaps[i].centroid_comp[1] - snaps[i-1].centroid_comp[1]
        speeds.append(math.sqrt(dx*dx + dy*dy) / dt)
    return speeds


def _displacement_vector_px_s(snaps: list[TrackSnapshot]) -> tuple[float, float]:
    """
    Net displacement vector over a list of snapshots, in px/s.
    Uses total elapsed time = sum of dt_s.
    """
    total_dt = sum(s.dt_s for s in snaps[1:]) + EPS
    dx = snaps[-1].centroid_comp[0] - snaps[0].centroid_comp[0]
    dy = snaps[-1].centroid_comp[1] - snaps[0].centroid_comp[1]
    return dx / total_dt, dy / total_dt


def _stable_history(track: TrackState, min_len: int) -> list[TrackSnapshot] | None:
    """Return history list if long enough, else None."""
    h = list(track.history)
    return h if len(h) >= min_len else None


# ---------------------------------------------------------------------------
# H1 — Rapid Approach (ego-corrected)
# ---------------------------------------------------------------------------

def h1_rapid_approach(
    stable_tracks: Dict[int, TrackedObject],
    ego: EgoMotion,
    frame_shape: tuple,
    cfg: dict,
) -> float:
    """
    Measures how much faster an object's bbox area grows compared to
    what ego-motion alone would predict.  Highway acceleration → score ≈ 0.
    """
    area_rel_max = cfg["area_rel_max"]
    min_hist = cfg["h1_min_history"]
    target = {"person", "car", "bus", "truck", "motorcycle", "bicycle"}

    best = 0.0
    for track in stable_tracks.values():
        if track.class_name not in target:
            continue
        h = _stable_history(track, min_hist)
        if h is None:
            continue

        mid = len(h) // 2
        # Blend raw and EMA area based on track age.
        # Short tracks: EMA hasn't converged yet (lags real growth) → lean on raw area.
        # Long tracks:  EMA is fully converged → use smoothed area for jitter immunity.
        ema_blend = min(len(h) / 10.0, 1.0)  # 0 at 0 frames → 1.0 at 10+ frames
        def _area(s):
            return ema_blend * s.bbox_area_smooth + (1.0 - ema_blend) * s.bbox_area_norm

        area_early = np.mean([_area(s) for s in h[:mid]])
        area_late  = np.mean([_area(s) for s in h[mid:]])
        total_dt   = sum(s.dt_s for s in h[mid:]) + EPS

        raw_rate = (area_late - area_early) / total_dt   # normalised area / s

        # Ego-implied area growth rate: area ∝ scale², so rate ≈ (scale²-1)/dt
        if ego.valid:
            ego_area_rate = (ego.scale ** 2 - 1.0) / (total_dt + EPS)
        else:
            ego_area_rate = 0.0

        relative_rate = raw_rate - ego_area_rate
        score = float(np.clip(relative_rate / (area_rel_max + EPS), 0.0, 1.0))
        best = max(best, score)

    return best


# ---------------------------------------------------------------------------
# H2 — Lateral Cut-In (trajectory angle change)
# ---------------------------------------------------------------------------

def h2_lateral_cutin(
    stable_tracks: Dict[int, TrackedObject],
    ego: EgoMotion,
    frame_shape: tuple,
    cfg: dict,
) -> float:
    """
    Detects a sharp bend in a vehicle's ego-compensated trajectory:
    moving laterally then heading toward the camera.
    """
    v_min = cfg["v_min_px_s"]
    min_hist = cfg["h2_min_history"]
    vehicle = {"car", "bus", "truck", "motorcycle"}

    best = 0.0
    for track in stable_tracks.values():
        if track.class_name not in vehicle:
            continue
        h = _stable_history(track, min_hist)
        if h is None:
            continue

        mid = len(h) // 2
        vx_e, vy_e = _displacement_vector_px_s(h[:mid+1])
        vx_l, vy_l = _displacement_vector_px_s(h[mid:])

        mag_e = math.sqrt(vx_e**2 + vy_e**2)
        mag_l = math.sqrt(vx_l**2 + vy_l**2)

        if mag_e < v_min or mag_l < v_min:
            continue

        cos_angle = np.clip(
            (vx_e * vx_l + vy_e * vy_l) / (mag_e * mag_l),
            -1.0, 1.0
        )
        theta = math.acos(cos_angle)   # radians, [0, π]
        score = theta / math.pi
        best = max(best, score)

    return float(best)


# ---------------------------------------------------------------------------
# H3 — Sudden Braking / Object Deceleration
# ---------------------------------------------------------------------------

def h3_sudden_braking(
    stable_tracks: Dict[int, TrackedObject],
    ego: EgoMotion,
    frame_shape: tuple,
    cfg: dict,
) -> float:
    """
    Fits linear regression to ego-compensated speed-over-time.
    Negative slope = deceleration.  Speeds pre-smoothed with 3-point average.
    """
    beta_max = cfg["decel_beta_max"]   # px/s²
    min_hist = cfg["h3_min_history"]
    vehicle = {"car", "bus", "truck", "motorcycle"}

    best = 0.0
    for track in stable_tracks.values():
        if track.class_name not in vehicle:
            continue
        h = _stable_history(track, min_hist)
        if h is None:
            continue

        speeds = _centroid_velocities_px_s(h)
        if len(speeds) < 4:
            continue

        # 3-point moving average smoothing
        kernel = np.ones(3) / 3.0
        speeds_smooth = np.convolve(speeds, kernel, mode="valid")
        t = np.arange(len(speeds_smooth), dtype=float)

        if len(t) < 2:
            continue

        try:
            slope, *_ = linregress(t, speeds_smooth)
        except Exception:
            continue

        # Negative slope means deceleration
        score = float(np.clip(-slope / (beta_max + EPS), 0.0, 1.0))
        best = max(best, score)

    return best


# ---------------------------------------------------------------------------
# H4 — Near-Miss Proximity (class-normalised)
# ---------------------------------------------------------------------------

def h4_nearmiss_proximity(
    stable_tracks: Dict[int, TrackedObject],
    ego: EgoMotion,
    frame_shape: tuple,
    cfg: dict,
) -> float:
    """
    Objects that are unusually large for their class AND positioned low in frame
    (i.e., close in dashcam perspective).
    """
    class_ref = cfg["class_ref_area"]    # dict: class_name → reference area
    top_k     = cfg["proximity_top_k"]
    threshold = cfg["proximity_sum_threshold"]
    H, W = frame_shape

    scores = []
    for track in stable_tracks.values():
        snap = track.last_snapshot()
        if snap is None:
            continue
        ref = class_ref.get(track.class_name, 0.05)
        # Use EMA-smoothed area to suppress YOLO bbox jitter
        relative_size = np.clip(snap.bbox_area_smooth / (ref + EPS), 0.0, 3.0) / 3.0
        vertical_prox = snap.centroid_comp[1] / max(H, 1)   # 0=top, 1=bottom
        scores.append(relative_size * vertical_prox)

    if not scores:
        return 0.0

    scores.sort(reverse=True)
    top_sum = sum(scores[:top_k])
    return float(np.clip(top_sum / (threshold + EPS), 0.0, 1.0))


# ---------------------------------------------------------------------------
# H5 — Scene Complexity + Motion Entropy
# ---------------------------------------------------------------------------

class _ComplexityState:
    """Rolling statistics for H5 background baseline."""
    def __init__(self, window: int):
        self._window = window
        self._counts: deque = deque(maxlen=window)

    def push(self, count: int) -> None:
        self._counts.append(count)

    def zscore(self, count: int) -> float:
        if len(self._counts) < 5:
            return 0.0
        mu = np.mean(self._counts)
        sigma = np.std(self._counts) + EPS
        return (count - mu) / sigma


# Module-level singleton (reset per pipeline run via reset_h5_state)
_complexity_state: _ComplexityState | None = None


def init_h5_state(window: int) -> None:
    global _complexity_state
    _complexity_state = _ComplexityState(window)


def h5_complexity_entropy(
    stable_tracks: Dict[int, TrackedObject],
    ego: EgoMotion,
    frame_shape: tuple,
    cfg: dict,
) -> float:
    """
    Complexity (Z-score of track count) × Motion Entropy (Shannon entropy of
    ego-compensated velocity directions).  Static crowded scenes score low.
    """
    global _complexity_state
    z_max      = cfg["complexity_z_max"]
    n_bins     = cfg["motion_entropy_bins"]
    bg_window  = cfg["complexity_bg_window"]

    if _complexity_state is None:
        init_h5_state(bg_window)

    count = len(stable_tracks)
    _complexity_state.push(count)

    z = _complexity_state.zscore(count)
    C = float(np.clip(z / z_max, 0.0, 1.0))

    # Motion entropy
    if count < 3:
        M = 0.0
    else:
        angles = []
        for track in stable_tracks.values():
            h = list(track.history)
            if len(h) < 2:
                continue
            snaps = h[-min(5, len(h)):]
            vx, vy = _displacement_vector_px_s(snaps)
            if math.sqrt(vx**2 + vy**2) < 1.0:
                continue
            angles.append(math.atan2(vy, vx))

        if len(angles) < 3:
            M = 0.0
        else:
            bins = np.linspace(-math.pi, math.pi, n_bins + 1)
            counts_b, _ = np.histogram(angles, bins=bins)
            p = counts_b / (counts_b.sum() + EPS)
            p = p[p > 0]
            entropy = -np.sum(p * np.log(p + EPS))
            M = float(entropy / math.log(n_bins + EPS))

    return float(C * (0.5 + 0.5 * M))


# ---------------------------------------------------------------------------
# H6 — Pedestrian / Cyclist in Road Zone
# ---------------------------------------------------------------------------

def h6_pedestrian_road(
    stable_tracks: Dict[int, TrackedObject],
    ego: EgoMotion,
    frame_shape: tuple,
    cfg: dict,
) -> float:
    """
    Vulnerable road users within the road zone of the frame,
    weighted by ego-compensated motion magnitude.
    """
    y_min_f, y_max_f, x_min_f, x_max_f = cfg["road_zone"]
    disp_max = cfg["ped_disp_max_px_s"]
    static_score = cfg["ped_static_score"]
    vru = {"person", "bicycle"}
    H, W = frame_shape

    y_min = y_min_f * H;  y_max = y_max_f * H
    x_min = x_min_f * W;  x_max = x_max_f * W

    total = 0.0
    for track in stable_tracks.values():
        if track.class_name not in vru:
            continue
        snap = track.last_snapshot()
        if snap is None:
            continue
        cx, cy = snap.centroid_comp
        in_zone = (y_min <= cy <= y_max) and (x_min <= cx <= x_max)
        if not in_zone:
            continue

        h = list(track.history)
        speeds = _centroid_velocities_px_s(h[-10:]) if len(h) >= 2 else []
        mean_speed = np.mean(speeds) if speeds else 0.0

        if mean_speed < 3.0:   # effectively static
            total += static_score
        else:
            motion_score = float(np.clip(mean_speed / (disp_max + EPS), 0.0, 1.0))
            total += 0.4 + 0.6 * motion_score

    return float(np.clip(total, 0.0, 1.0))


# ---------------------------------------------------------------------------
# H7 — Stable Birth/Death Rate + Spatial Spread
# ---------------------------------------------------------------------------

def h7_birth_death_rate(
    frame_idx: int,
    births: list[tuple[int, tuple]],
    deaths: list[tuple[int, tuple]],
    frame_shape: tuple,
    cfg: dict,
    bg_event_history: deque,
) -> float:
    """
    High rate of stable-track births+deaths, weighted by spatial spread.
    Events spread across the full frame score higher than localised clusters.
    """
    window    = cfg["birth_death_window"]
    ev_range  = cfg["birth_death_range"]
    grid_n    = cfg["spatial_grid"]
    H, W = frame_shape

    # Events in current window
    threshold_frame = frame_idx - window
    recent_events = [
        c for (fi, c) in births + deaths if fi >= threshold_frame
    ]
    event_rate = len(recent_events) / max(window, 1)

    # Background rate
    bg_event_history.append(event_rate)
    mu_e = float(np.mean(bg_event_history)) if bg_event_history else 0.0
    excess = max(event_rate - mu_e, 0.0)
    base_score = float(np.clip(excess / (ev_range + EPS), 0.0, 1.0))

    if not recent_events or base_score < EPS:
        return 0.0

    # Spatial entropy
    cell_counts = np.zeros((grid_n, grid_n), dtype=float)
    for (cx, cy) in recent_events:
        col = int(np.clip(cx / (W / grid_n), 0, grid_n - 1))
        row = int(np.clip(cy / (H / grid_n), 0, grid_n - 1))
        cell_counts[row, col] += 1

    flat = cell_counts.flatten()
    total = flat.sum() + EPS
    p = flat / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p + EPS))
    max_entropy = math.log(grid_n ** 2 + EPS)
    spread = float(entropy / max_entropy)

    return float(base_score * (0.4 + 0.6 * spread))


# ---------------------------------------------------------------------------
# H8 — Time-To-Collision (TTC)
# ---------------------------------------------------------------------------

def h8_time_to_collision(
    stable_tracks: Dict[int, TrackedObject],
    ego: EgoMotion,
    frame_shape: tuple,
    cfg: dict,
) -> float:
    """
    Estimates TTC using bbox area growth rate:
      TTC ≈ area_now / (2 * dA/dt_relative)
    Ego-implied area growth (from scale) is subtracted.
    Low TTC → high score.
    """
    ttc_window  = cfg["ttc_window"]
    ttc_max     = cfg["ttc_max_s"]
    min_rate    = cfg["ttc_min_area_rate"]
    target = {"person", "car", "bus", "truck", "motorcycle", "bicycle"}

    best = 0.0
    for track in stable_tracks.values():
        if track.class_name not in target:
            continue
        h = list(track.history)
        if len(h) < ttc_window:
            continue

        snaps = h[-ttc_window:]
        # Blend raw/EMA area based on track age (same as H1)
        ema_blend = min(len(h) / 10.0, 1.0)
        areas = np.array(
            [ema_blend * s.bbox_area_smooth + (1.0 - ema_blend) * s.bbox_area_norm
             for s in snaps]
        )

        # Time axis in seconds (cumulative dt_s)
        times = np.cumsum([0.0] + [s.dt_s for s in snaps[1:]])
        if times[-1] < EPS:
            continue

        # Linear regression on area vs time
        try:
            slope, intercept, *_ = linregress(times, areas)
        except Exception:
            continue

        dA_dt = float(slope)   # area/s

        # Subtract ego-implied growth: area ∝ scale², rate ≈ scale²-1 per dt
        if ego.valid:
            total_t = times[-1] + EPS
            ego_area_rate = (ego.scale ** 2 - 1.0) / total_t
        else:
            ego_area_rate = 0.0

        dA_dt_rel = dA_dt - ego_area_rate

        if dA_dt_rel < min_rate:
            continue   # not approaching

        area_now = areas[-1]
        ttc_s = area_now / (2.0 * dA_dt_rel + EPS)
        score = float(np.clip(1.0 - ttc_s / ttc_max, 0.0, 1.0))
        best = max(best, score)

    return best


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

HEURISTIC_NAMES = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"]

# Human-readable labels shown to end users in CLI output, JSON report, and Gradio app.
# Internal heuristic codes are kept for debugging; this map drives all user-facing text.
HEURISTIC_LABELS: dict[str, str] = {
    "H1": "Sudden Approach",
    "H2": "Lane Cut-In",
    "H3": "Sudden Braking",
    "H4": "Close Proximity",
    "H5": "Chaotic Traffic",
    "H6": "Pedestrian / Cyclist Hazard",
    "H7": "Rapid Scene Change",
    "H8": "Collision Risk",
}


def score_frame(
    stable_tracks: Dict[int, TrackedObject],
    ego: EgoMotion,
    frame_shape: tuple,
    cfg: dict,
    frame_idx: int,
    births: list,
    deaths: list,
    bg_event_history: deque,
) -> dict[str, float]:
    """
    Run all 8 heuristics and return a dict of raw scores keyed by heuristic name.
    """
    return {
        "H1": h1_rapid_approach(stable_tracks, ego, frame_shape, cfg),
        "H2": h2_lateral_cutin(stable_tracks, ego, frame_shape, cfg),
        "H3": h3_sudden_braking(stable_tracks, ego, frame_shape, cfg),
        "H4": h4_nearmiss_proximity(stable_tracks, ego, frame_shape, cfg),
        "H5": h5_complexity_entropy(stable_tracks, ego, frame_shape, cfg),
        "H6": h6_pedestrian_road(stable_tracks, ego, frame_shape, cfg),
        "H7": h7_birth_death_rate(frame_idx, births, deaths, frame_shape, cfg, bg_event_history),
        "H8": h8_time_to_collision(stable_tracks, ego, frame_shape, cfg),
    }
