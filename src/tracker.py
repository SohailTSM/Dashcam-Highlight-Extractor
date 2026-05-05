"""
tracker.py
----------
Velocity-aware multi-object tracker using a cost matrix that combines:
  - 1 - IoU (bounding-box overlap)
  - Normalised centroid distance (using velocity-extrapolated predicted position)
  - Class mismatch penalty

Only 'stable' tracks (hits >= min_hits) are used by heuristics.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import Detection, FrameDetection
from .ego_motion import EgoMotion, compensate_centroid

log = logging.getLogger(__name__)


@dataclass
class TrackSnapshot:
    """One observation of a track at a single (processed) frame."""
    frame_idx:        int
    timestamp_s:      float
    bbox:             tuple       # (x1, y1, x2, y2) raw pixels
    centroid_raw:     tuple       # (cx, cy) from detector
    centroid_comp:    tuple       # (cx, cy) after ego-motion inverse
    bbox_area_norm:   float       # raw YOLO area (noisy — do not use in H1/H8)
    bbox_area_smooth: float       # EMA-smoothed area — use this in H1 and H8
    ego: EgoMotion                # ego state at this frame
    dt_s: float = 0.0             # elapsed time since previous snapshot (seconds)


@dataclass
class TrackState:
    """Per-object temporal state."""
    track_id:           int
    class_name:         str
    class_id:           int
    hits:               int = 0
    consecutive_misses: int = 0
    history:            deque = field(default_factory=deque)   # deque[TrackSnapshot]
    first_seen:         int = 0
    last_seen:          int = 0
    is_active:          bool = True
    is_stable:          bool = False
    predicted_centroid: Optional[tuple] = None

    def last_snapshot(self) -> Optional[TrackSnapshot]:
        return self.history[-1] if self.history else None

    def update_prediction(self) -> None:
        """Extrapolate centroid using last velocity (ego-compensated)."""
        if len(self.history) < 2:
            snap = self.last_snapshot()
            self.predicted_centroid = snap.centroid_comp if snap else None
            return
        s1, s2 = self.history[-2], self.history[-1]
        vx = s2.centroid_comp[0] - s1.centroid_comp[0]
        vy = s2.centroid_comp[1] - s1.centroid_comp[1]
        self.predicted_centroid = (
            s2.centroid_comp[0] + vx,
            s2.centroid_comp[1] + vy,
        )


def _iou(b1: tuple, b2: tuple) -> float:
    """Compute IoU between two (x1,y1,x2,y2) bboxes."""
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / max(a1 + a2 - inter, 1e-6)


class Tracker:
    """
    Multi-object tracker with velocity-aware matching.

    Key design decisions:
    - Tracks exist in 'tentative' state until min_hits confirmed detections.
    - Only stable tracks (is_stable=True) feed heuristics.
    - Birth/death events for H7 are only recorded for stable tracks.
    - All centroids stored are ego-compensated.
    - Track history inheritance: when a new track becomes stable and a recently-dead
      track of the same class is nearby, the dead track's history is prepended so
      that H1/H2/H3/H8 (which need N snapshots) work through ID-switch events.
    """

    def __init__(self, cfg: dict, video_fps: float, frame_stride: int):
        self.max_age       = cfg["max_age"]
        self.min_hits      = cfg["min_hits"]
        self.cost_thresh   = cfg["cost_threshold"]
        self.track_window  = cfg["track_window"]
        self.alpha         = cfg["alpha"]
        self.beta          = cfg["beta"]
        self.gamma         = cfg["gamma"]
        self.class_penalty = cfg["class_penalty"]

        self.fps           = video_fps
        self.stride        = frame_stride
        self.dt_s          = frame_stride / max(video_fps, 1e-6)
        # EMA alpha for bbox area smoothing (lower = more smoothing, higher = faster response)
        self.area_ema_alpha: float = cfg.get("area_ema_alpha", 0.35)
        # History inheritance: frames a dead track stays in the buffer
        self.inherit_window:   int   = cfg.get("history_inherit_window", 20)
        # Max normalised centroid distance to consider a dead track a candidate
        self.inherit_dist_norm: float = cfg.get("history_inherit_distance", 0.18)

        self._tracks:  Dict[int, TrackState] = {}
        self._next_id: int = 0

        # Recently-dead tracks kept for history inheritance
        # Each entry: (death_frame_idx, TrackState)
        self._recently_dead: deque = deque(maxlen=200)

        # H7 event recording
        self.stable_births: list[tuple[int, tuple]] = []   # (frame_idx, centroid)
        self.stable_deaths: list[tuple[int, tuple]] = []

        # Frame diagonal (set on first frame)
        self._diag: float = 1.0

    @property
    def active_tracks(self) -> Dict[int, TrackState]:
        return {tid: t for tid, t in self._tracks.items() if t.is_active}

    @property
    def stable_tracks(self) -> Dict[int, TrackState]:
        return {tid: t for tid, t in self._tracks.items() if t.is_active and t.is_stable}

    def update(
        self,
        frame_det: FrameDetection,
        ego: EgoMotion,
    ) -> Dict[int, TrackState]:
        """
        Match detections to existing tracks, update histories, handle births/deaths.
        Returns the full stable_tracks dict after update.
        """
        h, w = frame_det.frame_shape
        self._diag = np.sqrt(h**2 + w**2) or 1.0

        dets = frame_det.detections
        frame_idx = frame_det.frame_idx
        ts = frame_det.timestamp_s

        # Update velocity predictions for all active tracks
        for t in self._tracks.values():
            if t.is_active:
                t.update_prediction()

        matched, unmatched_dets, unmatched_tracks = self._match(dets)

        # --- Update matched tracks ---
        for det_idx, tid in matched:
            det = dets[det_idx]
            det.track_id = tid
            track = self._tracks[tid]

            cx_comp, cy_comp = compensate_centroid(
                det.centroid[0], det.centroid[1], ego
            )

            prev_snap = track.last_snapshot()
            dt = self.dt_s if prev_snap is not None else 0.0

            # EMA-smooth the bbox area to suppress YOLO detection jitter.
            # Same object at same distance can give ±5-15% area variation frame-to-frame.
            alpha = self.area_ema_alpha
            if prev_snap is not None:
                smooth_area = alpha * det.bbox_area_norm + (1.0 - alpha) * prev_snap.bbox_area_smooth
            else:
                smooth_area = det.bbox_area_norm   # first observation: no history to smooth with

            snap = TrackSnapshot(
                frame_idx=frame_idx,
                timestamp_s=ts,
                bbox=det.bbox,
                centroid_raw=det.centroid,
                centroid_comp=(cx_comp, cy_comp),
                bbox_area_norm=det.bbox_area_norm,
                bbox_area_smooth=smooth_area,
                ego=ego,
                dt_s=dt,
            )
            track.history.append(snap)
            track.hits += 1
            track.consecutive_misses = 0
            track.last_seen = frame_idx

            was_stable = track.is_stable
            track.is_stable = track.hits >= self.min_hits
            if not was_stable and track.is_stable:
                # Attempt to inherit history from a recently-dead same-class track
                self._try_inherit_history(track, frame_idx)
                self.stable_births.append((frame_idx, (cx_comp, cy_comp)))
                log.debug(f"Track {tid} ({track.class_name}) became stable at frame {frame_idx}")

        # --- Age unmatched tracks ---
        for tid in unmatched_tracks:
            track = self._tracks[tid]
            track.consecutive_misses += 1
            if track.consecutive_misses > self.max_age:
                if track.is_stable:
                    last = track.last_snapshot()
                    c = last.centroid_comp if last else (0.0, 0.0)
                    self.stable_deaths.append((frame_idx, c))
                    # Keep in recently-dead buffer for potential history inheritance
                    self._recently_dead.append((frame_idx, track))
                track.is_active = False

        # --- Create new tracks for unmatched detections ---
        for det_idx in unmatched_dets:
            det = dets[det_idx]
            tid = self._next_id
            self._next_id += 1
            det.track_id = tid

            cx_comp, cy_comp = compensate_centroid(
                det.centroid[0], det.centroid[1], ego
            )
            snap = TrackSnapshot(
                frame_idx=frame_idx,
                timestamp_s=ts,
                bbox=det.bbox,
                centroid_raw=det.centroid,
                centroid_comp=(cx_comp, cy_comp),
                bbox_area_norm=det.bbox_area_norm,
                bbox_area_smooth=det.bbox_area_norm,  # first observation: raw = smooth
                ego=ego,
                dt_s=0.0,
            )
            history: deque = deque(maxlen=self.track_window)
            history.append(snap)
            self._tracks[tid] = TrackState(
                track_id=tid,
                class_name=det.class_name,
                class_id=det.class_id,
                hits=1,
                history=history,
                first_seen=frame_idx,
                last_seen=frame_idx,
                is_active=True,
                is_stable=False,
            )

        return self.stable_tracks

    def _try_inherit_history(self, new_track: TrackState, frame_idx: int) -> None:
        """
        When a new track first becomes stable, search the recently-dead buffer for
        a same-class track that:
          1. Died within inherit_window frames ago (recently enough to be the same object)
          2. Has its last centroid within inherit_dist_norm of the new track's first centroid

        If found, prepend that dead track's history to the new track's history deque.
        This gives H1, H2, H3, H8 the continuous trajectory data they need even when
        an ID switch occurs (e.g., during a lane cut-in).

        Only the best candidate (lowest distance) is used.
        """
        new_snap = new_track.last_snapshot()
        if new_snap is None:
            return

        new_cx, new_cy = new_snap.centroid_comp

        best_dist   = float("inf")
        best_dead   = None

        stale_cutoff = frame_idx - self.inherit_window

        for (death_frame, dead_track) in self._recently_dead:
            # Skip if too old
            if death_frame < stale_cutoff:
                continue
            # Must be same class
            if dead_track.class_name != new_track.class_name:
                continue
            # Must have history to donate
            dead_snap = dead_track.last_snapshot()
            if dead_snap is None:
                continue

            # Normalised centroid distance
            dx = dead_snap.centroid_comp[0] - new_cx
            dy = dead_snap.centroid_comp[1] - new_cy
            dist_norm = (dx*dx + dy*dy) ** 0.5 / (self._diag + 1e-9)

            if dist_norm < self.inherit_dist_norm and dist_norm < best_dist:
                best_dist = dist_norm
                best_dead = dead_track

        if best_dead is None:
            return

        # Prepend the dead track's history before the new track's current snapshots
        new_snaps   = list(new_track.history)
        dead_snaps  = list(best_dead.history)
        combined    = dead_snaps + new_snaps

        # Rebuild deque respecting the track_window limit
        new_track.history = deque(combined[-self.track_window:], maxlen=self.track_window)

        log.debug(
            f"Track {new_track.track_id} ({new_track.class_name}) inherited "
            f"{len(dead_snaps)} snapshots from dead track {best_dead.track_id} "
            f"(dist_norm={best_dist:.3f})"
        )


    def _match(
        self,
        dets: List[Detection],
    ) -> tuple[list[tuple[int,int]], list[int], list[int]]:
        """
        Match detections to active tracks using weighted cost matrix.
        Returns (matched, unmatched_det_indices, unmatched_track_ids).
        """
        active_ids = list(self.active_tracks.keys())

        if not dets or not active_ids:
            return [], list(range(len(dets))), active_ids

        cost = np.zeros((len(dets), len(active_ids)), dtype=np.float32)

        for di, det in enumerate(dets):
            for tj, tid in enumerate(active_ids):
                track = self._tracks[tid]
                last = track.last_snapshot()
                if last is None:
                    cost[di, tj] = 1.0
                    continue

                # IoU term
                iou_val = _iou(det.bbox, last.bbox)
                iou_cost = 1.0 - iou_val

                # Centroid distance term (use predicted position)
                pred = track.predicted_centroid or last.centroid_comp
                dx = det.centroid[0] - pred[0]
                dy = det.centroid[1] - pred[1]
                dist_norm = np.sqrt(dx*dx + dy*dy) / self._diag

                # Class mismatch penalty
                class_cost = 0.0 if det.class_id == track.class_id else self.class_penalty

                cost[di, tj] = (
                    self.alpha * iou_cost +
                    self.beta  * dist_norm +
                    self.gamma * class_cost
                )

        row_ind, col_ind = linear_sum_assignment(cost)

        matched, unmatched_dets, unmatched_tracks = [], [], []
        matched_det_set, matched_track_set = set(), set()

        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] <= self.cost_thresh:
                matched.append((ri, active_ids[ci]))
                matched_det_set.add(ri)
                matched_track_set.add(active_ids[ci])

        unmatched_dets   = [i for i in range(len(dets))  if i not in matched_det_set]
        unmatched_tracks = [tid for tid in active_ids if tid not in matched_track_set]

        return matched, unmatched_dets, unmatched_tracks
