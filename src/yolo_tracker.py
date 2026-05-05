"""
yolo_tracker.py
---------------
Replaces both detector.py and tracker.py.

Uses YOLO's built-in model.track() with ByteTrack for persistent, reliable track IDs.
ByteTrack handles occlusion, lane changes, and re-identification far better than a
custom IoU-based matcher.

Public API consumed by pipeline.py:

    tracker = YOLOTracker(cfg["detector"], fps, frame_stride)
    for frame_idx, ts, frame, ego, active in tracker.run(video_path, ego_estimator):
        ...

`active` is Dict[int, TrackedObject] — track_id → object with snapshot history.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple
import logging
import math

import cv2
import numpy as np
from ultralytics import YOLO

from .ego_motion import EgoMotion, compensate_centroid

log = logging.getLogger(__name__)
EPS = 1e-9

COCO_NAMES: Dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car",
    3: "motorcycle", 5: "bus", 7: "truck",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrackSnapshot:
    """Ego-compensated observation of one tracked object at one processed frame."""
    frame_idx:        int
    timestamp_s:      float
    bbox:             tuple        # (x1, y1, x2, y2) raw pixels
    centroid_raw:     tuple        # (cx, cy) from YOLO
    centroid_comp:    tuple        # (cx, cy) ego-compensated
    bbox_area_norm:   float        # raw area / frame_area  (noisy — keep for reference)
    bbox_area_smooth: float        # EMA-smoothed area      (use in H1 / H4 / H8)
    confidence:       float        # YOLO detection confidence
    ego:              EgoMotion
    dt_s:             float = 0.0  # elapsed seconds since previous snapshot


@dataclass
class TrackedObject:
    """
    One persistently-tracked object. ByteTrack guarantees track_id stays consistent
    across occlusions and moderate appearance changes, eliminating the class-label
    confusion seen in the old IoU tracker.
    """
    track_id:   int
    class_name: str
    class_id:   int
    history:    deque                # deque[TrackSnapshot]
    first_seen: int = 0
    last_seen:  int = 0

    def last_snapshot(self) -> Optional[TrackSnapshot]:
        return self.history[-1] if self.history else None


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class YOLOTracker:
    """
    Thin wrapper around model.track() that:
      - Runs ByteTrack internally (no custom Hungarian matching)
      - Accumulates per-track snapshot history with EMA area smoothing
      - Tracks stable births/deaths for H7
      - Calls the ego-motion estimator per frame
    """

    # A track must survive this many frames before it counts as "stable" for H7
    BIRTH_MIN_FRAMES: int = 4

    def __init__(self, cfg: dict, video_fps: float, frame_stride: int):
        model_path = cfg.get("model_path", "weights/yolov8n.pt")
        self.model      = YOLO(model_path)
        self.conf       = cfg.get("confidence_threshold", 0.35)
        self.iou_thresh = cfg.get("iou_threshold", 0.45)
        self.classes    = cfg.get("target_classes", [0, 1, 2, 3, 5, 7])
        self.track_win  = cfg.get("track_window", 30)
        self.ema_alpha  = cfg.get("area_ema_alpha", 0.35)
        self.tracker_cfg = cfg.get("tracker_config", "bytetrack.yaml")
        self.device     = cfg.get("device", "cpu")

        self.frame_stride = frame_stride
        self.fps          = video_fps
        self.dt_s         = frame_stride / max(video_fps, EPS)

        # Persistent state (reset each run() call)
        self._objects:    Dict[int, TrackedObject] = {}
        self._area_ema:   Dict[int, float]         = {}
        self._stable_ids: set                      = set()

        # H7 birth/death lists (appended during run())
        self.stable_births: List[Tuple[int, tuple]] = []
        self.stable_deaths: List[Tuple[int, tuple]] = []

    @staticmethod
    def get_video_meta(video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return {"fps": fps, "total_frames": total, "width": w, "height": h}

    def _reset(self) -> None:
        self._objects.clear()
        self._area_ema.clear()
        self._stable_ids.clear()
        self.stable_births.clear()
        self.stable_deaths.clear()

    def run(
        self,
        video_path: str,
        ego_estimator,
    ) -> Iterator[Tuple[int, float, np.ndarray, EgoMotion, Dict[int, TrackedObject]]]:
        """
        Generator. For each processed frame yields:
            (frame_idx, timestamp_s, frame_bgr, ego, active_objects)

        `active_objects` contains only objects seen in THIS frame that have >= 2 snapshots
        (so all heuristics have at least a velocity estimate).
        """
        self._reset()

        try:
            results = self.model.track(
                source=video_path,
                stream=True,
                persist=True,
                tracker=self.tracker_cfg,
                vid_stride=self.frame_stride,
                classes=self.classes,
                conf=self.conf,
                iou=self.iou_thresh,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            # Fall back to botsort if bytetrack config not found
            log.warning(f"tracker config '{self.tracker_cfg}' failed ({e}), falling back to botsort.yaml")
            results = self.model.track(
                source=video_path,
                stream=True,
                persist=True,
                tracker="botsort.yaml",
                vid_stride=self.frame_stride,
                classes=self.classes,
                conf=self.conf,
                iou=self.iou_thresh,
                device=self.device,
                verbose=False,
            )

        frame_count  = 0
        prev_ids: set = set()

        for result in results:
            frame_idx   = frame_count * self.frame_stride
            timestamp_s = frame_idx / max(self.fps, EPS)
            frame       = result.orig_img
            fh, fw      = frame.shape[:2]
            frame_area  = max(fh * fw, 1)

            # --- Ego motion ---
            gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bboxes = []
            if result.boxes is not None and len(result.boxes):
                bboxes = result.boxes.xyxy.cpu().numpy().tolist()
            ego = ego_estimator.update(gray, frame_idx, bboxes)

            # --- Process detections ---
            curr_ids: set = set()

            if result.boxes is not None and result.boxes.id is not None:
                ids   = result.boxes.id.int().cpu().numpy()
                xyxys = result.boxes.xyxy.cpu().numpy()
                clss  = result.boxes.cls.int().cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                for track_id, xyxy, cls_id, conf in zip(ids, xyxys, clss, confs):
                    track_id   = int(track_id)
                    cls_id     = int(cls_id)
                    class_name = COCO_NAMES.get(cls_id, "unknown")

                    x1, y1, x2, y2 = map(float, xyxy)
                    cx_raw = (x1 + x2) / 2.0
                    cy_raw = (y1 + y2) / 2.0
                    cx_comp, cy_comp = compensate_centroid(cx_raw, cy_raw, ego)

                    raw_area   = ((x2 - x1) * (y2 - y1)) / frame_area
                    prev_ema   = self._area_ema.get(track_id, raw_area)
                    smooth_area = self.ema_alpha * raw_area + (1.0 - self.ema_alpha) * prev_ema
                    self._area_ema[track_id] = smooth_area

                    if track_id not in self._objects:
                        hist = deque(maxlen=self.track_win)
                        self._objects[track_id] = TrackedObject(
                            track_id=track_id,
                            class_name=class_name,
                            class_id=cls_id,
                            history=hist,
                            first_seen=frame_idx,
                            last_seen=frame_idx,
                        )
                        dt = 0.0
                    else:
                        dt = self.dt_s
                        # YOLO may refine the class label frame-to-frame; accept majority vote
                        # For simplicity use the latest (ByteTrack usually keeps class consistent)
                        self._objects[track_id].class_name = class_name
                        self._objects[track_id].class_id   = cls_id

                    snap = TrackSnapshot(
                        frame_idx=frame_idx,
                        timestamp_s=timestamp_s,
                        bbox=(x1, y1, x2, y2),
                        centroid_raw=(cx_raw, cy_raw),
                        centroid_comp=(cx_comp, cy_comp),
                        bbox_area_norm=raw_area,
                        bbox_area_smooth=smooth_area,
                        confidence=float(conf),
                        ego=ego,
                        dt_s=dt,
                    )
                    obj = self._objects[track_id]
                    obj.history.append(snap)
                    obj.last_seen = frame_idx
                    curr_ids.add(track_id)

                    # H7 birth detection
                    if len(obj.history) >= self.BIRTH_MIN_FRAMES and track_id not in self._stable_ids:
                        self._stable_ids.add(track_id)
                        self.stable_births.append((frame_idx, (cx_comp, cy_comp)))

            # H7 death detection: IDs that were in prev frame but not this one
            for tid in prev_ids - curr_ids:
                if tid in self._stable_ids:
                    obj = self._objects.get(tid)
                    if obj:
                        snap = obj.last_snapshot()
                        c = snap.centroid_comp if snap else (0.0, 0.0)
                        self.stable_deaths.append((frame_idx, c))

            prev_ids = curr_ids

            # Yield active objects with enough history for heuristics
            active: Dict[int, TrackedObject] = {
                tid: obj
                for tid, obj in self._objects.items()
                if tid in curr_ids and len(obj.history) >= 2
            }

            yield frame_idx, timestamp_s, frame, ego, active
            frame_count += 1
