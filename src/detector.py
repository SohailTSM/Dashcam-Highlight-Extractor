"""
detector.py
-----------
Thin wrapper around YOLOv8n that yields FrameDetection objects.
Only target COCO classes are returned (person, bicycle, car, motorcycle, bus, truck).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List
import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

# COCO class names for reference
COCO_NAMES: dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck",
}


@dataclass
class Detection:
    """Single object detected in one frame."""
    track_id: int           # assigned later by tracker; -1 until assigned
    class_id: int
    class_name: str
    bbox: tuple             # (x1, y1, x2, y2) in pixels, floats
    confidence: float
    centroid: tuple         # (cx, cy) derived from bbox
    bbox_area_norm: float   # bbox_area / (H * W), normalised [0, 1]


@dataclass
class FrameDetection:
    """All detections for a single frame."""
    frame_idx: int
    timestamp_s: float
    detections: List[Detection]
    frame_shape: tuple      # (H, W)


class Detector:
    """
    Wraps YOLOv8n for frame-by-frame inference.

    Usage
    -----
    detector = Detector(cfg)
    for frame_det, frame in detector.run(video_path):
        ...
    """

    def __init__(self, cfg: dict):
        from ultralytics import YOLO  # deferred import to keep startup fast
        model_path = cfg["model_path"]
        if not Path(model_path).exists():
            raise FileNotFoundError(f"YOLO weights not found: {model_path}")
        self.model = YOLO(model_path)
        self.conf = cfg["confidence_threshold"]
        self.iou  = cfg["iou_threshold"]
        self.target_classes: list[int] = cfg["target_classes"]
        self.frame_stride: int = cfg.get("frame_stride", 1)

    def run(
        self,
        video_path: str,
    ) -> Generator[tuple[FrameDetection, np.ndarray], None, None]:
        """
        Yields (FrameDetection, raw_frame) for every *processed* frame.
        Skipped frames (due to stride) are NOT yielded.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.frame_stride == 0:
                    h, w = frame.shape[:2]
                    timestamp_s = frame_idx / fps
                    detections = self._infer(frame, h, w)
                    yield FrameDetection(
                        frame_idx=frame_idx,
                        timestamp_s=timestamp_s,
                        detections=detections,
                        frame_shape=(h, w),
                    ), frame

                frame_idx += 1
        finally:
            cap.release()

    def _infer(self, frame: np.ndarray, h: int, w: int) -> list[Detection]:
        """Run YOLO on a single frame and return filtered Detection list."""
        try:
            results = self.model(
                frame,
                conf=self.conf,
                iou=self.iou,
                classes=self.target_classes,
                verbose=False,
            )
        except Exception as e:
            log.warning(f"YOLO inference failed on frame: {e}")
            return []

        detections: list[Detection] = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in self.target_classes:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area_norm = ((x2 - x1) * (y2 - y1)) / max(h * w, 1)
            detections.append(Detection(
                track_id=-1,
                class_id=cls_id,
                class_name=COCO_NAMES.get(cls_id, str(cls_id)),
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                centroid=(cx, cy),
                bbox_area_norm=area_norm,
            ))

        return detections

    @staticmethod
    def get_video_meta(video_path: str) -> dict:
        """Return basic video metadata without opening the full stream."""
        cap = cv2.VideoCapture(video_path)
        meta = {
            "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        cap.release()
        return meta
