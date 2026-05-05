"""
ego_motion.py
-------------
Estimates per-frame camera (ego) motion using sparse Lucas-Kanade optical flow
on background keypoints, then fits an affine transform via RANSAC.

Outputs an EgoMotion object per frame containing:
  tx, ty   — translation components (px)
  angle    — rotation (radians)
  scale    — zoom factor (1.0 = none)
  M        — full 2x3 affine matrix (for centroid warping)
  M_inv    — inverse affine (for centroid compensation)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class EgoMotion:
    frame_idx: int
    tx: float = 0.0
    ty: float = 0.0
    angle: float = 0.0      # radians
    scale: float = 1.0
    confidence: float = 0.0  # inlier fraction
    valid: bool = False
    M: np.ndarray = field(default_factory=lambda: np.eye(2, 3, dtype=np.float32))
    M_inv: np.ndarray = field(default_factory=lambda: np.eye(2, 3, dtype=np.float32))


# LK optical flow parameters
_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def _build_bg_mask(frame_shape: tuple, bboxes: list[tuple]) -> np.ndarray:
    """
    Build a binary mask that is 255 on background pixels.
    Foreground = dilated union of YOLO bboxes.
    """
    h, w = frame_shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    pad = 10
    for (x1, y1, x2, y2) in bboxes:
        x1i = max(0, int(x1) - pad)
        y1i = max(0, int(y1) - pad)
        x2i = min(w, int(x2) + pad)
        y2i = min(h, int(y2) + pad)
        mask[y1i:y2i, x1i:x2i] = 0
    return mask


def _decompose_affine(M: np.ndarray) -> tuple[float, float, float, float]:
    """Extract (tx, ty, scale, angle) from a 2x3 affine matrix."""
    tx = float(M[0, 2])
    ty = float(M[1, 2])
    scale = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
    angle = float(np.arctan2(M[1, 0], M[0, 0]))
    return tx, ty, scale, angle


class EgoMotionEstimator:
    """
    Stateful: keeps previous frame's grayscale + keypoints between calls.

    Usage
    -----
    estimator = EgoMotionEstimator(cfg)
    for each processed frame:
        ego = estimator.update(frame_gray, frame_idx, bboxes)
    """

    def __init__(self, cfg: dict):
        self.use_affine: bool = cfg.get("use_affine", True)
        self.max_corners: int = cfg.get("max_corners", 200)
        self.quality: float = cfg.get("quality_level", 0.01)
        self.min_dist: float = cfg.get("min_distance", 10)
        self.min_bg_pts: int = cfg.get("min_bg_points", 10)
        self.fb_thresh: float = cfg.get("fb_error_threshold", 1.0)
        self.ransac_thresh: float = cfg.get("ransac_threshold", 3.0)
        self.min_conf: float = cfg.get("min_confidence", 0.2)

        self._prev_gray: np.ndarray | None = None
        self._prev_pts:  np.ndarray | None = None   # shape (N,1,2)

    def update(
        self,
        gray: np.ndarray,
        frame_idx: int,
        bboxes: list[tuple],
    ) -> EgoMotion:
        """
        Estimate ego-motion between previous and current gray frame.

        Parameters
        ----------
        gray       : current frame as uint8 grayscale
        frame_idx  : index of the current frame
        bboxes     : list of (x1,y1,x2,y2) from current detections (used for BG mask)
        """
        result = EgoMotion(frame_idx=frame_idx)

        if self._prev_gray is None:
            # First frame — nothing to estimate
            self._prev_gray = gray
            self._prev_pts  = self._detect_keypoints(gray, bboxes)
            return result

        if self._prev_pts is None or len(self._prev_pts) < self.min_bg_pts:
            self._prev_gray = gray
            self._prev_pts  = self._detect_keypoints(gray, bboxes)
            return result

        # ---------- Lucas-Kanade tracking ----------
        curr_pts, st, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **_LK_PARAMS
        )

        # Forward-backward consistency check
        prev_pts_back, st_back, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._prev_gray, curr_pts, None, **_LK_PARAMS
        )
        fb_error = np.linalg.norm(
            self._prev_pts - prev_pts_back, axis=2
        ).squeeze()
        good_mask = (
            (st.squeeze() == 1) &
            (st_back.squeeze() == 1) &
            (fb_error < self.fb_thresh)
        )

        pts_prev_good = self._prev_pts[good_mask].reshape(-1, 1, 2)
        pts_curr_good = curr_pts[good_mask].reshape(-1, 1, 2)

        if len(pts_prev_good) < self.min_bg_pts:
            log.debug(f"Frame {frame_idx}: too few flow points ({len(pts_prev_good)}), skipping ego")
            self._prev_gray = gray
            self._prev_pts  = self._detect_keypoints(gray, bboxes)
            return result

        # ---------- Affine estimation with RANSAC ----------
        try:
            M, inliers = cv2.estimateAffinePartial2D(
                pts_prev_good, pts_curr_good,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_thresh,
            )
        except Exception as e:
            log.debug(f"Frame {frame_idx}: affine estimation failed: {e}")
            self._prev_gray = gray
            self._prev_pts  = self._detect_keypoints(gray, bboxes)
            return result

        if M is None:
            self._prev_gray = gray
            self._prev_pts  = self._detect_keypoints(gray, bboxes)
            return result

        conf = float(inliers.sum()) / max(len(pts_prev_good), 1) if inliers is not None else 0.0

        if conf < self.min_conf:
            log.debug(f"Frame {frame_idx}: low ego-motion confidence ({conf:.2f}), using identity")
            self._prev_gray = gray
            self._prev_pts  = self._detect_keypoints(gray, bboxes)
            return result

        tx, ty, scale, angle = _decompose_affine(M)

        # Compute inverse affine for centroid compensation
        M_full = np.vstack([M, [0, 0, 1]])        # 3x3
        try:
            M_inv_full = np.linalg.inv(M_full)
            M_inv = M_inv_full[:2, :]               # back to 2x3
        except np.linalg.LinAlgError:
            M_inv = np.eye(2, 3, dtype=np.float32)

        result.tx = tx
        result.ty = ty
        result.angle = angle
        result.scale = scale
        result.confidence = conf
        result.valid = True
        result.M = M.astype(np.float32)
        result.M_inv = M_inv.astype(np.float32)

        # ---------- Update state for next frame ----------
        self._prev_gray = gray
        self._prev_pts  = self._detect_keypoints(gray, bboxes)

        return result

    def _detect_keypoints(
        self,
        gray: np.ndarray,
        bboxes: list[tuple],
    ) -> np.ndarray | None:
        """Detect Shi-Tomasi corners on background pixels."""
        mask = _build_bg_mask(gray.shape, bboxes)
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality,
            minDistance=self.min_dist,
            mask=mask,
        )
        return pts  # shape (N,1,2) or None


def compensate_centroid(cx: float, cy: float, ego: EgoMotion) -> tuple[float, float]:
    """
    Apply inverse affine transform to a raw centroid.
    Returns the ego-compensated centroid (object motion in camera-relative coords).
    """
    if not ego.valid:
        return cx, cy
    pt = np.array([cx, cy, 1.0], dtype=np.float64)
    comp = ego.M_inv @ pt   # shape (2,)
    return float(comp[0]), float(comp[1])
