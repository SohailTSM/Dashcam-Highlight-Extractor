"""
scorer.py
---------
Aggregates per-frame heuristic raw scores into weighted composite FrameScores,
then applies Gaussian temporal smoothing over the full video.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import logging

import numpy as np
from scipy.ndimage import gaussian_filter1d

log = logging.getLogger(__name__)

# Weight keys must match heuristic names exactly
WEIGHTS_KEYS = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"]
WEIGHT_MAP = {
    "H1": "w_rapid_approach",
    "H2": "w_lateral_cutin",
    "H3": "w_sudden_braking",
    "H4": "w_nearmiss_proximity",
    "H5": "w_scene_complexity",
    "H6": "w_pedestrian_road",
    "H7": "w_birth_death_rate",
    "H8": "w_ttc",
}


@dataclass
class FrameScore:
    frame_idx:        int
    timestamp_s:      float
    composite:        float                   # weighted sum before smoothing
    heuristic_scores: dict[str, float]        # raw per-heuristic values
    smoothed:         float = 0.0             # filled in after full-video pass


class Scorer:
    """
    Accumulates FrameScore objects during video processing,
    then smooths the composite score series.
    """

    def __init__(self, cfg: dict, video_fps: float, frame_stride: int):
        self.weights: dict[str, float] = {
            hk: cfg[WEIGHT_MAP[hk]] for hk in WEIGHTS_KEYS
        }
        # Normalise weights to sum to 1.0
        total = sum(self.weights.values()) + 1e-9
        self.weights = {k: v / total for k, v in self.weights.items()}

        self.sigma_s:    float = cfg.get("smooth_sigma_s", 0.5)
        self.effective_fps = video_fps / max(frame_stride, 1)

        self._scores: List[FrameScore] = []

    def accumulate(
        self,
        frame_idx: int,
        timestamp_s: float,
        heuristic_scores: dict[str, float],
    ) -> FrameScore:
        """Compute weighted composite and store FrameScore."""
        composite = sum(
            self.weights.get(k, 0.0) * v
            for k, v in heuristic_scores.items()
        )
        fs = FrameScore(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            composite=float(np.clip(composite, 0.0, 1.0)),
            heuristic_scores=heuristic_scores,
        )
        self._scores.append(fs)
        return fs

    def smooth(self) -> List[FrameScore]:
        """
        Apply Gaussian smoothing to the composite score series (in-place on smoothed field).
        sigma is in seconds, converted to frames using effective_fps.
        """
        if not self._scores:
            return self._scores

        sigma_frames = max(self.sigma_s * self.effective_fps, 0.5)
        raw = np.array([fs.composite for fs in self._scores], dtype=np.float32)
        smoothed = gaussian_filter1d(raw, sigma=sigma_frames)

        for fs, s in zip(self._scores, smoothed):
            fs.smoothed = float(np.clip(s, 0.0, 1.0))

        return self._scores

    @property
    def frame_scores(self) -> List[FrameScore]:
        return self._scores

    def reset(self) -> None:
        self._scores = []
