"""
segmenter.py
------------
Converts a list of FrameScore objects into HighlightSegment objects by:
  1. Thresholding smoothed composite score
  2. Gap-filling (merge segments separated by short gaps)
  3. Filtering by minimum / maximum duration
  4. Returning top-K by peak score
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import logging

import numpy as np

from .scorer import FrameScore
from .heuristics import HEURISTIC_LABELS

log = logging.getLogger(__name__)

HEURISTIC_NAMES = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"]
DOMINANT_THRESHOLD = 0.35    # min mean raw score for a heuristic to appear in event_types


@dataclass
class HighlightSegment:
    start_s:              float
    end_s:                float
    peak_score:           float
    mean_score:           float
    triggered_heuristics: List[str]
    clip_path:            Optional[str] = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict:
        return {
            "start":   round(self.start_s, 2),
            "end":     round(self.end_s, 2),
            "duration": round(self.duration_s, 2),
            "peak_score": round(self.peak_score, 4),
            "mean_score": round(self.mean_score, 4),
            # Internal codes kept for debugging
            "triggered_heuristics": self.triggered_heuristics,
            # Human-readable labels for end users
            "event_types": [
                HEURISTIC_LABELS.get(h, h) for h in self.triggered_heuristics
            ],
            "clip": self.clip_path,
        }


class Segmenter:
    def __init__(self, cfg: dict):
        self.threshold     = cfg["score_threshold"]
        self.gap_fill_s    = cfg["gap_fill_s"]
        self.min_dur_s     = cfg["min_duration_s"]
        self.max_dur_s     = cfg["max_duration_s"]
        self.max_highlights = cfg["max_highlights"]

    def extract(self, frame_scores: List[FrameScore]) -> List[HighlightSegment]:
        """
        Full segmentation pipeline.
        Returns list of HighlightSegment sorted by peak_score descending.
        """
        if not frame_scores:
            log.info("No frame scores — returning empty highlight list.")
            return []

        # --- Step 1: collect frames above threshold ---
        above = [fs for fs in frame_scores if fs.smoothed >= self.threshold]
        if not above:
            log.info("No frames above score threshold — no highlights.")
            return []

        # --- Step 2: group into contiguous runs (with gap filling) ---
        groups = self._group_frames(frame_scores)

        # --- Step 3: build HighlightSegment per group ---
        segments = []
        for group in groups:
            seg = self._build_segment(group)
            if seg is not None:
                segments.append(seg)

        # --- Step 4: sort by peak score, return top-K ---
        segments.sort(key=lambda s: s.peak_score, reverse=True)
        top = segments[:self.max_highlights]
        # Re-sort chronologically for output clarity
        top.sort(key=lambda s: s.start_s)
        log.info(f"Extracted {len(top)} highlight segments.")
        return top

    # ------------------------------------------------------------------
    def _group_frames(self, frame_scores: List[FrameScore]) -> List[List[FrameScore]]:
        """
        Group frames into highlight runs.
        Gaps shorter than gap_fill_s are bridged.
        """
        groups: List[List[FrameScore]] = []
        current: List[FrameScore] = []
        last_above_ts: float | None = None

        for fs in frame_scores:
            is_above = fs.smoothed >= self.threshold

            if is_above:
                if current and last_above_ts is not None:
                    gap = fs.timestamp_s - last_above_ts
                    if gap > self.gap_fill_s:
                        groups.append(current)
                        current = []
                current.append(fs)
                last_above_ts = fs.timestamp_s
            else:
                if current and last_above_ts is not None:
                    gap = fs.timestamp_s - last_above_ts
                    if gap > self.gap_fill_s:
                        groups.append(current)
                        current = []
                    else:
                        # Bridge: include this frame in current group
                        current.append(fs)

        if current:
            groups.append(current)

        return groups

    def _build_segment(self, group: List[FrameScore]) -> Optional[HighlightSegment]:
        """Build one HighlightSegment from a group. Apply duration filters."""
        if not group:
            return None

        start_s = group[0].timestamp_s
        end_s   = group[-1].timestamp_s
        dur     = end_s - start_s

        if dur < self.min_dur_s:
            return None

        scores  = np.array([fs.smoothed for fs in group])
        peak    = float(scores.max())
        mean    = float(scores.mean())

        # Duration cap: centre on peak
        if dur > self.max_dur_s:
            peak_idx  = int(np.argmax(scores))
            peak_fs   = group[peak_idx]
            half      = self.max_dur_s / 2.0
            start_s   = max(group[0].timestamp_s, peak_fs.timestamp_s - half)
            end_s     = min(group[-1].timestamp_s, peak_fs.timestamp_s + half)
            group = [fs for fs in group if start_s <= fs.timestamp_s <= end_s]
            if not group:
                return None

        # Triggered heuristics: mean raw score > threshold per heuristic
        triggered = []
        for hname in HEURISTIC_NAMES:
            vals = [fs.heuristic_scores.get(hname, 0.0) for fs in group]
            if np.mean(vals) >= DOMINANT_THRESHOLD:
                triggered.append(hname)

        return HighlightSegment(
            start_s=round(start_s, 2),
            end_s=round(end_s, 2),
            peak_score=peak,
            mean_score=mean,
            triggered_heuristics=triggered,
        )
