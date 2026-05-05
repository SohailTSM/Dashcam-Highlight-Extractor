"""
exporter.py
-----------
Pure consumer of pipeline outputs. Does NOT recompute any scores.

Produces:
  - annotated.mp4  : full video with bboxes, per-frame score bar, highlight glow
  - highlight_N.mp4: one clip per segment, padded by clip_padding_s
  - report.json    : structured JSON with all timestamps and heuristic info

All writing is done through OpenCV; no ffmpeg dependency required.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, NamedTuple
import json
import logging

import cv2
import numpy as np

from .scorer import FrameScore
from .segmenter import HighlightSegment
from .ego_motion import EgoMotion

log = logging.getLogger(__name__)


class FrozenTrackBox(NamedTuple):
    """
    Immutable snapshot of a single track's display data at one processed frame.
    Stored instead of TrackState references to avoid the mutability bug where
    track.last_snapshot() returns the end-of-video state at annotation time.
    """
    bbox:       tuple   # (x1, y1, x2, y2) in pixels at this frame
    class_name: str
    track_id:   int
    is_stable:  bool

# Colour palette (BGR)
COL_BOX      = (0, 200, 255)    # orange – normal bbox
COL_STABLE   = (0, 255, 120)    # green  – stable track bbox
COL_GLOW     = (0, 0, 200)      # red    – highlight frame border
COL_SCORE    = (255, 215, 0)    # gold   – score bar fill
COL_TEXT     = (255, 255, 255)
COL_EGO      = (255, 100, 50)   # cyan   – ego arrow


def _draw_score_bar(frame: np.ndarray, score: float, bar_h: int = 20) -> None:
    """Draw a horizontal score bar at the top of the frame."""
    h, w = frame.shape[:2]
    fill_w = int(w * score)
    cv2.rectangle(frame, (0, 0), (w, bar_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, 0), (fill_w, bar_h), COL_SCORE, -1)
    label = f"Score: {score:.2f}"
    cv2.putText(frame, label, (6, bar_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_TEXT, 1, cv2.LINE_AA)


def _draw_highlight_glow(frame: np.ndarray, thickness: int = 8) -> None:
    """Draw a red border around the frame to indicate highlight."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COL_GLOW, thickness)


def _draw_ego_arrow(frame: np.ndarray, ego: EgoMotion) -> None:
    """Draw a small arrow indicating ego-motion direction."""
    if not ego.valid:
        return
    h, w = frame.shape[:2]
    cx, cy = w // 2, h - 30
    ex = int(cx + ego.tx * 2)
    ey = int(cy + ego.ty * 2)
    cv2.arrowedLine(frame, (cx, cy), (ex, ey), COL_EGO, 2, tipLength=0.4)
    cv2.putText(frame, f"ego tx={ego.tx:.1f} ty={ego.ty:.1f}",
                (4, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_EGO, 1, cv2.LINE_AA)


def _draw_tracks(
    frame: np.ndarray,
    frozen_tracks: Dict[int, FrozenTrackBox],
) -> None:
    """Draw bounding boxes for all stable tracks at their frame-time positions."""
    for fbox in frozen_tracks.values():
        x1, y1, x2, y2 = [int(v) for v in fbox.bbox]
        colour = COL_STABLE if fbox.is_stable else COL_BOX
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = f"{fbox.class_name}#{fbox.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 2, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)


class Exporter:
    """
    Writes all outputs for a single pipeline run.
    All methods are idempotent and safe to call even if upstream data is empty.
    """

    def __init__(self, cfg: dict, output_dir: Path):
        self.padding_s = cfg.get("clip_padding_s", 0.5)
        self.bar_h     = cfg.get("score_bar_height", 20)
        self.fourcc    = cv2.VideoWriter_fourcc(*cfg.get("fourcc", "mp4v"))
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Annotated full video
    # ------------------------------------------------------------------

    def write_annotated_video(
        self,
        video_path: str,
        frame_scores: List[FrameScore],
        frame_track_map: Dict[int, Dict[int, FrozenTrackBox]],
        frame_ego_map:   Dict[int, EgoMotion],
        highlight_frames: set,
        fps: float,
        frame_stride: int,
    ) -> Path:
        """
        Re-reads the source video and writes an annotated copy.
        frame_track_map and frame_ego_map are indexed by processed frame_idx.
        For skipped frames, the previous frame's data is re-used.
        """
        out_path = self.output_dir / "annotated.mp4"
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.error(f"Cannot reopen video for annotation: {video_path}")
            return out_path

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_path), self.fourcc, fps, (w, h))

        # Build lookup: frame_idx → FrameScore (by processed idx)
        score_lookup: Dict[int, FrameScore] = {fs.frame_idx: fs for fs in frame_scores}

        last_score   = FrameScore(0, 0.0, 0.0, {})
        last_tracks: Dict[int, FrozenTrackBox] = {}
        last_ego     = EgoMotion(frame_idx=0)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Use processed data if available; else repeat previous
            if frame_idx in score_lookup:
                last_score  = score_lookup[frame_idx]
                last_tracks = frame_track_map.get(frame_idx, {})
                last_ego    = frame_ego_map.get(frame_idx, last_ego)

            is_highlight = frame_idx in highlight_frames
            _draw_tracks(frame, last_tracks)
            _draw_score_bar(frame, last_score.smoothed, self.bar_h)
            _draw_ego_arrow(frame, last_ego)
            if is_highlight:
                _draw_highlight_glow(frame)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        
        # Transcode to H.264 for web browser compatibility
        self._transcode_to_h264(out_path)
        
        log.info(f"Annotated video saved: {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Highlight clips
    # ------------------------------------------------------------------

    def write_clips(
        self,
        video_path: str,
        segments: List[HighlightSegment],
        fps: float,
    ) -> List[HighlightSegment]:
        """
        Write one clip per segment. Attaches clip_path to each segment (in-place).
        Returns the updated list.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.error(f"Cannot open video for clip export: {video_path}")
            return segments

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        for i, seg in enumerate(segments):
            clip_path = self.output_dir / f"highlight_{i:02d}.mp4"
            try:
                self._extract_clip(video_path, seg, fps, (w, h), clip_path)
                seg.clip_path = str(clip_path)
            except Exception as e:
                log.error(f"Failed to write clip {i}: {e}")

        return segments

    def _extract_clip(
        self,
        video_path: str,
        seg: HighlightSegment,
        fps: float,
        size: tuple,
        out_path: Path,
    ) -> None:
        start_s = max(0.0, seg.start_s - self.padding_s)
        end_s   = seg.end_s + self.padding_s

        cap = cv2.VideoCapture(video_path)
        start_frame = int(start_s * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        writer = cv2.VideoWriter(str(out_path), self.fourcc, fps, size)
        frame_idx = start_frame

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts = frame_idx / fps
            if ts > end_s:
                break
            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        
        # Transcode to H.264 for web browser compatibility
        self._transcode_to_h264(out_path)
        
        log.info(f"Clip saved: {out_path} [{start_s:.1f}s – {end_s:.1f}s]")

    def _transcode_to_h264(self, filepath: Path) -> None:
        """
        OpenCV's mp4v codec is not supported by web browsers (Gradio).
        OpenCV's avc1 (H.264) codec often crashes on Linux due to hardware bugs.
        Solution: Let OpenCV write mp4v safely, then use FFmpeg to silently transcode to H.264.
        """
        import os, subprocess
        tmp_path = str(filepath) + ".tmp.mp4"
        try:
            os.rename(filepath, tmp_path)
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", tmp_path,
                "-c:v", "libx264", "-preset", "fast",
                str(filepath)
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                log.warning(f"Failed to transcode to H.264, falling back to mp4v. {res.stderr}")
                os.rename(tmp_path, filepath)  # revert on failure
            else:
                os.remove(tmp_path)
        except Exception as e:
            log.warning(f"Transcode error: {e}")

    # ------------------------------------------------------------------
    # JSON Report
    # ------------------------------------------------------------------

    def write_report(self, segments: List[HighlightSegment]) -> Path:
        report_path = self.output_dir / "report.json"
        data = {
            "total_highlights": len(segments),
            "highlights": [seg.to_dict() for seg in segments],
        }
        with open(report_path, "w") as f:
            json.dump(data, f, indent=2)
        log.info(f"Report saved: {report_path}")
        return report_path
