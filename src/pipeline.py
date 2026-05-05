"""
pipeline.py
-----------
Top-level orchestrator. Wires all modules together and exposes a single
clean public API:

    from src.pipeline import process_video
    result = process_video("data/input/clip.mp4")

Uses YOLOTracker (model.track() + ByteTrack) instead of the old custom
IoU-based tracker. The main loop is significantly simpler as a result.
"""

from __future__ import annotations
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional
import datetime
import logging

import cv2
import yaml

from .yolo_tracker import YOLOTracker, TrackedObject, TrackSnapshot
from .ego_motion   import EgoMotionEstimator, EgoMotion
from .heuristics   import score_frame, init_h5_state
from .scorer       import Scorer, FrameScore
from .segmenter    import Segmenter, HighlightSegment
from .exporter     import Exporter, FrozenTrackBox

log = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def process_video(
    input_path: str,
    config_path: str = "config.yaml",
    output_base: str = "data/output",
    progress_callback=None,
) -> dict:
    """
    End-to-end dashcam highlight extraction.

    Returns
    -------
    {
        "timestamps"      : [...],   # list of highlight dicts with event_types
        "annotated_video" : str|None,
        "clips"           : [...],
        "report"          : str,
        "error"           : null|str,
    }
    """
    empty: dict = {
        "timestamps": [], "annotated_video": None,
        "clips": [], "report": None, "error": None,
    }

    # ------------------------------------------------------------------
    # 0. Config + input validation
    # ------------------------------------------------------------------
    try:
        cfg = _load_config(config_path)
    except Exception as e:
        return {**empty, "error": f"Config load failed: {e}"}

    if not Path(input_path).exists():
        return {**empty, "error": f"Input video not found: {input_path}"}

    def _progress(frac: float, msg: str) -> None:
        log.info(f"[{frac*100:.0f}%] {msg}")
        if progress_callback:
            try:
                progress_callback(frac, msg)
            except Exception:
                pass

    _progress(0.0, "Initialising pipeline...")

    # ------------------------------------------------------------------
    # 1. Output directory
    # ------------------------------------------------------------------
    run_id     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Video metadata
    # ------------------------------------------------------------------
    try:
        meta = YOLOTracker.get_video_meta(input_path)
    except Exception as e:
        return {**empty, "error": f"Cannot read video metadata: {e}"}

    fps          = meta["fps"]
    total_frames = meta["total_frames"]
    frame_stride = cfg["detector"].get("frame_stride", 2)
    total_proc   = max(total_frames // frame_stride, 1)

    _progress(0.02, f"Video: {total_frames} frames @ {fps:.1f} FPS  stride={frame_stride}")

    # ------------------------------------------------------------------
    # 3. Initialise modules
    # ------------------------------------------------------------------
    tracker   = YOLOTracker(cfg["detector"], fps, frame_stride)
    ego_est   = EgoMotionEstimator(cfg["ego_motion"])
    scorer    = Scorer(cfg["segmenter"] | cfg["heuristics"], fps, frame_stride)
    segmenter = Segmenter(cfg["segmenter"])
    exporter  = Exporter(cfg["exporter"], output_dir)

    init_h5_state(cfg["heuristics"].get("complexity_bg_window", 90))
    bg_event_history: deque = deque(maxlen=150)

    write_annotated: bool = cfg["exporter"].get("write_annotated", False)
    frame_track_map: Dict[int, Dict[int, FrozenTrackBox]] = {}
    frame_ego_map:   Dict[int, EgoMotion]                 = {}

    # ------------------------------------------------------------------
    # 4. Main loop  (tracker.run() drives frame reading + ByteTrack)
    # ------------------------------------------------------------------
    _progress(0.05, "Running YOLO tracking + heuristics...")

    processed = 0
    try:
        for frame_idx, ts, frame, ego, active in tracker.run(input_path, ego_est):

            h_scores = score_frame(
                stable_tracks=active,
                ego=ego,
                frame_shape=frame.shape[:2],
                cfg=cfg["heuristics"],
                frame_idx=frame_idx,
                births=tracker.stable_births,
                deaths=tracker.stable_deaths,
                bg_event_history=bg_event_history,
            )

            scorer.accumulate(frame_idx, ts, h_scores)

            if write_annotated:
                frozen: Dict[int, FrozenTrackBox] = {}
                for tid, obj in active.items():
                    snap = obj.last_snapshot()
                    if snap is not None:
                        frozen[tid] = FrozenTrackBox(
                            bbox=snap.bbox,
                            class_name=obj.class_name,
                            track_id=tid,
                            is_stable=True,      # all yielded objects are stable
                        )
                frame_track_map[frame_idx] = frozen
                frame_ego_map[frame_idx]   = ego

            processed += 1
            if processed % 30 == 0:
                frac = 0.05 + 0.75 * (processed / total_proc)
                _progress(frac, f"Processed {processed}/{total_proc} frames")

    except Exception as e:
        log.exception("Error in main processing loop")
        return {**empty, "error": f"Processing failed: {e}"}

    if not scorer.frame_scores:
        return {**empty, "error": "No frames processed (empty video?)."}

    # ------------------------------------------------------------------
    # 5. Smooth + segment
    # ------------------------------------------------------------------
    _progress(0.80, "Smoothing scores and extracting highlights...")
    frame_scores = scorer.smooth()
    segments     = segmenter.extract(frame_scores)

    if not segments:
        _progress(1.0, "No highlights found above threshold.")
        return {
            **empty,
            "annotated_video": None,
            "clips": [],
            "report": str(exporter.write_report([])),
        }

    # ------------------------------------------------------------------
    # 6. Annotated video (opt-in)
    # ------------------------------------------------------------------
    ann_path = None
    if write_annotated:
        highlight_frames: set = set()
        for seg in segments:
            for fi in range(int(seg.start_s * fps), int(seg.end_s * fps) + 1):
                highlight_frames.add(fi)

        _progress(0.83, "Writing annotated video...")
        try:
            ann_path = exporter.write_annotated_video(
                video_path=input_path,
                frame_scores=frame_scores,
                frame_track_map=frame_track_map,
                frame_ego_map=frame_ego_map,
                highlight_frames=highlight_frames,
                fps=fps,
                frame_stride=frame_stride,
            )
        except Exception as e:
            log.error(f"Annotated video export failed: {e}")
    else:
        log.info("Annotated video skipped (write_annotated=false).")

    # ------------------------------------------------------------------
    # 7. Clips + report
    # ------------------------------------------------------------------
    _progress(0.88, "Writing highlight clips...")
    try:
        segments = exporter.write_clips(input_path, segments, fps)
    except Exception as e:
        log.error(f"Clip export failed: {e}")

    report_path = exporter.write_report(segments)
    _progress(1.0, f"Done — {len(segments)} highlight(s) saved to {output_dir}")

    return {
        "timestamps":       [seg.to_dict() for seg in segments],
        "annotated_video":  str(ann_path) if ann_path else None,
        "clips":            [seg.clip_path for seg in segments if seg.clip_path],
        "report":           str(report_path),
        "error":            None,
    }
