---
title: Dashcam Highlight Extractor
emoji: 🚗
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Dashcam Highlight Extractor

A CPU-friendly pipeline for automatically extracting dangerous or noteworthy segments from dashcam footage using rule-based heuristics on YOLO-tracked object trajectories.

---

## Features

- Persistent object tracking via **YOLOv8n + ByteTrack** — no custom IoU matching
- **8 physics-informed heuristics** covering approach, cut-ins, braking, proximity, scene complexity, pedestrian risk, and time-to-collision
- **Ego-motion compensation** (sparse Lucas-Kanade optical flow + RANSAC affine) removes camera movement from object trajectories
- Outputs timestamped JSON report, extracted MP4 clips, and an optional annotated video
- **Gradio web UI** (`app.py`) and **CLI** (`run.py`) entry points
- Fully configurable via a single `config.yaml`

---

## Requirements

```bash
Python 3.10+
```

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the YOLOv8n weights (done automatically on first run, or manually):

```bash
mkdir -p weights
# weights/yolov8n.pt is fetched by ultralytics on first model load
```

---

## Usage

### CLI

```bash
python run.py --input data/input/clip.mp4
```

**All options:**

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to the dashcam video |
| `--config` | `config.yaml` | Configuration file |
| `--output` | `data/output` | Base directory for all output files |
| `--threshold` | 0.28 | Composite score threshold (0–1) to create a highlight |
| `--max-highlights` | 30 | Maximum number of highlights to extract |
| `--stride` | 2 | Process every Nth frame (higher = faster, lower = more accurate) |
| `--annotated` | off | Write a full annotated video with bounding boxes and score overlay |

**Examples:**

```bash
# Basic run
python run.py --input data/input/dashcam.mp4

# Stricter threshold, fewer results
python run.py --input data/input/dashcam.mp4 --threshold 0.35 --max-highlights 10

# Include annotated video output
python run.py --input data/input/dashcam.mp4 --annotated

# Process faster (skip more frames)
python run.py --input data/input/dashcam.mp4 --stride 4
```

### Web UI (Gradio)

```bash
python app.py
```

Open the printed URL in your browser. Upload a video, adjust parameters, and download results directly.

---

## Outputs

All outputs are written to a timestamped subdirectory, e.g. `data/output/20260502_231920/`:

| File | Description |
|---|---|
| `report.json` | Full JSON report with timestamps, scores, and event labels |
| `highlight_01.mp4` … | Extracted MP4 clips, one per detected segment |
| `annotated.mp4` | *(optional)* Full video with bounding boxes and score bar overlay |

### report.json structure

```json
{
  "input": "data/input/clip.mp4",
  "highlights": [
    {
      "start": 12.4,
      "end": 15.8,
      "duration": 3.4,
      "peak_score": 0.612,
      "mean_score": 0.481,
      "triggered_heuristics": ["H1", "H8"],
      "event_types": ["Sudden Approach", "Collision Risk"],
      "clip": "data/output/.../highlight_01.mp4"
    }
  ]
}
```

**Event type labels:**

| Code | Label |
|---|---|
| H1 | Sudden Approach |
| H2 | Lane Cut-In |
| H3 | Sudden Braking |
| H4 | Close Proximity |
| H5 | Chaotic Traffic |
| H6 | Pedestrian / Cyclist Hazard |
| H7 | High Traffic Activity |
| H8 | Collision Risk |

---

## Project Structure

```
dashcam_extractor/
│
├── run.py               # CLI entry point
├── app.py               # Gradio web UI entry point
├── config.yaml          # All tunable parameters
├── requirements.txt
│
├── weights/
│   └── yolov8n.pt       # COCO-pretrained YOLOv8 nano weights
│
├── data/
│   ├── input/           # Place source videos here
│   └── output/          # Auto-created per-run output directories
│
└── src/
    ├── __init__.py
    ├── pipeline.py      # Top-level orchestrator — wires all modules
    ├── yolo_tracker.py  # YOLOv8 model.track() wrapper + ByteTrack + history accumulation
    ├── ego_motion.py    # LK optical flow + RANSAC affine ego-motion estimation
    ├── heuristics.py    # 8 rule-based heuristic scoring functions (H1–H8)
    ├── scorer.py        # Per-frame score accumulation and Gaussian smoothing
    ├── segmenter.py     # Threshold → gap-fill → filter → top-K highlight extraction
    └── exporter.py      # Clip export (FFmpeg via OpenCV) and annotated video writer
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `pipeline.py` | Coordinates the full pipeline; the only module `run.py` and `app.py` call |
| `yolo_tracker.py` | Runs `model.track()` on each frame; accumulates per-track `TrackSnapshot` history with EMA-smoothed bounding box areas; reports birth/death events for H7 |
| `ego_motion.py` | Tracks Shi-Tomasi corners on background pixels; fits a partial affine transform via RANSAC to estimate camera translation, rotation, and zoom per frame |
| `heuristics.py` | Implements H1–H8; each function receives the current active track dictionary and ego motion and returns a 0–1 score |
| `scorer.py` | Computes a weighted composite score per frame from H1–H8 sub-scores; applies Gaussian temporal smoothing post-hoc |
| `segmenter.py` | Converts the smoothed score curve into highlight segments by thresholding, gap-filling, duration filtering, and top-K ranking |
| `exporter.py` | Writes MP4 clips, the JSON report, and (optionally) a full annotated video |

---

## Configuration

All parameters live in `config.yaml`. The most commonly tuned values:

```yaml
detector:
  frame_stride: 2          # 1 = every frame (slow), 4 = every 4th (fast)
  confidence_threshold: 0.35
  tracker_config: bytetrack.yaml   # or botsort.yaml

heuristics:
  area_rel_max: 0.08       # H1: how fast an object must grow to trigger Sudden Approach
  ttc_max_s: 2.5           # H8: TTC below this is flagged as Collision Risk

segmenter:
  score_threshold: 0.28    # lower = more highlights, higher = fewer
  max_highlights: 30
  min_duration_s: 1.5
```

---

## Notes

- **GPU is not required.** The pipeline runs fully on CPU; on a modern laptop expect ~1–3× realtime speed at `frame_stride=2`.
- **ByteTrack fallback:** If `bytetrack.yaml` is not found in your ultralytics installation, the pipeline automatically falls back to `botsort.yaml`.
- `tracker.py` and `detector.py` are legacy modules kept for reference. They are **not used** by the current pipeline; `yolo_tracker.py` supersedes both.
