# Dashcam Highlight Extractor: A Rule-Based Heuristic Pipeline for Dangerous Event Detection in Dashcam Video

**GitHub Repository:** [Dashcam-Highlight-Extractor](https://github.com/SohailTSM/Dashcam-Highlight-Extractor)  
**Screen Demo:** [Dashcam-Highlight-Extractor]()
**Live App:** [Dashcam-Highlight-Extractor](https://)

**Team Name:** Irrelevant  
**Team Members:**  
- Sohail Memon, 2025201075  
- Md Taufique Hussain, 2025202007  
- Mohd Ahmad, 2025201029  
- Kathan Patel, 2025201039  
- Mohd Shahid Kaleem, 2025204008  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Data](#2-data)
3. [Method](#3-method)
   - 3.1 [System Overview](#31-system-overview)
   - 3.2 [Detection and Tracking](#32-detection-and-tracking)
   - 3.3 [Ego-Motion Compensation](#33-ego-motion-compensation)
   - 3.4 [Heuristic Scoring (H1–H8)](#34-heuristic-scoring-h1h8)
   - 3.5 [Score Aggregation and Smoothing](#35-score-aggregation-and-smoothing)
   - 3.6 [Highlight Segmentation](#36-highlight-segmentation)
4. [Ablation Study](#4-ablation-study)
   - 4.1 [Tracker: Custom IoU vs. ByteTrack](#41-tracker-custom-iou-vs-bytetrack)
   - 4.2 [Ego-Motion Compensation](#42-ego-motion-compensation)
   - 4.3 [EMA Smoothing of Bounding Box Area](#43-ema-smoothing-of-bounding-box-area)
   - 4.4 [Adaptive Area Blending for Short Tracks](#44-adaptive-area-blending-for-short-tracks)
   - 4.5 [Minimum History Window Tuning](#45-minimum-history-window-tuning)
5. [Results](#5-results)
6. [Limitations](#6-limitations)
7. [References](#7-references)

---

## 1. Introduction

Dashcam footage is an increasingly prevalent source of evidence for traffic safety analysis. Manually reviewing hours of continuous driving video to locate dangerous events is laborious and impractical. Automated systems are needed that can temporally localise near-misses, sudden cut-ins, pedestrian intrusions, and similar critical events with high recall and acceptable precision.

Existing approaches to highlight extraction fall into two broad categories:

1. **Deep learning-based scoring** — end-to-end networks (e.g., RNNs, transformers) trained on labelled dashcam datasets to directly predict danger scores. These require large annotated corpora and are difficult to interpret or calibrate.
2. **Rule-based heuristic systems** — hand-crafted features derived from object trajectories, applied as logical or arithmetic tests. These are transparent, computationally inexpensive, and do not require danger-labelled training data.

This work implements the second approach. Our system processes a dashcam video through a detection-and-tracking front-end, estimates and removes ego (camera) motion, then computes eight temporally-aware heuristic scores over tracked object histories. The resulting scalar danger signal is smoothed and segmented into highlight clips.

The system is designed to run in real time on CPU without a GPU, making it suitable for deployment on edge devices or as a preprocessing tool for larger pipelines.

---

## 2. Data

### 2.1 Input

The system processes any H.264/H.265 MP4 dashcam video. Development and evaluation were performed on a 10-minute near-miss compilation containing:

- Mixed urban and highway driving conditions
- Multiple near-miss events: sudden cut-ins, pedestrian crossings, hard braking, and tail-gating
- Approximate frame rate: 30 FPS, resolution: 1280 × 720

No additional training data was required. The system is detection-model-agnostic and operates on YOLO detection outputs only.

### 2.2 Object Categories

The following COCO class IDs are tracked (all other classes are filtered at the model level):

| COCO ID | Class      |
| ------- | ---------- |
| 0       | person     |
| 1       | bicycle    |
| 2       | car        |
| 3       | motorcycle |
| 5       | bus        |
| 7       | truck      |

---

## 3. Method

### 3.1 System Overview

The pipeline is a linear chain of stateless and stateful modules:

```
Video → [YOLOTracker: detection + ByteTrack + history]
      → [EgoMotionEstimator: affine camera motion]
      → [Heuristics H1–H8: per-frame sub-scores]
      → [Scorer: weighted composite + Gaussian smooth]
      → [Segmenter: threshold + gap-fill + top-K]
      → Highlight clips + JSON report
```

Every processed frame yields a per-frame composite danger score. The system operates in a single forward pass; no future frames are read-ahead.

### 3.2 Detection and Tracking

**Detector:** YOLOv8n (nano) pre-trained on COCO-2017. Inference confidence threshold: 0.35, NMS IoU threshold: 0.45. Only the six traffic-relevant classes listed above are forwarded.

**Tracker:** ByteTrack [Zhang et al., 2022] as implemented in Ultralytics `model.track()`. ByteTrack operates a two-stage association — high-confidence detections are matched first via IoU, then low-confidence detections are matched against remaining tracklets. This provides robust re-identification through partial occlusion and moderate appearance changes without appearance-feature embeddings.

**Frame stride:** Every _k_-th frame is processed (default _k_ = 2). ByteTrack is aware of the time gap; track velocity predictions internally account for skipped frames.

**Per-track history:** Each track maintains a fixed-length deque of `TrackSnapshot` objects (max 30 entries). Each snapshot records: raw bounding box, ego-compensated centroid, raw bounding box area normalised by frame area (`bbox_area_norm`), EMA-smoothed area (`bbox_area_smooth`), and the elapsed time since the previous snapshot (`dt_s`).

**EMA area smoothing:** YOLOv8's bounding box regression exhibits frame-to-frame jitter of ±5–15% of the true area even for stationary objects. To suppress this, an exponential moving average (EMA) is maintained per track:

$$\hat{A}_t = \alpha \cdot A_t + (1 - \alpha) \cdot \hat{A}_{t-1}, \quad \alpha = 0.35$$

The smoothed area $\hat{A}_t$ is used by all heuristics that operate on area growth rates (H1, H4, H8).

### 3.3 Ego-Motion Compensation

A moving dashcam introduces apparent motion of all objects in the scene even when the objects are stationary. Without compensation, this produces systematic false positives in H1 (approach rate) and H2 (lateral cut-in).

**Method:** Shi-Tomasi corner detection on background pixels (object bounding box regions masked out). Sparse Lucas-Kanade optical flow tracks these corners to the next frame. Forward-backward consistency checking rejects unreliable tracks. The surviving point correspondences are used to estimate a partial affine transform:

$$M = \begin{bmatrix} s\cos\theta & -s\sin\theta & t_x \\ s\sin\theta & s\cos\theta & t_y \end{bmatrix}$$

via RANSAC (reprojection threshold 3 px). The extracted parameters (translation $t_x, t_y$; scale $s$; rotation $\theta$) are stored in an `EgoMotion` object per frame.

**Centroid compensation:** Object centroids are warped by $M^{-1}$ before being stored in `TrackSnapshot.centroid_comp`, removing camera-induced displacement from all subsequent kinematic calculations.

**Area compensation:** Apparent area grows as $s^2$ under a zoom-in camera motion. H1 and H8 subtract the ego-implied area growth rate $\dot{A}_\text{ego} = (s^2 - 1)/\Delta t$ from the measured growth rate before scoring.

If ego-motion estimation fails (insufficient background keypoints or low RANSAC inlier fraction), the frame is marked `ego.valid = False` and all compensation defaults to identity.

### 3.4 Heuristic Scoring (H1–H8)

Each heuristic function accepts the current active track dictionary and the ego motion object and returns a scalar in $[0, 1]$. A score of 0 means the condition is absent; 1 means maximally triggered.

#### H1 — Rapid Approach

Detects objects growing rapidly in apparent size (approaching the camera). Uses the first and second halves of the track history to compute an ego-corrected area growth rate:

$$\text{rate} = \frac{\bar{A}_\text{late} - \bar{A}_\text{early}}{\Delta t_\text{late}} - \dot{A}_\text{ego}$$

$$H_1 = \text{clip}\!\left(\frac{\text{rate}}{A_\text{max}}, 0, 1\right), \quad A_\text{max} = 0.08 \text{ area/s}$$

Minimum history: 7 frames. **Adaptive area blending** (see §4.4) applies.

#### H2 — Lateral Cut-In

Detects objects with a trajectory direction angle change exceeding a threshold. Computes displacement vectors for the first and second halves of the history:

$$\Delta\theta = |\text{angle}(\mathbf{v}_\text{late}) - \text{angle}(\mathbf{v}_\text{early})|$$

Objects moving laterally at speed $> 15$ px/s with a significant direction change receive a high score. Minimum history: 7 frames.

#### H3 — Sudden Braking

Detects rapidly decelerating vehicles (indicated by a fast-decreasing ego-compensated displacement rate). Speed history is fit to detect negative acceleration peaks:

$$H_3 = \text{clip}\!\left(\frac{-\dot{v}}{\beta_\text{max}}, 0, 1\right), \quad \beta_\text{max} = 30 \text{ px/s}^2$$

Minimum history: 6 frames.

#### H4 — Close Proximity

Detects objects whose normalised bounding box area significantly exceeds a class-specific reference area corresponding to "normal" following distance:

| Class      | Reference area |
| ---------- | -------------- |
| car        | 0.060          |
| truck      | 0.090          |
| bus        | 0.120          |
| motorcycle | 0.025          |
| person     | 0.035          |
| bicycle    | 0.030          |

Top-3 objects by excess area are summed; H4 fires when the sum exceeds 0.50. Uses EMA-smoothed area. No minimum history required.

#### H5 — Scene Complexity / Motion Entropy

Tracks the number of objects and the angular entropy of velocity vectors in a normalised histogram. A z-score against a rolling 90-frame baseline detects anomalously complex scenes:

$$H_5 = \text{clip}\!\left(\frac{z - z_\text{bg}}{z_\text{max}}, 0, 1\right)$$

#### H6 — Pedestrian / Cyclist in Road Zone

Fires when persons or cyclists are detected within the central road region (configurable as a fraction of frame dimensions; default: $y \in [35\%, 85\%]$, $x \in [10\%, 90\%]$). Score is modulated by object speed — stationary pedestrians in the road zone score higher than fast-moving ones (who are more likely to be on the pavement).

#### H7 — Birth/Death Rate (Traffic Activity)

Anomalously high rates of new track creation and deletion indicate chaotic traffic. Counts stable birth and death events within a sliding window (20 frames) and compares against a background rate.

#### H8 — Time-To-Collision (TTC)

Linear regression on the EMA-smoothed area time-series over the most recent 7 snapshots:

$$\hat{A}(t) = \hat{A}_0 + \dot{A} \cdot t$$

The ego-corrected growth rate $\dot{A}_\text{rel} = \dot{A} - \dot{A}_\text{ego}$ gives an estimate of TTC:

$$\text{TTC} = \frac{\hat{A}_\text{now}}{2 \cdot \dot{A}_\text{rel}}$$

$$H_8 = \text{clip}\!\left(1 - \frac{\text{TTC}}{\text{TTC}_\text{max}}, 0, 1\right), \quad \text{TTC}_\text{max} = 2.5 \text{ s}$$

Only fires if $\dot{A}_\text{rel} > 0.008$ area/s to reject ambient jitter. Adaptive area blending applies.

### 3.5 Score Aggregation and Smoothing

The per-heuristic scores $H_i \in [0,1]$ are combined into a composite per-frame score via a weighted sum, where weights are calibrated such that $\sum w_i = 1$:

| Heuristic            | Weight |
| -------------------- | ------ |
| H1 Rapid Approach    | 0.15   |
| H2 Lane Cut-In       | 0.15   |
| H3 Sudden Braking    | 0.10   |
| H4 Close Proximity   | 0.12   |
| H5 Scene Complexity  | 0.08   |
| H6 Pedestrian Hazard | 0.12   |
| H7 Traffic Activity  | 0.05   |
| H8 Time-To-Collision | 0.15   |

_(remaining 0.08 is held as normalisation margin)_

The composite score curve is then convolved with a Gaussian kernel ($\sigma = 0.4$ s) to temporally smooth transient spikes.

### 3.6 Highlight Segmentation

The smoothed score curve is converted to binary above/below a threshold $\tau = 0.28$. Contiguous above-threshold intervals are merged if the gap between them is $\leq 1.0$ s (gap-filling), then filtered by minimum duration $\geq 1.5$ s and maximum $\leq 12$ s. The final set is ranked by peak score and the top 30 are retained.

Each resulting segment is annotated with the set of dominant heuristics — those whose mean raw score within the segment exceeds `DOMINANT_THRESHOLD = 0.35` — converted to human-readable event type labels.

---

## 4. Ablation Study

### 4.1 Tracker: Custom IoU vs. ByteTrack

**Experiment:** We compared our initial custom tracker (Hungarian algorithm on an IoU + centroid distance + class-mismatch cost matrix) against the Ultralytics ByteTrack implementation.

**Custom tracker issues observed:**

- Frequent class label switches: a car would be labelled as motorcycle for 2–5 frames when partially occluded, creating a spurious birth+death event (H7) and invalidating its trajectory history (H1, H2, H3, H8).
- ID switches during lane changes: the very event we want to detect (lateral cut-in) disrupted trajectory continuity, causing H2 misses.
- Tuning 5+ parameters (α, β, γ, cost threshold, max age) was fragile across different video conditions.

**ByteTrack result:** Eliminated class label confusion entirely (class is assigned per detection, not per track state, and ByteTrack's two-stage association keeps consistent identity through lane changes that the IoU-only matcher lost). Track history inheritance (implemented as a partial mitigation for the old tracker) became unnecessary.

**Conclusion:** ByteTrack is strictly superior for this task. The custom tracker is retained in `tracker.py` only as reference.

### 4.2 Ego-Motion Compensation

**Experiment:** H1 and H8 were evaluated with and without ego-motion subtraction on a 30-second highway clip where the dashcam itself was accelerating.

| Condition           | H1 false positives | H8 false positives |
| ------------------- | ------------------ | ------------------ |
| No ego compensation | 8                  | 6                  |
| With compensation   | 1                  | 1                  |

**Observation:** Without compensation, the apparent area growth of all objects in the scene due to the camera zooming forward (scale > 1) consistently triggered H1 and H8 on vehicles at safe following distances.

**Conclusion:** Ego-motion compensation is essential for accurate heuristic evaluation on footage with camera zoom or acceleration.

### 4.3 EMA Smoothing of Bounding Box Area

**Experiment:** Measured the coefficient of variation (CV = std/mean) of the bounding box area for a stationary parked car over 60 frames.

| Area signal     | CV (lower = smoother) |
| --------------- | --------------------- |
| Raw YOLO output | 0.087                 |
| EMA (α = 0.35)  | 0.041                 |
| EMA (α = 0.15)  | 0.021                 |

**Observation:** Raw YOLO area jitter of ~9% CV caused H1 and H8 to fire spuriously on stationary objects at α = 1.0. EMA at α = 0.35 reduced jitter sufficiently without introducing significant lag for approaching objects. α = 0.15 over-smoothed: real approach events required 15+ frames before scoring above threshold.

**Conclusion:** α = 0.35 is the best trade-off between jitter immunity and responsiveness.

### 4.4 Adaptive Area Blending for Short Tracks

**Observation:** With α = 0.35, an EMA on a brand-new track (< 6 frames) lags significantly behind the true area because the EMA has not yet converged (initial bias). A car that suddenly cut in and doubled in size over 0.3 s would score H1 = 0.00 because the EMA still reflected the initial small area.

**Fix:** A linear blend between raw area and EMA area, proportional to track age:

$$A_\text{blend}(t) = \min\!\left(\frac{n}{10}, 1\right) \cdot \hat{A}_t + \max\!\left(1 - \frac{n}{10}, 0\right) \cdot A_t$$

where $n$ is the current number of snapshots in the track history.

| Track length | EMA blend weight | Raw blend weight |
| ------------ | ---------------- | ---------------- |
| 1–4 frames   | 10–40%           | 60–90%           |
| 5–9 frames   | 50–90%           | 10–50%           |
| ≥ 10 frames  | 100%             | 0%               |

**Result:** Sudden cut-ins (0.04 → 0.085 area in 7 frames) now correctly score H1 ≈ 0.89, while jitter-only tracks (±5% random fluctuation) correctly score ≈ 0.05.

### 4.5 Minimum History Window Tuning

All heuristics requiring temporal context (H1, H2, H3, H8) have a configurable minimum history parameter. We evaluated detection rates across different values.

| Min history        | H1 recall (cut-ins) | H1 false positives            |
| ------------------ | ------------------- | ----------------------------- |
| 5 frames (~0.33s)  | High                | High (EMA lag → jitter fires) |
| 7 frames (~0.47s)  | Good                | Low                           |
| 10 frames (~0.67s) | Misses short events | Very low                      |

**Conclusion:** 7 frames provides the best recall/precision balance with the adaptive blending scheme in place. Short events (< 0.47 s) are inherently below the detection horizon, but sub-0.5 s events are rare in practice and covered by the current-frame heuristics (H4).

---

## 5. Results

The system was evaluated qualitatively on a 10-minute near-miss dashcam compilation.

**Segment extraction:**

- **6 highlights** detected at default threshold 0.28
- **3 highlights** detected at tighter threshold 0.35
- Median highlight duration: 3.2 s

**Event type distribution (default threshold):**

| Event Type                       | Occurrences |
| -------------------------------- | ----------- |
| Close Proximity (H4)             | 5           |
| Collision Risk (H8)              | 4           |
| Pedestrian / Cyclist Hazard (H6) | 3           |
| Sudden Approach (H1)             | 2           |
| Lane Cut-In (H2)                 | 1           |
| Sudden Braking (H3)              | 1           |

**Qualitative observations:**

- All visually obvious near-miss events (sudden cut-ins, hard braking, pedestrian crossings) were captured.
- H4 (Close Proximity) was the most reliable heuristic — its independence from track history makes it robust to short-lived objects.
- H7 (Traffic Activity) contributed marginally and inflated scores during normal busy intersections; its weight (0.05) appropriately limits its impact.
- Processing speed: ~1.5× realtime on Intel Core i7 (no GPU), `frame_stride = 2`.

---

## 6. Limitations

1. **No ground-truth evaluation.** Without a labelled dataset of dangerous events in the test video, precision and recall cannot be quantified. Only qualitative review was performed.

2. **Fixed road zone for H6.** The pedestrian road zone is defined as a fixed frame-fraction rectangle. This breaks on curved roads, intersections, and non-forward-facing cameras. A lane-detection front-end would generalise this.

3. **Monocular depth ambiguity.** Bounding box area is used as a proxy for distance. This is only valid for objects of known physical size approaching head-on. Small distant trucks appear smaller than large nearby motorcycles; class-specific reference areas (H4) partially mitigate this but do not eliminate the ambiguity.

4. **Ego-motion model.** The sparse affine model assumes a rigid scene. Dense traffic (many foreground objects masking background) degrades optical flow quality, causing ego-motion to be marked invalid more frequently. A gyroscope-based IMU would be more reliable.

5. **Short-lived objects.** Objects visible for fewer than 7 frames cannot trigger H1, H2, or H8. This is an inherent limitation of trajectory-based heuristics.

6. **Fixed weights.** The eight heuristic weights were hand-tuned on a single video. A labelled corpus would allow data-driven weight calibration (e.g., logistic regression on composite scores against annotated events).

7. **Night and adverse weather.** YOLOv8n detection performance degrades significantly in low-light or rain conditions, which can silence all heuristics simultaneously.

---

## 7. References

1. **Jocher, G. et al.** (2023). _Ultralytics YOLOv8._ https://github.com/ultralytics/ultralytics

2. **Zhang, Y. et al.** (2022). _ByteTrack: Multi-Object Tracking by Associating Every Detection Box._ ECCV 2022. arXiv:2110.06864

3. **Lucas, B. D. & Kanade, T.** (1981). _An Iterative Image Registration Technique with an Application to Stereo Vision._ IJCAI.

4. **Shi, J. & Tomasi, C.** (1994). _Good Features to Track._ CVPR 1994.

5. **Fischler, M. A. & Bolles, R. C.** (1981). _Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography._ Commun. ACM, 24(6), 381–395.

6. **Lin, T.-Y. et al.** (2014). _Microsoft COCO: Common Objects in Context._ ECCV 2014. arXiv:1405.0312

7. **Geiger, A. et al.** (2012). _Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite._ CVPR 2012. _(Reference for dashcam evaluation methodology.)_
