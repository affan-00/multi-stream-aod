# Event-Gated Multi-Stream Abandoned Object Detection (CPU-Only)

Official implementation, automated evaluation harness, ground-truth annotations, and equal-task continuous YOLOv8n baselines for:

> **Event-Gated Dual-Head Pipeline with Illumination-Normalized Edge Segmentation for Multi-Stream Abandoned Object Detection on Commodity CPUs**  
> Mohd Affan, Amit Seth — Department of Computer Science & Engineering, Sharda University

## Overview

This repository provides a GPU-free, thread-isolated multi-stream Abandoned Object Detection (AOD) pipeline engineered for commodity multi-core CPUs. Each camera stream runs an independent three-stage engine:

1. **Illumination-Normalized Novel-Edge Segmentation:** Compensates gradual luminance drift, masks permanent architectural edges, resets automatically on abrupt global brightness shifts, and applies a 15-frame startup warm-up.
2. **Admissibility Filtering, Bipartite IoU Tracking & Drift-Bounded Static Gating:** Tracks stationary candidates using joint contour/bounding-box area bounds, 45 px duplicate suppression, IoU association ($\tau_{\text{IoU}} = 0.40$), and a locked centroid drift bound ($\delta_{\max} = 40.0\text{ px}$, $\theta_s = 146\text{ frames}$).
3. **Event-Gated Verification, Multi-Probe Owner Lookup & Alert Persistence:** Invokes MobileNet-SSD and cached full-frame YOLOv3-tiny (RGB) strictly on stationary candidates (consuming 2.5–3.5% of total processing time across five profiled ABODA sequences), performs multi-probe retrospective owner lookup over a stride-5 circular buffer (200 slots = 1,000 frames), and retains confirmed alerts across transient occlusions.

> **Note:** Streams are processed independently in isolated threads without cross-camera association or re-identification.

## Key Results (Apple M4 CPU, 10 Logical Cores, 16 GB RAM, No GPU)

Evaluated automatically on all 11 public ABODA sequences (37,653 frames, 12 annotated abandonment events) under RGB input (`YOLO_SWAPRB=1`):

| System (Equal-Task ABODA Evaluation) | TP | FP | FN | Precision [95% CI] | Recall [95% CI] | F1 | Compute FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed (Event-Gated, RGB)** | **7** | **4** | **5** | **0.636 [0.35, 0.85]** | **0.583 [0.32, 0.81]** | **0.609** | **407.4** |
| YOLOv8n bag-only + rule (416 px, conf 0.25) | 1 | 0 | 11 | 1.000 [0.21, 1.00] | 0.083 [0.01, 0.35] | 0.154 | 74.4 |
| YOLOv8n bag-only + rule (640 px, conf 0.10) | 3 | 2 | 9 | 0.600 [0.23, 0.88] | 0.250 [0.09, 0.53] | 0.353 | 37.0 |
| YOLOv8n any-non-person + rule (416 px, conf 0.25) | 2 | 37 | 10 | 0.051 [0.01, 0.17] | 0.167 [0.05, 0.45] | 0.078 | 73.1 |
| YOLOv8n any-non-person + rule (640 px, conf 0.10) | 2 | 64 | 10 | 0.030 [0.01, 0.10] | 0.167 [0.05, 0.45] | 0.051 | 35.0 |

* **Leave-One-Video-Out (LOVO) Validation:** F1 = **0.609** (identical to fixed defaults $\delta_{\max}=40\text{ px}, \theta_s=146\text{ f}$).
* **Strict 15-frame (0.5 s) Early Tolerance:** 6 TP, 5 FP, 6 FN (P = 0.545 [0.28, 0.79], R = 0.500 [0.25, 0.75], F1 = 0.522).
* **Same-Hardware Preloaded Throughput (5-run median):** **268.1 FPS** (1 stream) and **588.1 FPS aggregate** (4 streams, 147.0 FPS/stream) — **4.2–10.9× faster** than continuous MobileNet-SSD, YOLOv8n, YOLOv3-tiny, and ungated dual-head baselines.
* **Fresh-Process Multi-Stream Scaling (incl. decoding):** **765.1 aggregate FPS** across 4 threads (**191.3 FPS/stream**, 1,525 MB RSS, 64.0% CPU).

## Repository Structure

```text
.
├── abandoned_object_detector.py   # Core edge_adaptive AOD engine
├── detector.py                    # YOLOv3-tiny OpenCV DNN wrapper (YOLO_SWAPRB switch)
├── multi_camera_aod.py            # Multi-threaded 2x2 mosaic & FIFO event logger
├── score_aboda_auto.py            # Automated ABODA scoring, 9-row ablation & LOVO sweep
├── baseline_yolov8n_aod.py        # Equal-task continuous YOLOv8n + identical decision rule
├── diagnose_baseline.py           # 80-class COCO visibility diagnostic on ABODA events
├── profile_hardware_scaling.py    # 5-sequence stage latency & fresh-process 1–4 stream scaling
├── benchmark_day3.py              # Same-hardware CPU throughput benchmark (preloaded frames)
├── gt_aboda.json                  # Ground-truth departure frames & bounding boxes (11 videos, 12 events)
├── results/                       # Saved output logs matching all manuscript tables
├── yolo/                          # yolov3-tiny.cfg, coco.names (place yolov3-tiny.weights here)
├── mobilenet_data/                # MobileNetSSD_deploy.prototxt.txt (place .caffemodel here)
└── ABODA/                         # Place public ABODA video1.avi .. video11.avi here (not tracked)s