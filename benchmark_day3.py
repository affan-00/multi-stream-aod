#!/usr/bin/env python3
"""
Controlled same-hardware CPU throughput & latency benchmark (Protocol b: Table 4a).
Compares Proposed (event-gated) against continuous per-frame MobileNet-SSD, YOLOv3-tiny,
ungated dual-head, and YOLOv8n on identical preloaded frames (400 single-stream / 4x200 concurrent).

Note: All slices start at frame 1 (steady-state non-alert frames). Full-sequence neural trigger cost
is measured separately by profile_hardware_scaling.py (Table 5).
"""
import argparse
import os
import threading
import time

import cv2
import numpy as np

from abandoned_object_detector import AbandonedObjectDetector
from detector import Detector


def load_frames(video_path: str, n: int):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < n:
        ret, f = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, f = cap.read()
            if not ret:
                break
        frames.append(f)
    cap.release()
    return frames


def make_runner(kind: str):
    if kind == "proposed":
        det = AbandonedObjectDetector("bench", background_method="edge_adaptive")
        return lambda f: det.process(f)

    if kind == "mobilenet_cont":
        net = cv2.dnn.readNetFromCaffe(
            "mobilenet_data/MobileNetSSD_deploy.prototxt.txt",
            "mobilenet_data/MobileNetSSD_deploy.caffemodel",
        )
        def _run(f):
            blob = cv2.dnn.blobFromImage(cv2.resize(f, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            net.forward()
        return _run

    if kind == "yolov3_tiny_cont":
        y = Detector("yolo", 0.3, 0.1)
        return lambda f: y.detect(f)

    if kind == "dual_ungated":
        net = cv2.dnn.readNetFromCaffe(
            "mobilenet_data/MobileNetSSD_deploy.prototxt.txt",
            "mobilenet_data/MobileNetSSD_deploy.caffemodel",
        )
        y = Detector("yolo", 0.3, 0.1)
        def _run(f):
            blob = cv2.dnn.blobFromImage(cv2.resize(f, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            net.forward()
            y.detect(f)
        return _run

    if kind == "yolov8n_cont":
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        return lambda f: model.predict(f, device="cpu", imgsz=416, verbose=False)

    raise ValueError(kind)


def bench_method(kind: str, frames_1: list, frames_4: list):
    fn1 = make_runner(kind)
    for f in frames_1[:5]:
        fn1(f)
    lat_ms = []
    t0 = time.perf_counter()
    for f in frames_1:
        s = time.perf_counter()
        fn1(f)
        lat_ms.append((time.perf_counter() - s) * 1000.0)
    wall_1 = time.perf_counter() - t0
    fps_1 = len(frames_1) / wall_1 if wall_1 > 0 else 0.0

    runners = [make_runner(kind) for _ in range(4)]
    barrier = threading.Barrier(4)

    def worker(idx: int):
        r = runners[idx]
        seq = frames_4[idx]
        barrier.wait()
        for f in seq:
            r(f)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_4 = time.perf_counter() - t0
    total_4 = sum(len(s) for s in frames_4)
    agg_fps_4 = total_4 / wall_4 if wall_4 > 0 else 0.0

    arr = np.array(lat_ms, dtype=np.float64)
    q25, med, q75, p95 = np.percentile(arr, [25, 50, 75, 95])
    return {
        "fps_1": fps_1,
        "agg_fps_4": agg_fps_4,
        "per_stream_4": agg_fps_4 / 4.0,
        "med_ms": float(med),
        "iqr_ms": float(q75 - q25),
        "p95_ms": float(p95),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./ABODA")
    ap.add_argument("--runs", type=int, default=1, help="Number of repeated runs to execute.")
    args = ap.parse_args()

    print("Loading preloaded benchmark frames into RAM...")
    frames_1 = load_frames(os.path.join(args.dir, "video1.avi"), 400)
    pool = ["video1.avi", "video3.avi", "video9.avi", "video10.avi"]
    frames_4 = [load_frames(os.path.join(args.dir, v), 200) for v in pool]

    methods = [
        ("Proposed (Event-Gated)", "proposed"),
        ("Continuous MobileNet-SSD", "mobilenet_cont"),
        ("Continuous YOLOv3-tiny", "yolov3_tiny_cont"),
        ("Ungated MNet + YOLOv3-tiny", "dual_ungated"),
    ]
    try:
        import ultralytics  # noqa: F401
        methods.append(("Continuous YOLOv8n (416 px)", "yolov8n_cont"))
    except Exception:
        print("[Note] ultralytics not installed; skipping YOLOv8n.")

    for run_idx in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"\n--- Run {run_idx}/{args.runs} ---")
        results = {}
        for title, key in methods:
            print(f"Benchmarking: {title} ...")
            results[key] = bench_method(key, frames_1, frames_4)

        prop = results["proposed"]
        print("\n=== Controlled Same-Hardware CPU Benchmark (400 single-stream / 4x200 concurrent) ===")
        print("| Method | 1-Stream FPS | 4-Stream Agg FPS | Per-Stream (4x) | Median (ms) | IQR (ms) | P95 (ms) | 4x Speedup |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for title, key in methods:
            r = results[key]
            sp = "Reference" if key == "proposed" else f"{prop['agg_fps_4'] / r['agg_fps_4']:.1f}x"
            print(
                f"| {title} | {r['fps_1']:.1f} | {r['agg_fps_4']:.1f} | {r['per_stream_4']:.1f} | "
                f"{r['med_ms']:.2f} | {r['iqr_ms']:.2f} | {r['p95_ms']:.2f} | {sp} |"
            )


if __name__ == "__main__":
    main()