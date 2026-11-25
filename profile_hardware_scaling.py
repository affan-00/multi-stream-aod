#!/usr/bin/env python3
"""
Stage-wise latency profiler (5 ABODA sequences) and fresh-process 1-4 stream concurrency benchmark.
Uses AbandonedObjectDetector(background_method="edge_adaptive") directly (no eval_aboda_v4 wrapper).
"""
import argparse
import os
import platform
import resource
import subprocess
import sys
import threading
import time

import cv2

from abandoned_object_detector import AbandonedObjectDetector


def get_system_specs() -> dict:
    cpu = platform.processor() or "Unknown"
    ram_gb = 0.0
    if sys.platform == "darwin":
        try:
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
            ram_gb = mem_bytes / (1024 ** 3)
        except Exception:
            pass
    return {
        "cpu": cpu,
        "cores": os.cpu_count() or 1,
        "ram_gb": ram_gb,
        "os": f"{platform.system()} {platform.mac_ver()[0] if sys.platform == 'darwin' else platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "opencv": cv2.__version__,
    }


def get_rss_mb() -> float:
    """Current resident set size (MB) of this process."""
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()
        return int(out) / 1024.0
    except Exception:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024.0


def profile_stages(video_path: str):
    cap = cv2.VideoCapture(video_path)
    det = AbandonedObjectDetector("stage_prof", background_method="edge_adaptive")

    t_seg = t_mnet = t_yolo_veto = t_yolo_owner = 0.0
    n_frames = n_mnet = n_yolo_veto = n_yolo_owner = 0

    orig_mnet = det._is_mobilenet_person
    orig_yolo_veto = det._is_covered_by_yolo_person
    orig_owner = det._find_owner
    orig_seg = det._segment

    def timed_mnet(crop):
        nonlocal t_mnet, n_mnet
        t0 = time.perf_counter()
        res = orig_mnet(crop)
        t_mnet += time.perf_counter() - t0
        n_mnet += 1
        return res

    def timed_yolo_veto(frame, dim):
        nonlocal t_yolo_veto, n_yolo_veto
        cached = (det._yolo_cache_frame == det.frame_no)
        t0 = time.perf_counter()
        res = orig_yolo_veto(frame, dim)
        dt = time.perf_counter() - t0
        if not cached:
            t_yolo_veto += dt
            n_yolo_veto += 1
        return res

    def timed_owner(obj):
        nonlocal t_yolo_owner, n_yolo_owner
        t0 = time.perf_counter()
        res = orig_owner(obj)
        t_yolo_owner += time.perf_counter() - t0
        n_yolo_owner += 1
        return res

    def timed_seg(frame):
        nonlocal t_seg
        t0 = time.perf_counter()
        res = orig_seg(frame)
        t_seg += time.perf_counter() - t0
        return res

    det._segment = timed_seg
    det._is_mobilenet_person = timed_mnet
    det._is_covered_by_yolo_person = timed_yolo_veto
    det._find_owner = timed_owner

    t_total = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n_frames += 1
        t0 = time.perf_counter()
        det.process(frame)
        t_total += time.perf_counter() - t0
    cap.release()

    t_track = max(0.0, t_total - (t_seg + t_mnet + t_yolo_veto + t_yolo_owner))
    return {
        "frames": n_frames,
        "seg_ms": (t_seg / n_frames) * 1000,
        "track_ms": (t_track / n_frames) * 1000,
        "mnet_calls": n_mnet,
        "mnet_per_call_ms": (t_mnet / n_mnet) * 1000 if n_mnet else 0.0,
        "mnet_amort_ms": (t_mnet / n_frames) * 1000,
        "veto_calls": n_yolo_veto,
        "veto_per_call_ms": (t_yolo_veto / n_yolo_veto) * 1000 if n_yolo_veto else 0.0,
        "veto_amort_ms": (t_yolo_veto / n_frames) * 1000,
        "owner_calls": n_yolo_owner,
        "owner_per_call_ms": (t_yolo_owner / n_yolo_owner) * 1000 if n_yolo_owner else 0.0,
        "owner_amort_ms": (t_yolo_owner / n_frames) * 1000,
        "total_ms": (t_total / n_frames) * 1000,
        "fps": n_frames / t_total if t_total > 0 else 0.0,
    }


def benchmark_concurrency(stream_paths: list, max_frames: int = 1200):
    detectors = [AbandonedObjectDetector(f"CAM_{i+1}", background_method="edge_adaptive") for i in range(len(stream_paths))]
    caps = [cv2.VideoCapture(p) for p in stream_paths]
    counts = [0] * len(stream_paths)
    barrier = threading.Barrier(len(stream_paths))

    def worker(idx: int):
        cap = caps[idx]
        det = detectors[idx]
        barrier.wait()
        n = 0
        while n < max_frames:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            det.process(frame)
            n += 1
        counts[idx] = n

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(stream_paths))]
    cpu0 = os.times()
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    cpu1 = os.times()

    rss_mb = get_rss_mb()
    for c in caps:
        c.release()

    total_frames = sum(counts)
    agg_fps = total_frames / wall if wall > 0 else 0.0
    per_stream_fps = agg_fps / len(stream_paths)
    cpu_time = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
    cores = os.cpu_count() or 1
    return {
        "streams": len(stream_paths),
        "total_frames": total_frames,
        "wall_s": wall,
        "agg_fps": agg_fps,
        "per_stream_fps": per_stream_fps,
        "rss_mb": rss_mb,
        "cpu_pct": (cpu_time / wall) * 100.0 / cores if wall > 0 else 0.0,
        "cores_used": cpu_time / wall if wall > 0 else 0.0,
    }


def print_stage_table(name, st):
    print(f"\n=== Per-Stage Latency Breakdown ({name}, {st['frames']} frames) ===")
    print("| Pipeline Stage | Invocations | Latency / Call (ms) | Amortized / Frame (ms) | Share (%) |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    print(f"| 1. Illum.-Normalized Edge Segmentation | {st['frames']} | {st['seg_ms']:.3f} | {st['seg_ms']:.3f} | {100*st['seg_ms']/st['total_ms']:.1f}% |")
    print(f"| 2. Contour Extraction & Drift-Bounded Tracking | {st['frames']} | {st['track_ms']:.3f} | {st['track_ms']:.3f} | {100*st['track_ms']/st['total_ms']:.1f}% |")
    print(f"| 3. MobileNet-SSD Static Verification (Head 1) | {st['mnet_calls']} | {st['mnet_per_call_ms']:.2f} | {st['mnet_amort_ms']:.4f} | {100*st['mnet_amort_ms']/st['total_ms']:.2f}% |")
    print(f"| 4. YOLOv3-tiny Person Veto (Head 2a) | {st['veto_calls']} | {st['veto_per_call_ms']:.2f} | {st['veto_amort_ms']:.4f} | {100*st['veto_amort_ms']/st['total_ms']:.2f}% |")
    print(f"| 5. YOLOv3-tiny Multi-Probe Owner Lookup (Head 2b) | {st['owner_calls']} | {st['owner_per_call_ms']:.2f} | {st['owner_amort_ms']:.4f} | {100*st['owner_amort_ms']/st['total_ms']:.2f}% |")
    print(f"| **Total End-to-End** | **{st['frames']}** | — | **{st['total_ms']:.3f} ms ({st['fps']:.1f} FPS)** | **100.0%** |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=["video1", "video3", "video9", "video10", "video11"])
    ap.add_argument("--streams", type=int, default=0, help="Run a single stream count (1-4) in a fresh process for clean RSS.")
    ap.add_argument("--skip-stages", action="store_true")
    ap.add_argument("--skip-concurrency", action="store_true")
    args = ap.parse_args()

    specs = get_system_specs()
    print("=== Testbed Hardware & Software Specifications ===")
    print(f"- CPU: {specs['cpu']} ({specs['cores']} logical cores)")
    print(f"- RAM: {specs['ram_gb']:.1f} GB")
    print(f"- OS: {specs['os']}")
    print(f"- Runtime: Python {specs['python']} | OpenCV {specs['opencv']} (CPU-only, no GPU)")

    if not args.skip_stages:
        for v in args.videos:
            print_stage_table(v, profile_stages(f"./ABODA/{v}.avi"))

    if args.skip_concurrency:
        return
    pool = ["./ABODA/video1.avi", "./ABODA/video3.avi", "./ABODA/video9.avi", "./ABODA/video10.avi"]
    ks = [args.streams] if args.streams else [1, 2, 3, 4]
    print("\n=== Multi-Stream Concurrent Scaling (1,200 frames/stream; wall time INCLUDES video decoding) ===")
    print("| Concurrent Streams | Total Frames | Wall Time (s) | Aggregate FPS | Per-Stream FPS | RSS at end of run (MB) | System CPU (%) | Active Cores |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for k in ks:
        r = benchmark_concurrency(pool[:k], max_frames=1200)
        print(
            f"| {r['streams']} | {r['total_frames']} | {r['wall_s']:.2f} | {r['agg_fps']:.1f} | "
            f"{r['per_stream_fps']:.1f} | {r['rss_mb']:.1f} | {r['cpu_pct']:.1f}% | {r['cores_used']:.2f} |"
        )


if __name__ == "__main__":
    main()