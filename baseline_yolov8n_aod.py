#!/usr/bin/env python3
"""
Equal-task baseline: CONTINUOUS YOLOv8n (every frame) + the same abandonment decision rule as the proposed system.

Front end : YOLOv8n on every frame (CPU), bag classes only (backpack, handbag, suitcase) + person.
Decision  : identical to the proposed pipeline's back end:
            static >= dwell frames, drift <= 40 px, person-cover veto, 2 clean checks (every 20 frames),
            <= 2 vetoes, 45 px duplicate suppression around confirmed alerts.
Scoring   : score_aboda_auto.match_events (same GT, same tolerances, same Wilson CIs).

Usage:  python baseline_yolov8n_aod.py --dir ABODA --gt gt_aboda.json --weights yolov8n.pt
Requires: pip install ultralytics   (inside the same 'aod' env)
"""
import argparse
import glob
import json
import os
import time

import cv2

from score_aboda_auto import match_events, prf, wilson, natural_key

BAG_CLASSES = [24, 26, 28]  # COCO: backpack, handbag, suitcase
PERSON = 0


def iou(a, b):
    xA, yA, xB, yB = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    aa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    bb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return inter / (aa + bb - inter) if (aa + bb - inter) > 0 else 0.0


def containment(inner, outer):
    xA, yA, xB, yB = max(inner[0], outer[0]), max(inner[1], outer[1]), min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    area = (inner[2] - inner[0] + 1) * (inner[3] - inner[1] + 1)
    return inter / area if area > 0 else 0.0


class Track:
    def __init__(self, idx, box, frame_no):
        self.idx, self.box, self.first = idx, box, frame_no
        self.first_center = self._c(box)
        self.last, self.hits = frame_no, 1
        self.clean, self.vetoes, self.last_check = 0, 0, -10_000
        self.alerted = False

    @staticmethod
    def _c(b):
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def drift(self):
        c = self._c(self.box)
        return ((c[0] - self.first_center[0]) ** 2 + (c[1] - self.first_center[1]) ** 2) ** 0.5


class BaselineAOD:
    def __init__(self, dwell=146, drift=40.0, max_gap=30, check_every=20, dup_dist=45.0, match_iou=0.3):
        self.dwell, self.drift, self.max_gap = dwell, drift, max_gap
        self.check_every, self.dup_dist, self.match_iou = check_every, dup_dist, match_iou
        self.tracks, self.alerts, self.next_idx, self.frame_no = [], [], 0, 0

    def step(self, bags, persons):
        """bags / persons: lists of [x1, y1, x2, y2]. Returns list of new alert dicts."""
        self.frame_no += 1
        new = []
        for box in bags:
            c = Track._c(box)
            if any(((c[0] - Track._c(a["bbox"])[0]) ** 2 + (c[1] - Track._c(a["bbox"])[1]) ** 2) ** 0.5 < self.dup_dist
                   or iou(box, a["bbox"]) > 0.15 for a in self.alerts):
                continue
            best, best_s = None, self.match_iou
            for t in self.tracks:
                if t.alerted:
                    continue
                s = iou(box, t.box)
                if s > best_s:
                    best, best_s = t, s
            if best is None:
                self.tracks.append(Track(self.next_idx, box, self.frame_no))
                self.next_idx += 1
                continue
            best.box, best.last = box, self.frame_no
            best.hits += 1
            if best.drift() > self.drift:  # moved: restart dwell clock
                best.first, best.first_center, best.hits = self.frame_no, Track._c(box), 1
                best.clean = best.vetoes = 0
                continue
            span = self.frame_no - best.first
            if span >= self.dwell and best.hits >= 0.5 * span and \
                    self.frame_no - best.last_check >= self.check_every:
                best.last_check = self.frame_no
                covered = any(containment(box, p) > 0.40 or iou(box, p) > 0.25 for p in persons)
                if covered:
                    best.vetoes += 1
                    best.clean = 0
                else:
                    best.clean += 1
                if best.clean >= 2 and best.vetoes <= 2:
                    best.alerted = True
                    ev = {"frame": self.frame_no, "bbox": [int(v) for v in box], "owner": False}
                    self.alerts.append(ev)
                    new.append(ev)
        self.tracks = [t for t in self.tracks if t.alerted or self.frame_no - t.last <= self.max_gap]
        return new


def yolo_detect_fn(weights, imgsz, conf, any_non_person=False):
    from ultralytics import YOLO
    model = YOLO(weights)

    def detect(frame):
        kw = {} if any_non_person else {"classes": [PERSON] + BAG_CLASSES}
        r = model.predict(frame, imgsz=imgsz, conf=conf, device="cpu", verbose=False, **kw)[0]
        bags, persons = [], []
        for b_, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
            box = [int(v) for v in b_]
            (persons if int(c) == PERSON else bags).append(box)
        return bags, persons
    return detect


def run_video(vp, detect, cache_dir):
    stem = os.path.splitext(os.path.basename(vp))[0].lower()
    path = os.path.join(cache_dir, f"{stem}.json")
    if os.path.exists(path):
        return json.load(open(path))
    cap = cv2.VideoCapture(vp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    algo, events, frames, comp = BaselineAOD(), [], 0, 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        t0 = time.perf_counter()
        bags, persons = detect(frame)
        events += algo.step(bags, persons)
        comp += time.perf_counter() - t0
    cap.release()
    run = {"video": stem, "fps": fps, "frames": frames, "compute_s": comp, "events": events}
    os.makedirs(cache_dir, exist_ok=True)
    json.dump(run, open(path, "w"))
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./ABODA")
    ap.add_argument("--gt", default="gt_aboda.json")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--cache-dir", default="./baseline_cache")
    ap.add_argument("--pad", type=int, default=30)
    ap.add_argument("--max-delay-s", type=float, default=60.0)
    ap.add_argument("--early-tol-s", type=float, default=7.0)
    ap.add_argument("--any-non-person", action="store_true",
                    help="treat EVERY non-person COCO detection as a candidate object (instead of bag classes only)")
    args = ap.parse_args()

    gt = json.load(open(args.gt))
    videos = sorted(glob.glob(os.path.join(args.dir, "*.avi")), key=natural_key)
    detect = yolo_detect_fn(args.weights, args.imgsz, args.conf, args.any_non_person)
    TP = FP = FN = frames = 0
    comp, delays = 0.0, []
    print("| Video | GT | TP | FP | FN | Delay (s) | FPS |\n| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for vp in videos:
        stem = os.path.splitext(os.path.basename(vp))[0].lower()
        if stem not in gt:
            continue
        run = run_video(vp, detect, args.cache_dir)
        tps, fp, fn = match_events(gt[stem], run["events"], run["fps"], args.pad, args.max_delay_s,
                                   early_tol=int(args.early_tol_s * run["fps"]))
        d = [(ev["frame"] - gt[stem][gi]["frame"]) / run["fps"] for gi, ev in tps]
        delays += d
        TP += len(tps); FP += len(fp); FN += len(fn)
        frames += run["frames"]; comp += run["compute_s"]
        ds = f"{sum(d) / len(d):.1f}" if d else "-"
        print(f"| {stem} | {len(gt[stem])} | {len(tps)} | {len(fp)} | {len(fn)} | {ds} | {run['frames'] / run['compute_s']:.1f} |")
    p, r, f1 = prf(TP, FP, FN)
    plo, phi = wilson(TP, TP + FP)
    rlo, rhi = wilson(TP, TP + FN)
    md = sum(delays) / len(delays) if delays else float("nan")
    print("\n| Configuration | TP | FP | FN | Precision [95% CI] | Recall [95% CI] | F1 | Mean delay (s) | FPS |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    print(f"| Continuous YOLOv8n {'any-non-person' if args.any_non_person else 'bag-only'} + same decision rule ({args.imgsz}px, conf {args.conf}) | {TP} | {FP} | {FN} | "
          f"{p:.3f} [{plo:.2f},{phi:.2f}] | {r:.3f} [{rlo:.2f},{rhi:.2f}] | {f1:.3f} | {md:.1f} | {frames / comp:.1f} |")


if __name__ == "__main__":
    main()
