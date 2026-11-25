#!/usr/bin/env python3
"""
Why does the continuous YOLOv8n baseline miss ABODA objects?
For each annotated object, sample frames in the N seconds after the GT frame, run YOLOv8n (all 80 classes,
low confidence) at several input sizes, and report:
  * fraction of sampled frames with ANY detection overlapping the object,
  * fraction with a BAG-class detection (backpack/handbag/suitcase) overlapping it,
  * the class names YOLO assigns (with best confidence).

Usage: python diagnose_baseline.py --dir ABODA --gt gt_aboda.json --weights yolov8n.pt
"""
import argparse
import json
import os
from collections import defaultdict

import cv2

BAG = {"backpack", "handbag", "suitcase"}


def iou(a, b):
    xA, yA, xB, yB = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (aa + bb - inter) if (aa + bb - inter) > 0 else 0.0


def overlaps(det, gt, pad=15):
    cx, cy = (det[0] + det[2]) / 2, (det[1] + det[3]) / 2
    inside = gt[0] - pad <= cx <= gt[2] + pad and gt[1] - pad <= cy <= gt[3] + pad
    return inside or iou(det, gt) > 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./ABODA")
    ap.add_argument("--gt", default="gt_aboda.json")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--sizes", type=int, nargs="+", default=[416, 640])
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--step", type=int, default=5, help="sample every N frames")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    names = model.names
    gt = json.load(open(args.gt))

    for stem in sorted(gt, key=lambda s: int(s.replace("video", ""))):
        vp = os.path.join(args.dir, stem + ".avi")
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for gi, g in enumerate(gt[stem]):
            frames = [f for f in range(g["frame"], min(total, g["frame"] + int(args.seconds * fps)), args.step)]
            for size in args.sizes:
                n_any = n_bag = n = 0
                classes = defaultdict(lambda: [0, 0.0])
                for f in frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
                    ok, img = cap.read()
                    if not ok:
                        continue
                    n += 1
                    r = model.predict(img, imgsz=size, conf=args.conf, device="cpu", verbose=False)[0]
                    hit_any = hit_bag = False
                    for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                        if overlaps(b, g["bbox"]):
                            name = names[int(c)]
                            hit_any = True
                            hit_bag = hit_bag or name in BAG
                            classes[name][0] += 1
                            classes[name][1] = max(classes[name][1], p)
                    n_any += hit_any
                    n_bag += hit_bag
                top = sorted(classes.items(), key=lambda kv: -kv[1][0])[:3]
                desc = ", ".join(f"{k} x{v[0]} (max {v[1]:.2f})" for k, v in top) or "none"
                print(f"{stem} GT{gi} @{size}px: any-class {100 * n_any / max(n, 1):3.0f}% | "
                      f"bag-class {100 * n_bag / max(n, 1):3.0f}% | seen as: {desc}")
        cap.release()


if __name__ == "__main__":
    main()