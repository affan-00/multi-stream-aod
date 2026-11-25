#!/usr/bin/env python3
"""
Automatic, reproducible ABODA scoring (no manual T/F tagging).

Modes
  annotate  one-time helper to create gt_aboda.json (frame + bbox of each abandoned object)
  score     score the full pipeline with default parameters
  ablate    ablation ladder + leave-one-component-out variants
  sweep     (delta_max x dwell) grid + leave-one-video-out (LOVO) selection

GT file format (gt_aboda.json), frames are 1-based like the detector's frame_no:
  {"video1": [{"frame": 612, "bbox": [x1, y1, x2, y2]}], "video6": [{...}, {...}], ...}
  "frame" = first frame at which the object is clearly left alone. Define this once, state it in the paper.

Matching rule (stated so it can be copied into the paper):
  An alert matches an unmatched GT event if
    GT.frame - early_tol <= alert.frame <= GT.frame + max_delay_s * fps
    AND (IoU(alert bbox, GT bbox) > 0.05 OR alert centre lies inside GT bbox inflated by `pad` px).
  Each GT event is matched at most once. Every unmatched alert (including duplicates) is a FP.
  Every unmatched GT event is a FN.
"""
import argparse
import glob
import json
import math
import os
import re
import time

import cv2


# ------------------------------------------------------------------ scoring
def natural_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", os.path.basename(path))]


def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    aa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    bb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return inter / (aa + bb - inter) if (aa + bb - inter) > 0 else 0.0


def spatial_ok(box, gt_box, pad):
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    inside = (gt_box[0] - pad <= cx <= gt_box[2] + pad) and (gt_box[1] - pad <= cy <= gt_box[3] + pad)
    return inside or iou(box, gt_box) > 0.05


def match_events(gt_list, events, fps, pad=30, max_delay_s=60.0, early_tol=15):
    used, tps, fp = set(), [], []
    for ev in sorted(events, key=lambda e: e["frame"]):
        hit = None
        for gi, g in enumerate(gt_list):
            if gi in used:
                continue
            if g["frame"] - early_tol <= ev["frame"] <= g["frame"] + max_delay_s * fps and \
               spatial_ok(ev["bbox"], g["bbox"], pad):
                hit = gi
                break
        if hit is None:
            fp.append(ev)
        else:
            used.add(hit)
            tps.append((hit, ev))
    fn = [gi for gi in range(len(gt_list)) if gi not in used]
    return tps, fp, fn


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def alert_coverage(run, gt_list, tps, pad):
    """Fraction of sampled frames, from first matched alert to end of video, in which an abandoned-flagged
    track still overlaps the GT object. Measures persistence (alert does not silently vanish)."""
    vals = []
    for gi, ev in tps:
        g = gt_list[gi]
        frames = [int(f) for f in run["abandoned_by_frame"] if int(f) >= ev["frame"]]
        total = [f for f in range(ev["frame"], run["frames"] + 1) if f % 5 == 0]
        if not total:
            continue
        covered = 0
        for f in total:
            boxes = run["abandoned_by_frame"].get(str(f), [])
            if any(spatial_ok(b, g["bbox"], pad) for b in boxes):
                covered += 1
        vals.append(covered / len(total))
    return vals


# ------------------------------------------------------------------ detector variants
NO_LIMIT = 1e9


def make_detector(variant_cfg, stem, drift, dwell):
    """variant_cfg = (veto, persist). Uses the shipped AbandonedObjectDetector (edge_adaptive) directly;
    it is functionally identical to eval_aboda_v4.UpgradedAODv4, so that wrapper is no longer needed."""
    from abandoned_object_detector import AbandonedObjectDetector
    veto, persist, mnet = variant_cfg

    class Variant(AbandonedObjectDetector):
        def _is_mobilenet_person(self, crop):
            return super()._is_mobilenet_person(crop) if mnet else False

        def _is_covered_by_yolo_person(self, frame, dim):
            return super()._is_covered_by_yolo_person(frame, dim) if veto else False

        def process(self, frame):
            out = super().process(frame)
            if not persist:  # classical stale pruning also applies to confirmed alerts
                self.tracked_objects = [
                    o for o in self.tracked_objects
                    if not (o.is_abandoned and self.frame_no - o.last_frame > self.stale_after)
                ]
            return out

    det = Variant(stem, background_method="edge_adaptive")
    det.max_center_drift = drift
    det.static_frame_threshold = dwell
    return det


# name -> (drift, veto, persist[, mobilenet=True])
ABLATION = {
    "0 bare dwell rule (no neural, no drift, no persistence)": (NO_LIMIT, False, False, False),
    "1 + MobileNet-SSD verification":                          (NO_LIMIT, False, False, True),
    "2 + drift bound":                                         (40.0, False, False, True),
    "3 + persistence":                                         (40.0, False, True, True),
    "4 + YOLOv3-tiny person veto (full system)":               (40.0, True, True, True),
    "full w/o drift bound":                                    (NO_LIMIT, True, True),
    "full w/o persistence":                                    (40.0, True, False),
    "full w/o MobileNet check":                                (40.0, True, True, False),
    "full w/o any neural check":                               (40.0, False, True, False),
}


def run_video(vp, drift, veto, persist, dwell, cache_dir, mnet=True):
    stem = os.path.splitext(os.path.basename(vp))[0].lower()
    key = f"{stem}__d{drift:g}_s{dwell}_v{int(veto)}_p{int(persist)}{'' if mnet else '_m0'}.json"
    path = os.path.join(cache_dir, key)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    cap = cv2.VideoCapture(vp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    det = make_detector((veto, persist, mnet), stem, drift, dwell)
    events, abandoned_by_frame, frames, comp = [], {}, 0, 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        t0 = time.perf_counter()
        out = det.process(frame)
        comp += time.perf_counter() - t0
        for ev in out["new_events"]:
            events.append({"frame": int(ev["frame_no"]), "bbox": [int(v) for v in ev["bbox"]],
                           "owner": ev["owner_bbox"] is not None})
        if frames % 5 == 0:
            boxes = [[int(v) for v in o.dim] for o in det.tracked_objects if o.is_abandoned]
            if boxes:
                abandoned_by_frame[str(frames)] = boxes
    cap.release()
    run = {"video": stem, "fps": fps, "frames": frames, "compute_s": comp,
           "events": events, "abandoned_by_frame": abandoned_by_frame}
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(run, f)
    return run


def evaluate(videos, gt, drift, veto, persist, dwell, args, mnet=True):
    rows, T = [], dict(gt=0, tp=0, fp=0, fn=0, own=0, frames=0, comp=0.0)
    delays, covs = [], []
    for vp in videos:
        stem = os.path.splitext(os.path.basename(vp))[0].lower()
        if stem not in gt:
            print(f"[warn] no GT for {stem}, skipped")
            continue
        run = run_video(vp, drift, veto, persist, dwell, args.cache_dir, mnet)
        g = gt[stem]
        tps, fp, fn = match_events(g, run["events"], run["fps"], args.pad, args.max_delay_s,
                                   early_tol=int(args.early_tol_s * run["fps"]))
        d = [(ev["frame"] - g[gi]["frame"]) / run["fps"] for gi, ev in tps]
        c = alert_coverage(run, g, tps, args.pad)
        delays += d
        covs += c
        own = sum(1 for _, ev in tps if ev["owner"])
        rows.append((stem, len(g), len(tps), len(fp), len(fn), d, c, own, run["frames"] / run["compute_s"]))
        T["gt"] += len(g); T["tp"] += len(tps); T["fp"] += len(fp); T["fn"] += len(fn)
        T["own"] += own; T["frames"] += run["frames"]; T["comp"] += run["compute_s"]
    return rows, T, delays, covs


def summarize(T, delays, covs):
    p, r, f1 = prf(T["tp"], T["fp"], T["fn"])
    plo, phi = wilson(T["tp"], T["tp"] + T["fp"])
    rlo, rhi = wilson(T["tp"], T["tp"] + T["fn"])
    return {
        "tp": T["tp"], "fp": T["fp"], "fn": T["fn"], "p": p, "r": r, "f1": f1,
        "p_ci": (plo, phi), "r_ci": (rlo, rhi),
        "delay": sum(delays) / len(delays) if delays else float("nan"),
        "cov": sum(covs) / len(covs) if covs else float("nan"),
        "own": T["own"], "gt": T["gt"],
        "fps": T["frames"] / T["comp"] if T["comp"] > 0 else 0.0,
    }


def fmt_row(name, s):
    return (f"| {name} | {s['tp']} | {s['fp']} | {s['fn']} | {s['p']:.3f} [{s['p_ci'][0]:.2f},{s['p_ci'][1]:.2f}] | "
            f"{s['r']:.3f} [{s['r_ci'][0]:.2f},{s['r_ci'][1]:.2f}] | {s['f1']:.3f} | "
            f"{s['delay']:.1f} | {100 * s['cov']:.0f}% | {s['own']}/{s['gt']} | {s['fps']:.1f} |")


HEADER = ("| Configuration | TP | FP | FN | Precision [95% CI] | Recall [95% CI] | F1 | Mean delay (s) | "
          "Alert persistence | Owner box returned | FPS (total frames / total compute) |\n"
          "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")


# ------------------------------------------------------------------ modes
def explain(videos, gt, args):
    """Print GT events next to alerts so every FP/FN can be traced to time, location, or a real miss."""
    for vp in videos:
        stem = os.path.splitext(os.path.basename(vp))[0].lower()
        if stem not in gt:
            continue
        run = run_video(vp, 40.0, True, True, 146, args.cache_dir)
        fps = run["fps"]
        print(f"\n{stem}: {run['frames']} frames @ {fps:.1f} fps")
        for g in gt[stem]:
            print(f"  GT    frame {g['frame']:>5} ({g['frame'] / fps:6.1f}s) bbox {g['bbox']}")
        if not run["events"]:
            print("  (no alerts)")
        for ev in run["events"]:
            cx, cy = (ev["bbox"][0] + ev["bbox"][2]) / 2, (ev["bbox"][1] + ev["bbox"][3]) / 2
            best = None
            for g in gt[stem]:
                gx, gy = (g["bbox"][0] + g["bbox"][2]) / 2, (g["bbox"][1] + g["bbox"][3]) / 2
                dist = math.hypot(cx - gx, cy - gy)
                if best is None or dist < best[0]:
                    best = (dist, (ev["frame"] - g["frame"]) / fps, spatial_ok(ev["bbox"], g["bbox"], args.pad))
            dist, dt, sp = best
            print(f"  ALERT frame {ev['frame']:>5} ({ev['frame'] / fps:6.1f}s) bbox {ev['bbox']} "
                  f"-> nearest GT: dt={dt:+.1f}s, centre dist={dist:.0f}px, spatial={'ok' if sp else 'NO'}")


def mode_score(videos, gt, args):
    rows, T, delays, covs = evaluate(videos, gt, 40.0, True, True, 146, args)
    print("\n| Video | GT | TP | FP | FN | Delay (s) | Persistence | Owner | FPS |\n| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for stem, n, tp, fp, fn, d, c, own, fps in rows:
        ds = f"{sum(d) / len(d):.1f}" if d else "-"
        cs = f"{100 * sum(c) / len(c):.0f}%" if c else "-"
        print(f"| {stem} | {n} | {tp} | {fp} | {fn} | {ds} | {cs} | {own}/{n} | {fps:.1f} |")
    print("\n" + HEADER)
    print(fmt_row("Full pipeline (delta_max=40, theta_s=146)", summarize(T, delays, covs)))
    if args.explain:
        explain(videos, gt, args)


def mode_ablate(videos, gt, args):
    print("\n" + HEADER)
    for name, cfg in ABLATION.items():
        drift, veto, persist = cfg[:3]
        mnet = cfg[3] if len(cfg) > 3 else True
        _, T, d, c = evaluate(videos, gt, drift, veto, persist, 146, args, mnet)
        print(fmt_row(name, summarize(T, d, c)))


def mode_sweep(videos, gt, args):
    drifts = [20.0, 30.0, 40.0, 60.0, 80.0, NO_LIMIT]
    dwells = [90, 120, 146, 180]
    stems = [os.path.splitext(os.path.basename(v))[0].lower() for v in videos]
    stems = [s for s in stems if s in gt]
    counts = {}  # (drift, dwell) -> stem -> (tp, fp, fn)
    for dr in drifts:
        for dw in dwells:
            rows, _, _, _ = evaluate(videos, gt, dr, True, True, dw, args)
            counts[(dr, dw)] = {r[0]: (r[2], r[3], r[4]) for r in rows}

    def total(cfg, held=None):
        t = [0, 0, 0]
        for s in stems:
            if s == held:
                continue
            for i in range(3):
                t[i] += counts[cfg][s][i]
        return t

    default = (40.0, 146)
    print("\n| delta_max | theta_s | TP | FP | FN | F1 (all videos, in-sample) |\n| :---: | :---: | :---: | :---: | :---: | :---: |")
    for cfg in counts:
        tp, fp, fn = total(cfg)
        print(f"| {'inf' if cfg[0] >= NO_LIMIT else cfg[0]:} | {cfg[1]} | {tp} | {fp} | {fn} | {prf(tp, fp, fn)[2]:.3f} |")

    # leave-one-video-out: choose config on the other videos, score on the held-out one
    lov = [0, 0, 0]
    print("\nLeave-one-video-out selection:")
    for s in stems:
        best = max(counts, key=lambda c: (prf(*total(c, held=s))[2], c == default))
        for i in range(3):
            lov[i] += counts[best][s][i]
        print(f"  held-out {s:<8} chose delta_max={'inf' if best[0] >= NO_LIMIT else best[0]}, theta_s={best[1]}")
    tp, fp, fn = lov
    p, r, f1 = prf(tp, fp, fn)
    print(f"\nLOVO result: TP={tp} FP={fp} FN={fn} P={p:.3f} R={r:.3f} F1={f1:.3f}")
    tp, fp, fn = total(default)
    print(f"Fixed default (40, 146): TP={tp} FP={fp} FN={fn} F1={prf(tp, fp, fn)[2]:.3f}")
    best_in = max(counts, key=lambda c: prf(*total(c))[2])
    tp, fp, fn = total(best_in)
    print(f"In-sample best {best_in}: F1={prf(tp, fp, fn)[2]:.3f}  (optimistic; do NOT report as performance)")


def mode_snap(videos, gt, args):
    """Save images for visual inspection: GT frames (green box) and every alert (red box, nearest GT in green)."""
    out = "snaps"
    os.makedirs(out, exist_ok=True)

    def grab(vp, frame_no):
        cap = cv2.VideoCapture(vp)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no - 1))
        ret, img = cap.read()
        cap.release()
        return img if ret else None

    for vp in videos:
        stem = os.path.splitext(os.path.basename(vp))[0].lower()
        if stem not in gt:
            continue
        run = run_video(vp, 40.0, True, True, 146, args.cache_dir)
        for i, g in enumerate(gt[stem]):
            for tag, fr in (("at", g["frame"]), ("plus10s", g["frame"] + int(10 * run["fps"]))):
                img = grab(vp, min(fr, run["frames"]))
                if img is None:
                    continue
                cv2.rectangle(img, tuple(g["bbox"][:2]), tuple(g["bbox"][2:]), (0, 255, 0), 2)
                cv2.putText(img, f"{stem} GT{i} {tag} f={fr}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.imwrite(os.path.join(out, f"{stem}_GT{i}_{tag}.jpg"), img)
        for ev in run["events"]:
            img = grab(vp, ev["frame"])
            if img is None:
                continue
            for g in gt[stem]:
                cv2.rectangle(img, tuple(g["bbox"][:2]), tuple(g["bbox"][2:]), (0, 255, 0), 1)
            cv2.rectangle(img, tuple(ev["bbox"][:2]), tuple(ev["bbox"][2:]), (0, 0, 255), 2)
            cv2.putText(img, f"{stem} ALERT f={ev['frame']} (red=alert, green=GT)", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(out, f"{stem}_ALERT_f{ev['frame']}.jpg"), img)
    print(f"images written to ./{out}/")


def mode_annotate(args):
    gt = json.load(open(args.gt)) if os.path.exists(args.gt) else {}
    stem = os.path.splitext(os.path.basename(args.annotate))[0].lower()
    cap = cv2.VideoCapture(args.annotate)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    print("keys: a/d = -1/+1 frame, w/s = +30/-30, r = mark object (select box, ENTER), q = save & quit")
    gt[stem] = []
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        vis = frame.copy()
        cv2.putText(vis, f"{stem} frame {idx + 1}/{n} marked={len(gt[stem])}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("annotate", vis)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("d"): idx = min(n - 1, idx + 1)
        elif k == ord("a"): idx = max(0, idx - 1)
        elif k == ord("w"): idx = min(n - 1, idx + 30)
        elif k == ord("s"): idx = max(0, idx - 30)
        elif k == ord("r"):
            x, y, w, h = cv2.selectROI("annotate", frame, False)
            if w > 0 and h > 0:
                gt[stem].append({"frame": idx + 1, "bbox": [int(x), int(y), int(x + w), int(y + h)]})
        elif k == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    with open(args.gt, "w") as f:
        json.dump(gt, f, indent=1)
    print(f"saved {len(gt[stem])} event(s) for {stem} -> {args.gt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["annotate", "score", "ablate", "sweep", "snap"], default="score")
    ap.add_argument("--dir", default="./ABODA")
    ap.add_argument("--gt", default="gt_aboda.json")
    ap.add_argument("--annotate", help="video path to annotate (mode=annotate)")
    ap.add_argument("--cache-dir", default="./aod_cache")
    ap.add_argument("--pad", type=int, default=30)
    ap.add_argument("--max-delay-s", type=float, default=60.0)
    ap.add_argument("--early-tol-s", type=float, default=7.0,
                    help="accept alerts up to this many seconds BEFORE the annotated frame (default 7 s ~ "
                         "dwell threshold 146 f + 2 confirmation checks; the dwell clock starts at placement, "
                         "the GT frame is when the owner leaves)")
    ap.add_argument("--explain", action="store_true", help="print GT vs alerts per video (score mode)")
    args = ap.parse_args()

    if args.mode == "annotate":
        return mode_annotate(args)
    gt = json.load(open(args.gt))
    videos = sorted(glob.glob(os.path.join(args.dir, "*.avi")), key=natural_key)
    {"score": mode_score, "ablate": mode_ablate, "sweep": mode_sweep, "snap": mode_snap}[args.mode](videos, gt, args)


if __name__ == "__main__":
    main()