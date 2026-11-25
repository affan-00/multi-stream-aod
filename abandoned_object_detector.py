import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from detector import Detector


def iou(box_a: List[int], box_b: List[int]) -> float:
    """Intersection-over-Union of two [x1, y1, x2, y2] boxes."""
    xA, yA = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xB, yB = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    areaA = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    areaB = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    denom = areaA + areaB - inter
    return inter / denom if denom > 0 else 0.0


def containment(inner_box: List[int], outer_box: List[int]) -> float:
    """Fraction of inner_box area contained inside outer_box."""
    xA, yA = max(inner_box[0], outer_box[0]), max(inner_box[1], outer_box[1])
    xB, yB = min(inner_box[2], outer_box[2]), min(inner_box[3], outer_box[3])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    area_in = (inner_box[2] - inner_box[0] + 1) * (inner_box[3] - inner_box[1] + 1)
    return inter / area_in if area_in > 0 else 0.0


@dataclass
class TrackedObject:
    idx: int
    center: Tuple[int, int]
    dim: List[int]
    frames: List[int] = field(default_factory=list)
    first_center: Optional[Tuple[int, int]] = None
    is_abandoned: bool = False
    owner_box: Optional[List[int]] = None
    drift_lock_delay: int = 30
    _locked_center: Optional[Tuple[int, int]] = None
    _last_classify_frame: int = -10_000

    def __post_init__(self):
        if self.first_center is None:
            self.first_center = self.center

    def update(self, frame_no: int, dim: List[int], center: Tuple[int, int]):
        self.frames.append(frame_no)
        self.dim = dim
        self.center = center
        if self._locked_center is None and self.static_count >= self.drift_lock_delay:
            self._locked_center = center

    @property
    def last_frame(self) -> int:
        return self.frames[-1]

    @property
    def static_count(self) -> int:
        return len(self.frames)

    @property
    def drift(self) -> float:
        ref = self._locked_center if self._locked_center is not None else self.first_center
        dx = self.center[0] - ref[0]
        dy = self.center[1] - ref[1]
        return (dx * dx + dy * dy) ** 0.5

    def __str__(self):
        return f"id:{self.idx} c:{self.center} f:{self.static_count} l:{self.last_frame}"


class AbandonedObjectDetector:
    MOBILENET_CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
        "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor",
    ]

    def __init__(
        self,
        camera_id: str,
        yolo_dir: str = "yolo",
        mobilenet_prototxt: str = "mobilenet_data/MobileNetSSD_deploy.prototxt.txt",
        mobilenet_model: str = "mobilenet_data/MobileNetSSD_deploy.caffemodel",
        static_frame_threshold: int = 146,
        stale_after: int = 20,
        warmup_frames: int = 15,
        min_contour_area: int = 250,
        max_contour_area: int = 20000,
        iou_match_thresh: float = 0.40,
        max_center_drift: float = 40.0,
        drift_lock_delay: int = 30,
        reclassify_interval: int = 20,
        canny_low: int = 16,
        canny_high: int = 162,
        background_method: str = "edge_adaptive",
        mog2_history: int = 500,
        mog2_var_threshold: float = 24.0,
        mog2_learning_rate: float = -1,
        detect_shadows: bool = True,
        detection_confidence: float = 0.30,
        nms_threshold: float = 0.10,
        history_len: int = 200,
        history_stride: int = 5,
        alerts_dir: str = "alerts",
    ):
        self.camera_id = camera_id
        self.static_frame_threshold = static_frame_threshold
        self.stale_after = stale_after
        self.warmup_frames = warmup_frames
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        self.iou_match_thresh = iou_match_thresh
        self.max_center_drift = max_center_drift
        self.drift_lock_delay = drift_lock_delay
        self.reclassify_interval = reclassify_interval
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_confidence = detection_confidence
        self.mog2_learning_rate = mog2_learning_rate
        self.history_stride = history_stride
        self.alerts_dir = alerts_dir
        self.background_method = background_method
        self.detect_shadows = detect_shadows

        if background_method == "mog2":
            self._bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=mog2_history, varThreshold=mog2_var_threshold, detectShadows=detect_shadows
            )
        elif background_method == "knn":
            self._bg_sub = cv2.createBackgroundSubtractorKNN(
                history=mog2_history, detectShadows=detect_shadows
            )
        elif background_method in ("edge_adaptive", "first_frame"):
            self._bg_sub = None
        else:
            raise ValueError(
                f"unknown background_method {background_method!r}; "
                "use 'edge_adaptive', 'first_frame', 'mog2', or 'knn'"
            )

        self.mobilenet = cv2.dnn.readNetFromCaffe(mobilenet_prototxt, mobilenet_model)
        self.yolo = Detector(yolo_dir, detection_confidence, nms_threshold)
        labels_path = os.path.join(yolo_dir, "coco.names")
        with open(labels_path, "r", encoding="utf-8") as f:
            self.yolo_labels = f.read().strip().split("\n")

        self._first_frame: Optional[np.ndarray] = None
        self._ref_gray: Optional[np.ndarray] = None
        self._ref_edges: Optional[np.ndarray] = None
        self._next_idx = 0
        self.tracked_objects: List[TrackedObject] = []
        self.frame_no = 0
        self.events: List[dict] = []
        self._history: deque = deque(maxlen=history_len)
        self._votes: Dict[int, List[int]] = {}
        self._yolo_cache_frame = -1
        self._yolo_cache_persons: List[List[int]] = []

    def _reset_ref(self, gray: np.ndarray):
        self._ref_gray = gray.copy()
        self._ref_edges = cv2.dilate(cv2.Canny(gray, 60, 150), np.ones((3, 3), np.uint8))

    def _segment(self, frame: np.ndarray) -> np.ndarray:
        if self.background_method == "edge_adaptive":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self._ref_gray is None:
                self._reset_ref(gray)
                return np.zeros_like(gray)

            ref_mean = float(np.mean(self._ref_gray))
            cur_mean = float(np.mean(gray))
            if abs(cur_mean - ref_mean) > 18.0:
                self._reset_ref(gray)
                return np.zeros_like(gray)

            scale = (ref_mean + 1.0) / (cur_mean + 1.0)
            norm_gray = np.clip(gray.astype(np.float32) * scale, 0, 255).astype(np.uint8)

            diff = cv2.absdiff(self._ref_gray, norm_gray)
            diff_edges = cv2.Canny(diff, self.canny_low, self.canny_high)
            cur_edges = cv2.Canny(norm_gray, 15, 50)
            novel_edges = cv2.bitwise_and(cur_edges, cv2.bitwise_not(self._ref_edges))
            combined = cv2.bitwise_and(diff_edges, cv2.dilate(novel_edges, np.ones((3, 3), np.uint8)))
            mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=5)

            if float(np.count_nonzero(mask)) / mask.size > 0.35:
                self._reset_ref(gray)
                return np.zeros_like(gray)
            return mask

        if self.background_method == "first_frame":
            if self._first_frame is None:
                self._first_frame = frame.copy()
            frame_diff = cv2.absdiff(self._first_frame, frame)
            edged = cv2.Canny(frame_diff, self.canny_low, self.canny_high)
            kernel = np.ones((5, 5), np.uint8)
            return cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=5)

        raw_mask = self._bg_sub.apply(frame, learningRate=self.mog2_learning_rate)
        if self.detect_shadows:
            _, mask = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)
        else:
            mask = raw_mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=5)
        return mask

    def _is_mobilenet_person(self, crop: np.ndarray) -> bool:
        if crop.size == 0:
            return False
        blob = cv2.dnn.blobFromImage(cv2.resize(crop, (300, 300)), 0.007843, (300, 300), 127.5)
        self.mobilenet.setInput(blob)
        dets = self.mobilenet.forward()
        for i in range(dets.shape[2]):
            if float(dets[0, 0, i, 2]) > self.min_confidence:
                if self.MOBILENET_CLASSES[int(dets[0, 0, i, 1])] == "person":
                    return True
        return False

    def _detect_yolo_persons(self, img: np.ndarray) -> List[List[int]]:
        boxes, idxs, class_ids, _ = self.yolo.detect(img)
        persons: List[List[int]] = []
        if idxs is not None and len(idxs) > 0:
            for i in idxs.flatten():
                if self.yolo_labels[class_ids[i]] == "person":
                    x, y, w, h = boxes[i]
                    persons.append([x, y, x + w, y + h])
        return persons

    def _is_covered_by_yolo_person(self, frame: np.ndarray, dim: List[int]) -> bool:
        if self._yolo_cache_frame != self.frame_no:
            self._yolo_cache_persons = self._detect_yolo_persons(frame)
            self._yolo_cache_frame = self.frame_no
        for pbox in self._yolo_cache_persons:
            if containment(dim, pbox) > 0.40 or iou(dim, pbox) > 0.25:
                return True
        return False

    def _find_owner(self, obj: TrackedObject) -> Optional[List[int]]:
        dim = obj.dim
        first_seen = obj.frames[0]
        by_frame = {fn: raw for fn, raw, _ in self._history}
        if not by_frame:
            return None

        all_keys = sorted(by_frame.keys())
        probe_frames = []
        for target in (first_seen - 30, first_seen - 15, first_seen, first_seen + 15):
            nearest = min(all_keys, key=lambda k: abs(k - target))
            if nearest not in probe_frames and abs(nearest - target) <= 40:
                probe_frames.append(nearest)

        cx = (dim[0] + dim[2]) / 2.0
        cy = (dim[1] + dim[3]) / 2.0
        pad = 45
        padded = [dim[0] - pad, dim[1] - pad, dim[2] + pad, dim[3] + pad]
        max_dist = max(dim[2] - dim[0], dim[3] - dim[1]) * 1.8 + 90.0

        best_score, best_person = 0.0, None
        for fn in probe_frames:
            for pbox in self._detect_yolo_persons(by_frame[fn]):
                ov = iou(pbox, padded)
                px = (pbox[0] + pbox[2]) / 2.0
                py = (pbox[1] + pbox[3]) / 2.0
                dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
                if ov > 0.0:
                    score = 1.0 + ov
                elif dist <= max_dist:
                    score = 1.0 - (dist / max_dist)
                else:
                    continue
                if score > best_score:
                    best_score, best_person = score, pbox
        return best_person

    def _save_alert_snapshot(self, frame: np.ndarray, obj: TrackedObject):
        os.makedirs(self.alerts_dir, exist_ok=True)
        x1, y1, x2, y2 = obj.dim
        annotated = frame.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        if obj.owner_box:
            ox1, oy1, ox2, oy2 = obj.owner_box
            cv2.rectangle(annotated, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
        fname = f"{self.camera_id}_obj{obj.idx}_frame{self.frame_no}.jpg"
        cv2.imwrite(os.path.join(self.alerts_dir, fname), annotated)

    def process(self, frame: np.ndarray) -> dict:
        self.frame_no += 1
        new_events: List[dict] = []
        annotated = frame.copy()

        motion_map = self._segment(frame)
        if self.frame_no % self.history_stride == 0 or self.frame_no == 1:
            self._history.append((self.frame_no, frame.copy(), motion_map))

        if self.frame_no <= self.warmup_frames:
            return {
                "camera_id": self.camera_id,
                "frame_no": self.frame_no,
                "annotated": annotated,
                "new_events": new_events,
            }

        contours, _ = cv2.findContours(motion_map.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        abandoned_objs = [o for o in self.tracked_objects if o.is_abandoned]

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_contour_area or area > self.max_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if (w * h) > self.max_contour_area or w > int(frame.shape[1] * 0.40) or h > int(frame.shape[0] * 0.45):
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            dim = [x, y, x + w, y + h]
            if x <= 3 or dim[2] >= frame.shape[1] - 3 or y <= 3 or dim[3] >= frame.shape[0] - 3:
                continue

            if any(
                iou(dim, ao.dim) > 0.15 or ((cx - ao.center[0]) ** 2 + (cy - ao.center[1]) ** 2) ** 0.5 < 45.0
                for ao in abandoned_objs
            ):
                continue

            matched = None
            best_score = self.iou_match_thresh
            for obj in self.tracked_objects:
                if obj.is_abandoned:
                    continue
                score = iou(dim, obj.dim)
                if score > best_score:
                    best_score, matched = score, obj

            if matched is None:
                matched = TrackedObject(self._next_idx, (cx, cy), dim, drift_lock_delay=self.drift_lock_delay)
                matched.frames.append(self.frame_no)
                self._next_idx += 1
                self.tracked_objects.append(matched)
            else:
                matched.update(self.frame_no, dim, (cx, cy))

            still_static = (
                matched.static_count > self.static_frame_threshold
                and matched.drift <= self.max_center_drift
            )
            due_for_check = (self.frame_no - matched._last_classify_frame) >= self.reclassify_interval

            if still_static and not matched.is_abandoned and due_for_check:
                matched._last_classify_frame = self.frame_no
                pad = 25
                crop = frame[
                    max(0, y - pad):min(frame.shape[0], y + h + pad),
                    max(0, x - pad):min(frame.shape[1], x + w + pad),
                ]
                person_vetoes, clean_streak = self._votes.get(matched.idx, [0, 0])
                is_person = self._is_mobilenet_person(crop) or self._is_covered_by_yolo_person(frame, dim)
                if is_person:
                    person_vetoes += 1
                    clean_streak = 0
                else:
                    clean_streak += 1
                self._votes[matched.idx] = [person_vetoes, clean_streak]

                if clean_streak >= 2 and person_vetoes <= 2:
                    matched.is_abandoned = True
                    matched.owner_box = self._find_owner(matched)
                    abandoned_objs.append(matched)
                    event = {
                        "camera_id": self.camera_id,
                        "object_id": matched.idx,
                        "frame_no": self.frame_no,
                        "location": (cx, cy),
                        "bbox": dim,
                        "owner_bbox": matched.owner_box,
                        "timestamp": time.time(),
                    }
                    new_events.append(event)
                    self.events.append(event)

        self.tracked_objects = [
            o for o in self.tracked_objects
            if (self.frame_no - o.last_frame <= self.stale_after) or o.is_abandoned
        ]
        for o in self.tracked_objects:
            if o.is_abandoned:
                x1, y1, x2, y2 = o.dim
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    annotated, "ABANDONED OBJECT", (o.center[0], max(20, o.center[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
                )

        return {
            "camera_id": self.camera_id,
            "frame_no": self.frame_no,
            "annotated": annotated,
            "new_events": new_events,
        }