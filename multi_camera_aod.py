import argparse
import queue
import threading
import time
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from abandoned_object_detector import AbandonedObjectDetector

CAMERA_SOURCES: Dict[str, Union[int, str]] = {
    "CAM_1": "ABODA/video1.avi",
    "CAM_2": "ABODA/video2.avi",
    "CAM_3": "ABODA/video9.avi",
    "CAM_4": "ABODA/video10.avi",
}

DISPLAY_TILE_SIZE = (480, 270)     
MOSAIC_COLS = 2

class CameraWorker(threading.Thread):

    def __init__(self, camera_id: str, source: Union[int, str], event_queue: "queue.Queue"):
        super().__init__(daemon=True, name=f"worker-{camera_id}")
        self.camera_id = camera_id
        self.source = source
        self.event_queue = event_queue
        self.detector = AbandonedObjectDetector(camera_id=camera_id)
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self.opened_ok = True

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[{self.camera_id}] ERROR: could not open source {self.source!r}")
            self.opened_ok = False
            return

        while not self._stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"[{self.camera_id}] stream ended.")
                break

            result = self.detector.process(frame)

            with self._lock:
                self._latest_frame = result["annotated"]

            for event in result["new_events"]:
                self.event_queue.put(event)

        cap.release()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def stop(self):
        self._stop_flag.set()


def make_mosaic(frames: List[Optional[np.ndarray]], tile_size, cols: int) -> np.ndarray:
    tiles = []
    for f in frames:
        tile = np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8) if f is None \
            else cv2.resize(f, tile_size)
        tiles.append(tile)

    rows = []
    for i in range(0, len(tiles), cols):
        row_tiles = tiles[i:i + cols]
        while len(row_tiles) < cols:
            row_tiles.append(np.zeros_like(tiles[0]))
        rows.append(cv2.hconcat(row_tiles))
    return cv2.vconcat(rows)


def main(headless: bool = False, max_seconds: Optional[float] = None):
    event_queue: "queue.Queue" = queue.Queue()
    workers = [CameraWorker(cam_id, src, event_queue) for cam_id, src in CAMERA_SOURCES.items()]
    for w in workers:
        w.start()

    start_time = time.time()
    try:
        while True:
            if not headless:
                frames = [w.get_latest_frame() for w in workers]
                mosaic = make_mosaic(frames, DISPLAY_TILE_SIZE, MOSAIC_COLS)
                cv2.imshow("Multi-Camera Abandoned Object Detection", mosaic)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)
            if max_seconds is not None and time.time() - start_time > max_seconds:
                print("Max run time reached, stopping.")
                break

    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C)...")

    finally:
        for w in workers:
            w.stop()
        for w in workers:
            w.join(timeout=2)
            
        if not headless:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-camera Abandoned Object Detection")
    parser.add_argument("--headless", action="store_true",
                         help="Run without opening a display window (for servers/SSH).")
    args = parser.parse_args()
    main(headless=args.headless)
