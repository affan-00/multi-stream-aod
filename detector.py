import os
import time

import cv2
import numpy as np


class Detector:
    """YOLOv3-tiny wrapper (OpenCV DNN).

    Changes vs. the original:
      * swap_rb: Darknet models expect RGB; OpenCV loads frames as BGR. The original used swapRB=False.
        Default is controlled by env var YOLO_SWAPRB (0 = original behaviour, 1 = corrected) so the SAME
        code can be run both ways for the paper without editing files.
      * Output-layer names are computed once in __init__ instead of on every detect() call (speed only).
    """

    def __init__(self, yoloFolder, confidence, threshold, swap_rb=None):
        weightsPath = os.path.sep.join([yoloFolder, "yolov3-tiny.weights"])
        configPath = os.path.sep.join([yoloFolder, "yolov3-tiny.cfg"])
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        self.confidence = confidence
        self.threshold = threshold
        if swap_rb is None:
            swap_rb = os.environ.get("YOLO_SWAPRB", "0") == "1"
        self.swap_rb = swap_rb
        ln = self.net.getLayerNames()
        self.out_layers = [ln[i - 1] for i in np.array(self.net.getUnconnectedOutLayers()).flatten()]

    def detect(self, image):
        (H, W) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=self.swap_rb, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.out_layers)
        boxes, confidences, classIDs = [], [], []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = int(np.argmax(scores))
                confidence = scores[classID]
                if confidence > self.confidence:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
        return boxes, idxs, classIDs, confidences
