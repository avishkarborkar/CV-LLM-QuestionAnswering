# vision/yolo_detector.py

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any


class YOLODetector:
    """
    Wrapper for YOLOv8 object detection.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        try:
            self.model = YOLO(model_path)
            self.device = device
            print(f"✅ YOLODetector initialized with model: {model_path} on device: {device}")
        except Exception as e:
            raise RuntimeError(f"🚨 Failed to initialize YOLO model: {str(e)}")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Performs object detection on a single frame.
        """
        try:
            results = self.model(frame, device=self.device, verbose=False)
            detections = []

            # This loop iterates through the results from the model
            for result in results:
                boxes = result.boxes

                # This loop iterates through each bounding box found in the frame
                for box in boxes:
                    # The variables x1, y1, etc., are DEFINED INSIDE this loop
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    score = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]

                    # CORRECT INDENTATION:
                    # This 'if' statement must also be INSIDE the 'for box in boxes:' loop.
                    # This ensures it runs for every box found and has access to its 'score' and 'x1'.
                    if score > 0.3:  # Using the lower confidence threshold
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "score": score,
                            "class_name": class_name
                        })

            return detections
        except Exception as e:
            # The NameError was being caught here and printed.
            print(f"🚨 Error during detection: {str(e)}")
            return []