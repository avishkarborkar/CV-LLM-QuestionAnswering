# vision/tracker.py

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import List, Dict, Any


class DeepSortTracker:
    def __init__(self):
        try:
            # The model is downloaded automatically by the library on first run.
            # Using 'osnet_x0_25', a lightweight and effective ReID model.
            self.tracker = DeepSort(max_age=30, n_init=2, embedder='mobilenet')
            print("✅ DeepSortTracker initialized successfully.")
        except Exception as e:
            # This will catch any other initialization errors.
            raise RuntimeError(f"🚨 DeepSortTracker initialization failed: {str(e)}")

    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Updates the tracker with new detections.

        Args:
            detections: A list of detection dictionaries from YOLO.
                        Expected format: [{'bbox': [x1,y1,x2,y2], 'score': float, 'class_name': str}, ...]
            frame: The current video frame.

        Returns:
            A list of tracked objects with their IDs.
            Format: [{'track_id': str, 'bbox': [x1,y1,x2,y2], 'class_name': str}, ...]
        """
        if not detections:
            # It's important to still call update_tracks to handle aging of existing tracks
            self.tracker.update_tracks([], frame=frame)
            return []

        # Format detections for deep_sort_realtime
        # It expects a list of tuples: ([(l,t,w,h)], score, class_name)
        dets_for_deepsort = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w, h = x2 - x1, y2 - y1
            bbox_ltwh = [x1, y1, w, h]

            dets_for_deepsort.append((bbox_ltwh, det["score"], det["class_name"]))

        # Update the tracker
        try:
            tracks = self.tracker.update_tracks(dets_for_deepsort, frame=frame)
        except Exception as e:
            print(f"🚨 Error updating DeepSort tracks: {e}")
            return []

        # Format results
        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # Get bounding box in (x1, y1, x2, y2) format
            class_name = track.get_det_class()  # Retrieve the class name associated with the track

            tracked_objects.append({
                "track_id": str(track_id),
                "bbox": [float(c) for c in ltrb],
                "class_name": str(class_name)
            })

        return tracked_objects