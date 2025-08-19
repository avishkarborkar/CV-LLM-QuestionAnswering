import cv2
import base64
import numpy as np
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Tuple
from sklearn.cluster import KMeans

# Import the new PoseEstimator module
from vision.pose_estimator import PoseEstimator


# --- Pydantic Models: The structure of our Knowledge Graph ---

class Entity(BaseModel):
    """Represents a single tracked object in a frame."""
    type: str
    bbox: List[float]
    dominant_color: Optional[Tuple[int, int, int]] = None
    pose_keypoints: Optional[List[List[float]]] = None  # Field for pose data


class FrameData(BaseModel):
    """Represents all the data extracted from a single video frame."""
    timestamp: float
    scene_brightness: Optional[float] = None
    entities: Dict[str, Entity]


class KnowledgeGraph(BaseModel):
    """The root object holding all extracted video information."""
    metadata: Dict[str, Any]
    frames: List[FrameData]


# --- Helper Functions for Data Extraction ---

def get_dominant_color(image_crop: np.ndarray, k: int = 3) -> Optional[Tuple[int, int, int]]:
    """Finds the dominant color in an image crop using K-Means clustering."""
    if image_crop.size == 0:
        return None
    try:
        pixels = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB).reshape((-1, 3))
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        _, counts = np.unique(labels, return_counts=True)
        dominant_center = centers[np.argmax(counts)]
        return tuple(int(c) for c in dominant_center)
    except Exception:
        return None


def get_scene_brightness(frame: np.ndarray) -> float:
    """Calculates the average brightness of a frame (0-255)."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray_frame)


# --- The Main Builder Function ---

def build_knowledge_graph(video_path: str, detector, tracker, max_seconds: int = 10) -> KnowledgeGraph:
    """
    Constructs a knowledge graph from a video file, including object tracking,
    color analysis, scene brightness, and pose estimation.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0

    kg = KnowledgeGraph(
        metadata={
            "video_path": video_path,
            "fps": fps,
            "resolution": (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
        },
        frames=[]
    )

    # Initialize the Pose Estimator once
    pose_estimator = PoseEstimator()

    frame_limit = int(max_seconds * fps)
    print(f"Processing up to {frame_limit} frames...")

    for frame_count in range(frame_limit):
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended.")
            break

        # 1. Analyze the overall frame for brightness
        brightness = get_scene_brightness(frame)

        # 2. Detect and track objects
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        # 3. For each tracked object, extract detailed attributes
        frame_entities = {}
        for track in tracks:
            track_id = track.get("track_id")
            bbox = track.get("bbox")
            class_name = track.get("class_name", "object")

            if not all([track_id, bbox, class_name]):
                continue

            # Get the image crop for the object
            x1, y1, x2, y2 = map(int, bbox)
            image_crop = frame[y1:y2, x1:x2]

            # Extract dominant color
            color_data = get_dominant_color(image_crop)

            # Extract pose keypoints ONLY if the object is a person
            pose_data = None
            if class_name == 'person':
                pose_data = pose_estimator.estimate(image_crop)

            # Store all extracted data in the Entity object
            frame_entities[track_id] = Entity(
                type=class_name,
                bbox=bbox,
                dominant_color=color_data,
                pose_keypoints=pose_data
            )

        # Only add frame data to the KG if there were objects in it
        if frame_entities:
            kg.frames.append(FrameData(
                timestamp=round(frame_count / fps, 2),
                entities=frame_entities,
                scene_brightness=brightness
            ))

    cap.release()
    print(f"\n✅ Knowledge Graph constructed with {len(kg.frames)} frames.")

    if not kg.frames:
        raise ValueError("No objects were detected or tracked in the video segment.")

    return kg