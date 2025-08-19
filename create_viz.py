import cv2
import numpy as np
import os

from vision.yolo_detector import YOLODetector
from vision.tracker import DeepSortTracker
from vision.pose_estimator import PoseEstimator
from vision.pose_logic import interpret_pose
from video_kg.builder import get_dominant_color

# --- Configuration ---
VIDEO_PATH = "/Users/avishkarborkar/ComputerVisionProjects/vision_llm_video_qa/data/videoplayback.mp4"  # Path to the interview video
OUTPUT_DIR = "/Users/avishkarborkar/ComputerVisionProjects/vision_llm_video_qa/data/results"
MAX_DURATION_SECONDS = 20


# --- Helper Functions for Drawing ---

def setup_video_writer(output_path, fps, width, height):
    """Initializes a VideoWriter object."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def draw_phase1_visuals(frame, detections, tracks):
    """Draws boxes for detection and tracking."""
    # Draw raw YOLO detections in YELLOW
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

    # Draw confirmed DeepSORT tracks in GREEN
    for track in tracks:
        track_id = track['track_id']
        x1, y1, x2, y2 = map(int, track['bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame


def draw_phase2_visuals(frame, person_entity, entity_bbox):
    """Draws detailed analysis for a single person."""
    x1, y1, x2, y2 = entity_bbox

    # Draw pose skeleton
    keypoints = person_entity.get('pose_keypoints')
    if keypoints:
        for x, y in keypoints:
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x) + x1, int(y) + y1), 3, (0, 0, 255), -1)

    # Draw a panel with the extracted attributes
    panel_x, panel_y = x2 + 10, y1
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 250, panel_y + 120), (0, 0, 0), -1)

    # Pose Label
    pose_label = interpret_pose(keypoints)
    cv2.putText(frame, f"Pose: {pose_label}", (panel_x + 10, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

    # Color Swatch
    color = person_entity.get('dominant_color')
    if color:
        # BGR for OpenCV
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.putText(frame, "Color:", (panel_x + 10, panel_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.rectangle(frame, (panel_x + 90, panel_y + 55), (panel_x + 140, panel_y + 80), color_bgr, -1)

    return frame


def create_phase3_video(output_path, width, height, fps):
    """Creates a static video from an image showing the final Q&A."""
    # Create a blank canvas
    canvas = np.zeros((height, width, 3), dtype="uint8")

    # --- Add Text to the Canvas ---
    # You can customize fonts and positions
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    cv2.putText(canvas, "Phase 3: Synthesis & Q&A", (50, 80), font, 1.2, (255, 255, 255), 2)

    # Extracted Facts (The KG Summary)
    cv2.putText(canvas, "[Extracted Facts]", (50, 150), font, 0.8, (0, 255, 255), 2)  # Yellow
    cv2.putText(canvas, "- Person (ID: 1) is sitting on the left.", (50, 190), font, 0.7, (255, 255, 255), 1)
    cv2.putText(canvas, "- Person (ID: 1) is wearing purple.", (50, 220), font, 0.7, (255, 255, 255), 1)

    # User Question
    cv2.putText(canvas, "[User Question]", (50, 300), font, 0.8, (0, 255, 0), 2)  # Green
    cv2.putText(canvas, "Q: Describe the person on the left and what they are doing.", (50, 340), font, 0.7,
                (255, 255, 255), 1)

    # LLM Answer
    cv2.putText(canvas, "[LLM Answer]", (50, 420), font, 0.8, (255, 100, 100), 2)  # Blue
    cv2.putText(canvas, "A: The person on the left is sitting down and wearing a purple dress.", (50, 460), font, 0.7,
                (255, 255, 255), 1)

    # --- Write to Video File ---
    writer = setup_video_writer(output_path, fps, width, height)
    # Write the same frame for 5 seconds to create a short clip
    for _ in range(int(fps * 5)):
        writer.write(canvas)
    writer.release()
    print(f"✅ Generated Phase 3 video: {output_path}")


# --- Main Generation Logic ---
def generate_showcase_videos():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- Initialize Modules ---
    print("Initializing models...")
    detector = YOLODetector("yolov8n.pt")
    tracker = DeepSortTracker()
    pose_estimator = PoseEstimator()
    print("Initialization complete.")

    # --- Get Video Properties ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {VIDEO_PATH}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # --- Setup Video Writers ---
    writer1 = setup_video_writer(os.path.join(OUTPUT_DIR, "1_detection_and_tracking.mp4"), fps, width, height)
    writer2 = setup_video_writer(os.path.join(OUTPUT_DIR, "2_attribute_analysis.mp4"), fps, width, height)

    print("Starting video processing to generate visualizations...")
    frame_count = 0
    while cap.isOpened() and frame_count < int(MAX_DURATION_SECONDS * fps):
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        # --- Generate Frames for Each Video ---
        # Video 1: Detection and Tracking
        frame1 = frame.copy()
        frame1 = draw_phase1_visuals(frame1, detections, tracks)
        writer1.write(frame1)

        # Video 2: Detailed Attribute Analysis
        frame2 = frame.copy()
        # Find a specific person to focus on (e.g., the first one detected)
        if tracks:
            focused_track = tracks[0]  # Focus on the first tracked object
            bbox_int = [int(c) for c in focused_track['bbox']]

            # Re-extract entity info for visualization
            entity_crop = frame[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]]
            entity_data = {
                'pose_keypoints': pose_estimator.estimate(entity_crop),
                'dominant_color': get_dominant_color(entity_crop)
            }
            frame2 = draw_phase2_visuals(frame2, entity_data, bbox_int)
        writer2.write(frame2)

        frame_count += 1
        print(f"\rProcessing frame {frame_count}...", end="")

    # --- Finalize and Release ---
    print("\nFinalizing videos...")
    cap.release()
    writer1.release()
    writer2.release()
    print(f"✅ Generated Phase 1 & 2 videos in '{OUTPUT_DIR}' directory.")

    # Generate the static Q&A video
    create_phase3_video(os.path.join(OUTPUT_DIR, "3_synthesis_qa.mp4"), width, height, fps)


if __name__ == "__main__":
    generate_showcase_videos()