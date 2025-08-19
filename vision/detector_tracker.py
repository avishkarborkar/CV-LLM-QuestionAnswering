import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", device="cpu"):
        """Initialize YOLOv8 model"""
        self.model = YOLO(model_path)
        self.model.to(device)
        self.class_names = self.model.names

    def detect(self, image: np.ndarray, conf_threshold=0.3):
        """
        Run object detection on input image
        Returns: List of detections in format [x1,y1,x2,y2,score,class_id]
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run inference
        results = self.model(image_rgb, verbose=False)

        # Process results
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, score, class_id in zip(boxes, scores, class_ids):
                if score < conf_threshold:
                    continue
                detections.append({
                    "bbox": box.tolist(),  # [x1,y1,x2,y2]
                    "score": float(score),
                    "class_id": int(class_id),
                    "class_name": self.class_names[class_id]
                })

        return detections


class DeepSortTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30)

    def update(self, detections, frame):
        """
        Input detections: list of [x1, y1, x2, y2, conf]
        Returns list of track objects with id, bbox.
        """
        # DeepSort expects detections as (xywh, confidence, class)
        dets_for_tracker = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["score"]
            w, h = x2 - x1, y2 - y1
            dets_for_tracker.append(([x1, y1, w, h], conf, det["class_name"]))

        tracks = self.tracker.update_tracks(dets_for_tracker, frame=frame)
        valid_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            valid_tracks.append({
                'track_id': track.track_id,
                'bbox': ltrb,
                'class_name': track.det_class if hasattr(track, 'det_class') else 'object'
            })
        return valid_tracks


def process_video(input_path, output_path, yolo_model_path, show_output=False):
    # Initialize models
    detector = YOLODetector(model_path=yolo_model_path)
    tracker = DeepSortTracker()

    # Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_to_process = int(10 * fps)  # Calculate frames in 10 seconds

    #total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame
    frame_count = 0
    while frame_count < frames_to_process:  # Only process first 10 seconds worth of frames
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        detections = detector.detect(frame)

        # Run tracking
        tracks = tracker.update(detections, frame)

        # Draw results on frame
        for track in tracks:
            x1, y1, x2, y2 = map(int, track['bbox'])
            track_id = track['track_id']
            class_name = track['class_name']

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label and track ID
            label = f"{class_name} ID:{track_id}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)

        # Show output if requested
        if show_output:
            cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1
        print(f"Processed frame {frame_count}/{frames_to_process}", end='\r')

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nProcessing complete. Output saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    input_video = "/Users/avishkarborkar/ComputerVisionProjects/vision_llm_video_qa/vox_samples/5r0dWxy17C8.mp4"
    output_video = "/Users/avishkarborkar/ComputerVisionProjects/vision_llm_video_qa/data/results/5r0dWxy17C8_yolo_deepsort.mp4"
    yolo_model = "/Users/avishkarborkar/ComputerVisionProjects/vision_llm_video_qa/models/detector/yolo11l.pt"

    process_video(
        input_path=input_video,
        output_path=output_video,
        yolo_model_path=yolo_model,
        show_output=True
    )