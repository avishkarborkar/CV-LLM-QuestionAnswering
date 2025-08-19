from ultralytics import YOLO
import numpy as np
import cv2


class PoseEstimator:
    def __init__(self, model_path: str = "/Users/avishkarborkar/ComputerVisionProjects/vision_llm_video_qa/models/detector/yolo11n-pose.pt"):
        """
        Initializes the Pose Estimator with a YOLOv8-Pose model.
        The model will be downloaded automatically by the ultralytics library on first run.
        """
        try:
            self.model = YOLO(model_path)
            print(f"✅ PoseEstimator initialized with model: {model_path}")
        except Exception as e:
            raise RuntimeError(f"🚨 Failed to initialize YOLO Pose model: {str(e)}")

    def estimate(self, person_crop: np.ndarray):
        """
        Estimates the pose for a cropped image of a single person.

        Args:
            person_crop: A NumPy array representing the cropped image of a person.

        Returns:
            A list of [x, y] coordinates for each keypoint, or None if no pose is found.
        """
        if person_crop.size == 0:
            return None

        try:
            # Run the pose estimation model on the cropped image
            results = self.model(person_crop, verbose=False)

            # Check if any keypoints were detected in the result
            if results and results[0].keypoints and results[0].keypoints.xy.numel() > 0:
                # Return the (x, y) coordinates of all keypoints for the first detected pose
                return results[0].keypoints.xy[0].tolist()
        except Exception as e:
            print(f"⚠️ Warning: Pose estimation failed. Error: {e}")

        return None