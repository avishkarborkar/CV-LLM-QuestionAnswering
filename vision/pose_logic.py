# vision/pose_logic.py
from typing import List, Optional


def interpret_pose(keypoints: Optional[List[List[float]]]) -> str:
    """
    Interprets pose keypoints to determine if a person is sitting or standing.
    This is a simplified heuristic-based approach.

    Args:
        keypoints: A list of [x, y] coordinates from YOLOv8-Pose.

    Returns:
        A string label: "sitting", "standing", or "unknown".
    """
    if keypoints is None or len(keypoints) < 17:
        return "unknown"

    try:
        # Keypoint indices for YOLOv8-Pose (COCO format)
        # We need shoulders, hips, and knees.
        # 5: left_shoulder, 6: right_shoulder
        # 11: left_hip, 12: right_hip
        # 13: left_knee, 14: right_knee

        # Get y-coordinates
        left_shoulder_y = keypoints[5][1]
        right_shoulder_y = keypoints[6][1]
        left_hip_y = keypoints[11][1]
        right_hip_y = keypoints[12][1]
        left_knee_y = keypoints[13][1]
        right_knee_y = keypoints[14][1]

        # Simple Heuristic: If knees are positioned significantly higher than hips
        # (i.e., smaller y-value), the person is likely sitting.
        # We check if the average knee y-coordinate is above the average hip y-coordinate.
        avg_hip_y = (left_hip_y + right_hip_y) / 2
        avg_knee_y = (left_knee_y + right_knee_y) / 2

        # If a key part is not detected (y-coordinate is 0), we can't be sure.
        if avg_hip_y == 0 or avg_knee_y == 0:
            return "unknown"

        # In a typical sitting pose, the knees are bent, so their y-coordinate in the image
        # is often higher (closer to the top of the image) than the hips.
        # A simple threshold can work. Let's check if the hip is below the knee.
        if avg_hip_y > avg_knee_y:
            return "sitting"
        else:
            return "standing"

    except (IndexError, TypeError):
        return "unknown"