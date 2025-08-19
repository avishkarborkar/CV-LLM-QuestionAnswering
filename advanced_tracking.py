#!/usr/bin/env python3
"""
Advanced People Tracking with Better Models and Important Keypoints.
Uses YOLOv8 Large for detection and YOLOv8 Large Pose for better keypoint estimation.
Focuses on the most important keypoints for pose analysis.
"""

import cv2
import numpy as np
import os
from vision.yolo_detector import YOLODetector
from vision.tracker import DeepSortTracker
from vision.pose_estimator import PoseEstimator
from vision.pose_logic import interpret_pose

class AdvancedPeopleTracker:
    def __init__(self, detection_model="models/detector/yolov8l.pt", 
                 pose_model="models/detector/yolov8l-pose.pt", device="cpu"):
        """Initialize with better models for improved accuracy."""
        self.detector = YOLODetector(model_path=detection_model, device=device)
        self.tracker = DeepSortTracker()
        self.pose_estimator = PoseEstimator(model_path=pose_model)
        
        # Color palette for different track IDs
        self.colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 128),  # Teal
            (128, 128, 0),  # Olive
        ]
        
        # Important keypoints for pose analysis (COCO format)
        self.important_keypoints = {
            0: "nose",
            1: "left_eye", 
            2: "right_eye",
            5: "left_shoulder",
            6: "right_shoulder",
            7: "left_elbow",
            8: "right_elbow",
            9: "left_wrist",
            10: "right_wrist",
            11: "left_hip",
            12: "right_hip",
            13: "left_knee",
            14: "right_knee",
            15: "left_ankle",
            16: "right_ankle"
        }
        
        # Key skeleton connections for important pose analysis
        self.important_skeleton = [
            (5, 6),   # Left to right shoulder
            (5, 7),   # Left shoulder to left elbow
            (6, 8),   # Right shoulder to right elbow
            (7, 9),   # Left elbow to left wrist
            (8, 10),  # Right elbow to right wrist
            (5, 11),  # Left shoulder to left hip
            (6, 12),  # Right shoulder to right hip
            (11, 12), # Left to right hip
            (11, 13), # Left hip to left knee
            (12, 14), # Right hip to right knee
            (13, 15), # Left knee to left ankle
            (14, 16), # Right knee to right ankle
            (0, 1),   # Nose to left eye
            (0, 2),   # Nose to right eye
        ]
        
        print("✅ AdvancedPeopleTracker initialized with better models!")
        print(f"📊 Detection model: {detection_model}")
        print(f"🎯 Pose model: {pose_model}")
        print(f"🔧 Device: {device}")
    
    def draw_important_keypoints(self, frame, keypoints, bbox, color, show_labels=True):
        """Draw only the important keypoints with labels."""
        if not keypoints:
            return
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw important keypoints
        for idx, (x, y) in enumerate(keypoints):
            if idx in self.important_keypoints and x > 0 and y > 0:
                # Adjust coordinates relative to bounding box
                abs_x = int(x) + x1
                abs_y = int(y) + y1
                
                # Draw keypoint circle (larger for important points)
                cv2.circle(frame, (abs_x, abs_y), 6, color, -1)
                cv2.circle(frame, (abs_x, abs_y), 8, (255, 255, 255), 2)
                
                # Draw keypoint label
                if show_labels:
                    label = self.important_keypoints[idx]
                    cv2.putText(frame, label, (abs_x + 10, abs_y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw important skeleton connections
        for start_idx, end_idx in self.important_skeleton:
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                keypoints[start_idx][0] > 0 and keypoints[start_idx][1] > 0 and
                keypoints[end_idx][0] > 0 and keypoints[end_idx][1] > 0):
                
                start_x = int(keypoints[start_idx][0]) + x1
                start_y = int(keypoints[start_idx][1]) + y1
                end_x = int(keypoints[end_idx][0]) + x1
                end_y = int(keypoints[end_idx][1]) + y1
                
                # Draw thicker lines for important connections
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 3)
    
    def draw_enhanced_bounding_box(self, frame, bbox, track_id, color, confidence=None, pose_label=None):
        """Draw enhanced bounding box with more information."""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box with thicker lines
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Create enhanced label
        label_parts = [f"ID: {track_id}"]
        if confidence:
            label_parts.append(f"Conf: {confidence:.2f}")
        if pose_label:
            label_parts.append(f"Pose: {pose_label}")
        
        label = " | ".join(label_parts)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 15), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_advanced_info_panel(self, frame, frame_count, fps, num_tracks, pose_info, detection_stats):
        """Draw advanced information panel with detailed stats."""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Draw main information
        cv2.putText(frame, f"Frame: {frame_count}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"People Tracked: {num_tracks}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw detection statistics
        if detection_stats:
            cv2.putText(frame, f"Detection Confidence: {detection_stats.get('avg_confidence', 0):.2f}", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw pose information
        y_offset = 135
        for track_id, pose_label in pose_info.items():
            if y_offset < 170:  # Limit to prevent overflow
                cv2.putText(frame, f"ID {str(track_id)}: {str(pose_label)}", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
    
    def process_video(self, video_path, output_path=None, max_frames=None, show_labels=True):
        """Process video with advanced tracking and pose estimation."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video Info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Output will be saved to: {output_path}")
        
        frame_count = 0
        start_time = cv2.getTickCount()
        
        print("🎬 Starting advanced video processing...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if max_frames and frame_count > max_frames:
                break
            
            # Create visualization frame
            vis_frame = frame.copy()
            
            # Detect objects with better model
            detections = self.detector.detect(frame)
            
            # Filter for people only
            people_detections = [det for det in detections if det.get('class_name') == 'person']
            
            # Track people
            tracks = self.tracker.update(people_detections, frame)
            
            # Calculate detection statistics
            detection_stats = {}
            if people_detections:
                confidences = [det.get('confidence', 0) for det in people_detections]
                detection_stats['avg_confidence'] = sum(confidences) / len(confidences)
            
            # Process each tracked person
            pose_info = {}
            for track in tracks:
                track_id = track.get('track_id')
                bbox = track.get('bbox')
                confidence = track.get('confidence')
                
                if not all([track_id, bbox]):
                    continue
                
                # Get color for this track ID
                color = self.colors[int(track_id) % len(self.colors)]
                
                # Extract person crop for pose estimation
                x1, y1, x2, y2 = map(int, bbox)
                person_crop = frame[y1:y2, x1:x2]
                
                pose_label = "unknown"
                if person_crop.size > 0:
                    # Estimate pose with better model
                    keypoints = self.pose_estimator.estimate(person_crop)
                    
                    # Draw important keypoints and skeleton
                    self.draw_important_keypoints(vis_frame, keypoints, bbox, color, show_labels)
                    
                    # Interpret pose
                    pose_label = interpret_pose(keypoints)
                    pose_info[track_id] = pose_label
                
                # Draw enhanced bounding box
                self.draw_enhanced_bounding_box(vis_frame, bbox, track_id, color, confidence, pose_label)
            
            # Calculate current FPS
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Draw advanced information panel
            self.draw_advanced_info_panel(vis_frame, frame_count, current_fps, len(tracks), pose_info, detection_stats)
            
            # Write frame
            if writer:
                writer.write(vis_frame)
            
            # Display frame
            cv2.imshow('Advanced People Tracking', vis_frame)
            
            # Print progress
            if frame_count % 30 == 0:
                print(f"📊 Frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) - People: {len(tracks)}")
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Advanced processing complete! Processed {frame_count} frames.")
        if output_path:
            print(f"💾 Output saved to: {output_path}")
    
    def process_webcam(self, max_frames=None, show_labels=True):
        """Process webcam feed with advanced tracking."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Could not open webcam")
        
        print("📹 Advanced webcam tracking started - Press 'q' to quit")
        
        frame_count = 0
        start_time = cv2.getTickCount()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if max_frames and frame_count > max_frames:
                break
            
            # Create visualization frame
            vis_frame = frame.copy()
            
            # Detect and track people
            detections = self.detector.detect(frame)
            people_detections = [det for det in detections if det.get('class_name') == 'person']
            tracks = self.tracker.update(people_detections, frame)
            
            # Process each tracked person
            pose_info = {}
            for track in tracks:
                track_id = track.get('track_id')
                bbox = track.get('bbox')
                confidence = track.get('confidence')
                
                if not all([track_id, bbox]):
                    continue
                
                color = self.colors[int(track_id) % len(self.colors)]
                
                # Pose estimation
                x1, y1, x2, y2 = map(int, bbox)
                person_crop = frame[y1:y2, x1:x2]
                
                pose_label = "unknown"
                if person_crop.size > 0:
                    keypoints = self.pose_estimator.estimate(person_crop)
                    self.draw_important_keypoints(vis_frame, keypoints, bbox, color, show_labels)
                    pose_label = interpret_pose(keypoints)
                    pose_info[track_id] = pose_label
                
                self.draw_enhanced_bounding_box(vis_frame, bbox, track_id, color, confidence, pose_label)
            
            # Calculate FPS
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Draw info panel
            self.draw_advanced_info_panel(vis_frame, frame_count, current_fps, len(tracks), pose_info, {})
            
            # Display frame
            cv2.imshow('Advanced Real-time Tracking', vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Advanced webcam processing stopped.")


def main():
    """Main function to demonstrate advanced tracking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced People Tracking with Better Models")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, help="Path to output video file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam instead of video file")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to process")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], 
                       help="Device to use for inference")
    parser.add_argument("--no-labels", action="store_true", help="Hide keypoint labels")
    
    args = parser.parse_args()
    
    # Initialize advanced tracker
    print("🚀 Initializing Advanced People Tracker...")
    tracker = AdvancedPeopleTracker(device=args.device)
    
    try:
        if args.webcam:
            print("📹 Starting advanced webcam tracking...")
            tracker.process_webcam(max_frames=args.max_frames, show_labels=not args.no_labels)
        elif args.video:
            if not os.path.exists(args.video):
                print(f"❌ Video file not found: {args.video}")
                return
            
            print(f"📹 Processing video: {args.video}")
            tracker.process_video(args.video, args.output, args.max_frames, show_labels=not args.no_labels)
        else:
            # Default: use sample video
            sample_video = "data/videoplayback.mp4"
            if os.path.exists(sample_video):
                print(f"📹 Using sample video: {sample_video}")
                output_path = "data/advanced_tracking_output.mp4"
                tracker.process_video(sample_video, output_path, args.max_frames, show_labels=not args.no_labels)
            else:
                print("❌ No video specified and no sample video found.")
                print("Usage examples:")
                print("  python advanced_tracking.py --video path/to/video.mp4")
                print("  python advanced_tracking.py --webcam")
                print("  python advanced_tracking.py --video input.mp4 --output output.mp4 --no-labels")
    
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
