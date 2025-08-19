#!/usr/bin/env python3
"""
Visualization script for people tracking with bounding boxes and pose keypoints.
This script demonstrates the system's ability to detect, track, and analyze human poses.
"""

import cv2
import numpy as np
import os
from vision.yolo_detector import YOLODetector
from vision.tracker import DeepSortTracker
from vision.pose_estimator import PoseEstimator
from vision.pose_logic import interpret_pose

class PeopleTrackingVisualizer:
    def __init__(self, model_path="yolov8n.pt", device="cpu"):
        """Initialize the tracking visualizer with detection and pose models."""
        self.detector = YOLODetector(model_path=model_path, device=device)
        self.tracker = DeepSortTracker()
        self.pose_estimator = PoseEstimator()
        
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
        ]
        
        print("✅ PeopleTrackingVisualizer initialized successfully!")
    
    def draw_bounding_box(self, frame, bbox, track_id, color, confidence=None):
        """Draw a bounding box with track ID and optional confidence."""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label with track ID and confidence
        label = f"ID: {str(track_id)}"
        if confidence:
            label += f" ({float(confidence):.2f})"
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_pose_keypoints(self, frame, keypoints, bbox, color):
        """Draw pose keypoints and skeleton connections."""
        if not keypoints:
            return
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Define skeleton connections (pairs of keypoint indices)
        skeleton = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Head to right hand
            (1, 5), (5, 6), (6, 7),          # Head to left hand
            (1, 8), (8, 9), (9, 10),         # Head to right foot
            (1, 11), (11, 12), (12, 13),     # Head to left foot
            (0, 14), (14, 16),               # Nose to right eye
            (0, 15), (15, 17),               # Nose to left eye
        ]
        
        # Draw keypoints
        for i, (x, y) in enumerate(keypoints):
            if x > 0 and y > 0:  # Valid keypoint
                # Adjust coordinates relative to bounding box
                abs_x = int(x) + x1
                abs_y = int(y) + y1
                
                # Draw keypoint circle
                cv2.circle(frame, (abs_x, abs_y), 4, color, -1)
                
                # Draw keypoint number (for debugging)
                cv2.putText(frame, str(i), (abs_x + 5, abs_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw skeleton connections
        for start_idx, end_idx in skeleton:
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                keypoints[start_idx][0] > 0 and keypoints[start_idx][1] > 0 and
                keypoints[end_idx][0] > 0 and keypoints[end_idx][1] > 0):
                
                start_x = int(keypoints[start_idx][0]) + x1
                start_y = int(keypoints[start_idx][1]) + y1
                end_x = int(keypoints[end_idx][0]) + x1
                end_y = int(keypoints[end_idx][1]) + y1
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)
    
    def draw_info_panel(self, frame, frame_count, fps, num_tracks, pose_info):
        """Draw information panel on the frame."""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text information
        cv2.putText(frame, f"Frame: {frame_count}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"People Tracked: {num_tracks}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw pose information
        y_offset = 110
        for track_id, pose_label in pose_info.items():
            if y_offset < 140:  # Limit to prevent overflow
                cv2.putText(frame, f"ID {str(track_id)}: {str(pose_label)}", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
    
    def process_video(self, video_path, output_path=None, max_frames=None):
        """Process video and create visualization with tracking and pose estimation."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video Info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Output will be saved to: {output_path}")
        
        frame_count = 0
        start_time = cv2.getTickCount()
        
        print("🎬 Starting video processing...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if max_frames and frame_count > max_frames:
                break
            
            # Create a copy for visualization
            vis_frame = frame.copy()
            
            # Detect objects
            detections = self.detector.detect(frame)
            
            # Filter for people only
            people_detections = [det for det in detections if det.get('class_name') == 'person']
            
            # Track people
            tracks = self.tracker.update(people_detections, frame)
            
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
                
                # Draw bounding box
                self.draw_bounding_box(vis_frame, bbox, track_id, color, confidence)
                
                # Extract person crop for pose estimation
                x1, y1, x2, y2 = map(int, bbox)
                person_crop = frame[y1:y2, x1:x2]
                
                if person_crop.size > 0:
                    # Estimate pose
                    keypoints = self.pose_estimator.estimate(person_crop)
                    
                    # Draw pose keypoints and skeleton
                    self.draw_pose_keypoints(vis_frame, keypoints, bbox, color)
                    
                    # Interpret pose
                    pose_label = interpret_pose(keypoints)
                    pose_info[track_id] = pose_label
            
            # Calculate current FPS
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Draw information panel
            self.draw_info_panel(vis_frame, frame_count, current_fps, len(tracks), pose_info)
            
            # Write frame if output is specified
            if writer:
                writer.write(vis_frame)
            
            # Display frame (optional - comment out if running headless)
            cv2.imshow('People Tracking with Pose Estimation', vis_frame)
            
            # Print progress
            if frame_count % 30 == 0:  # Every 30 frames
                print(f"📊 Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Processing complete! Processed {frame_count} frames.")
        if output_path:
            print(f"💾 Output saved to: {output_path}")
    
    def process_webcam(self, max_frames=None):
        """Process webcam feed for real-time visualization."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Could not open webcam")
        
        print("📹 Webcam started - Press 'q' to quit")
        
        frame_count = 0
        start_time = cv2.getTickCount()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if max_frames and frame_count > max_frames:
                break
            
            # Create a copy for visualization
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
                
                color = self.colors[track_id % len(self.colors)]
                self.draw_bounding_box(vis_frame, bbox, track_id, color, confidence)
                
                # Pose estimation
                x1, y1, x2, y2 = map(int, bbox)
                person_crop = frame[y1:y2, x1:x2]
                
                if person_crop.size > 0:
                    keypoints = self.pose_estimator.estimate(person_crop)
                    self.draw_pose_keypoints(vis_frame, keypoints, bbox, color)
                    pose_label = interpret_pose(keypoints)
                    pose_info[track_id] = pose_label
            
            # Calculate FPS
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Draw info panel
            self.draw_info_panel(vis_frame, frame_count, current_fps, len(tracks), pose_info)
            
            # Display frame
            cv2.imshow('Real-time People Tracking', vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Webcam processing stopped.")


def main():
    """Main function to demonstrate the visualizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="People Tracking with Pose Estimation Visualizer")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, help="Path to output video file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam instead of video file")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to process")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], 
                       help="Device to use for inference")
    
    args = parser.parse_args()
    
    # Initialize visualizer
    print("🚀 Initializing People Tracking Visualizer...")
    visualizer = PeopleTrackingVisualizer(device=args.device)
    
    try:
        if args.webcam:
            print("📹 Starting webcam visualization...")
            visualizer.process_webcam(max_frames=args.max_frames)
        elif args.video:
            if not os.path.exists(args.video):
                print(f"❌ Video file not found: {args.video}")
                return
            
            print(f"📹 Processing video: {args.video}")
            visualizer.process_video(args.video, args.output, args.max_frames)
        else:
            # Default: use sample video if available
            sample_video = "data/videoplayback.mp4"
            if os.path.exists(sample_video):
                print(f"📹 Using sample video: {sample_video}")
                output_path = "data/tracking_visualization.mp4"
                visualizer.process_video(sample_video, output_path, args.max_frames)
            else:
                print("❌ No video specified and no sample video found.")
                print("Usage examples:")
                print("  python visualize_tracking.py --video path/to/video.mp4")
                print("  python visualize_tracking.py --webcam")
                print("  python visualize_tracking.py --video input.mp4 --output output.mp4")
    
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
