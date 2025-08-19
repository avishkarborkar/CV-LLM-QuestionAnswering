#!/usr/bin/env python3
"""
Demo script for the People Tracking Visualization.
This script provides a simple interface to test the tracking and pose estimation.
"""

import os
import sys
from visualize_tracking import PeopleTrackingVisualizer

def demo_with_sample_video():
    """Demo using the sample video if available."""
    sample_video = "data/videoplayback.mp4"
    
    if not os.path.exists(sample_video):
        print(f"❌ Sample video not found: {sample_video}")
        print("Please ensure you have a video file to test with.")
        return False
    
    print("🎬 Starting People Tracking Demo...")
    print(f"📹 Using video: {sample_video}")
    print("🎯 Features demonstrated:")
    print("  - Person detection with YOLO")
    print("  - Object tracking with DeepSORT")
    print("  - Pose estimation with keypoints")
    print("  - Real-time visualization")
    print("  - Pose interpretation")
    print()
    
    try:
        # Initialize visualizer
        visualizer = PeopleTrackingVisualizer(device="cpu")
        
        # Process video with output
        output_path = "data/demo_tracking_output.mp4"
        visualizer.process_video(
            video_path=sample_video,
            output_path=output_path,
            max_frames=300  # Process first 300 frames for demo
        )
        
        print(f"✅ Demo completed! Output saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def demo_webcam():
    """Demo using webcam for real-time tracking."""
    print("📹 Starting Webcam Demo...")
    print("🎯 Real-time features:")
    print("  - Live person detection")
    print("  - Real-time tracking")
    print("  - Live pose estimation")
    print("  - Press 'q' to quit")
    print()
    
    try:
        visualizer = PeopleTrackingVisualizer(device="cpu")
        visualizer.process_webcam(max_frames=1000)  # Run for up to 1000 frames
        return True
        
    except Exception as e:
        print(f"❌ Webcam demo failed: {e}")
        return False

def main():
    """Main demo function with user choice."""
    print("🎬 People Tracking & Pose Estimation Demo")
    print("=" * 50)
    print()
    print("Choose demo type:")
    print("1. Video file demo (using sample video)")
    print("2. Webcam demo (real-time)")
    print("3. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == "1":
                print("\n" + "="*50)
                demo_with_sample_video()
                break
            elif choice == "2":
                print("\n" + "="*50)
                demo_webcam()
                break
            elif choice == "3":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break

if __name__ == "__main__":
    main()

