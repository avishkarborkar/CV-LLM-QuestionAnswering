#!/usr/bin/env python3
"""
Example usage of the Vision LLM Video QA system.
"""

import os
from analyze_video import analyze_video

def main():
    """Demonstrate the video analysis system with example questions."""
    
    # Example video path (update this to your video file)
    video_path = "data/videoplayback.mp4"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("Please update the video_path variable to point to your video file.")
        return
    
    # Example questions to demonstrate different capabilities
    example_questions = [
        "How many people are visible in the video?",
        "What is the dominant color of the person on the left?",
        "Describe the pose of the person in the center",
        "Who is sitting and who is standing?",
        "What is the spatial relationship between the two people?",
        "Is the scene well-lit or dark?",
        "Describe what you see in the video in detail"
    ]
    
    print("🎬 Vision LLM Video QA - Example Analysis")
    print("=" * 50)
    
    for i, question in enumerate(example_questions, 1):
        print(f"\n📝 Question {i}: {question}")
        print("-" * 40)
        
        try:
            # Analyze the video with the current question
            result = analyze_video(
                video_path=video_path,
                question=question,
                max_seconds=10  # Analyze first 10 seconds
            )
            
            print(f"🤖 Answer: {result}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 40)
    
    print("\n✅ Analysis complete!")
    print("\n💡 Tips:")
    print("- Make sure Ollama is running: 'ollama serve'")
    print("- Ensure you have the gemma:2b model: 'ollama pull gemma:2b'")
    print("- For faster processing, use GPU if available")
    print("- Adjust max_seconds parameter for longer/shorter analysis")

if __name__ == "__main__":
    main()
