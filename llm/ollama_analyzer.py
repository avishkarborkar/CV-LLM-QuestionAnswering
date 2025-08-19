# llm/ollama_analyzer.py
import requests
from typing import Any, Dict

# Import the new pose interpreter
from vision.pose_logic import interpret_pose


class OllamaAnalyzer:
    def __init__(self, model: str = "gemma:2b", base_url: str = "http://localhost:11434"):
        # ... (init method is the same, make sure gemma:2b is pulled)
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        print(f"✅ OllamaAnalyzer initialized with model: {self.model}")

    def analyze(self, kg: Any, question: str) -> str:
        """Analyzes the knowledge graph to answer a user's question."""
        if not hasattr(kg, 'frames') or not kg.frames:
            return "Error: The knowledge graph is empty."

        context = self._summarize_kg_for_llm(kg)
        prompt = self._build_prompt(context, question)

        print("\nQuerying Ollama with the following structured context:")
        print("-" * 20)
        print(context)
        print("-" * 20)

        # ... (the API call logic is the same)
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}},
                timeout=90
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.RequestException as e:
            return f"🚨 Ollama API call failed: {e}"

    def _summarize_kg_for_llm(self, kg: Any) -> str:
        """Creates a human-readable text summary of the knowledge graph for the LLM."""
        summary_parts = ["## Video Analysis Report ##"]

        for frame_data in kg.frames:
            summary_parts.append(f"\n--- Frame at {frame_data.timestamp:.2f} seconds ---")

            # Add brightness info
            brightness = frame_data.scene_brightness
            if brightness is not None:
                lighting = "Daytime" if brightness > 100 else "Nighttime"
                summary_parts.append(f"Scene Lighting: {lighting} (Brightness score: {brightness:.0f})")

            # Add info for each entity
            if not frame_data.entities:
                summary_parts.append("No objects detected in this frame.")
                continue

            summary_parts.append("Objects Present:")
            for track_id, entity in frame_data.entities.items():
                desc = f"- {entity.type.capitalize()} (ID: {track_id})"

                # Add color info
                if entity.dominant_color:
                    # Simple color naming can be added here, but for now, just list RGB
                    desc += f" | Dominant Color (RGB): {entity.dominant_color}"

                # Add interpreted pose info
                if entity.type == 'person':
                    pose_label = interpret_pose(entity.pose_keypoints)
                    desc += f" | Pose: {pose_label}"

                summary_parts.append(desc)

        return "\n".join(summary_parts)

    def _build_prompt(self, context: str, question: str) -> str:
        """Constructs the final prompt for the LLM."""
        return (
            "You are a meticulous video analysis expert. Your task is to answer a question based on the following 'Video Analysis Report'. "
            "Use only the information provided in the report.\n\n"
            f"{context}\n\n"
            "## User Question ##\n"
            f"Based strictly on the report above, answer this question: {question}\n\n"
            "## Answer ##\n"
        )