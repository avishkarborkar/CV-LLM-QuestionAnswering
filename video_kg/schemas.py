from pydantic import BaseModel
from typing import Dict, List

class Entity(BaseModel):
    type: str
    bbox: List[int]
    cropped_img: str  # Base64

class FrameData(BaseModel):
    timestamp: float
    entities: Dict[str, Entity]  # {track_id: Entity}

class KnowledgeGraph(BaseModel):
    metadata: Dict
    frames: List[FrameData]