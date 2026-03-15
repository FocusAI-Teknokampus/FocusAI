# backend/core/schemas.py
from pydantic import BaseModel
from datetime import datetime

class FrameData(BaseModel):
    timestamp: datetime
    gaze_on_screen: bool       
    hand_on_chin: bool         
    emotion_bored: bool        
    ear_score: float