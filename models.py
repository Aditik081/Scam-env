from pydantic import BaseModel
from typing import Optional

class Observation(BaseModel):
    text: str
    task: str

class Action(BaseModel):
    label: str  # "scam" or "safe"

class Reward(BaseModel):
    score: float