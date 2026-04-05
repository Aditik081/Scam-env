from pydantic import BaseModel

class Observation(BaseModel):
    text: str
    has_link: bool
    has_urgent_words: bool

class Action(BaseModel):
    action: str