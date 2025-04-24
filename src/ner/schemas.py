# src/ner/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class NERRequest(BaseModel):
    text: str

class Entity(BaseModel):
    entity: str
    word: str
    start: int
    end: int
    score: float

class NERResponse(BaseModel):
    text: str
    entities: List[Entity]
