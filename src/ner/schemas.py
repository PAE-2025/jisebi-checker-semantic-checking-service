# src/ner/schemas.py
from pydantic import BaseModel
from typing import List, Optional, Union

class NERRequest(BaseModel):
    text: Union[List[str], str]

class Entity(BaseModel):
    entity: str
    word: str
    start: int
    end: int
    score: float

class NERResponse(BaseModel):
    text: str
    entities: List[Entity]
