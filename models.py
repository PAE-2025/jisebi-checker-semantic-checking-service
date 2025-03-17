from pydantic import BaseModel
from typing import List

class JournalSearchRequest(BaseModel):
    title: str
    abstract: str

class JournalResult(BaseModel):
    scopus_id: str
    title: str
    abstract: str
    similarity: float
    common_keywords: List[str]
