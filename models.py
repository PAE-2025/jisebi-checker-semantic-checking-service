from pydantic import BaseModel
from typing import List

class JournalSearchRequest(BaseModel):
    title: str
    abstract: str

class JournalResult(BaseModel):
    doi: str
    title: str
    abstract: str
    similarity: float
    common_keywords: List[str]

class SearchResponse(BaseModel):
    query: str
    num_results: int
    journals: List[JournalResult]
