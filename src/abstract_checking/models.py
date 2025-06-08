from pydantic import BaseModel
from typing import List

# Model untuk input artikel manual
class ArticleInput(BaseModel):
    abstract: str
    keywords: str