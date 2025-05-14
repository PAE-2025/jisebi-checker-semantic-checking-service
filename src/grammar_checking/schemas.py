from pydantic import BaseModel
from typing import List, Dict, Union

class TextInput(BaseModel):
    text: Union[str, List[str]]

class TextOutput(BaseModel):
    original: str
    corrected: str
    highlighted_typos: List[Dict[str, str]]