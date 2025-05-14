# src/ner/router.py
from fastapi import APIRouter
from .service import run_ner
from .schemas import NERRequest, NERResponse

router = APIRouter(
    tags= ["NER"],
    responses={404: {"description": "Not found"}},
)

@router.post("/analyze", response_model=NERResponse)
async def recognize_entities(payload: NERRequest):
    entities = run_ner(payload.text)
    return {"text": payload.text, "entities": entities}
