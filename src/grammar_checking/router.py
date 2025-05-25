from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from src.grammar_checking.schemas import TextInput, TextOutput
from src.grammar_checking.service import process_text

router = APIRouter()

@router.post(
    "/process-text",
    status_code=status.HTTP_200_OK,
    description="Process text for grammar correction and typo detection with underlined typos",
    summary="Grammar Check and Typo Detection"
)
async def process_text_endpoint(input_text: TextInput):
    try:
        result = await process_text(input_text.text)
        return JSONResponse(
            status_code=200,
            content={
                "status": True,
                "data": result
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))