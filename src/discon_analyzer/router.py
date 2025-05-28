from fastapi import APIRouter, HTTPException, Depends
from src.discon_analyzer.schemas import PaperInput, PaperAnalysisResponse
from src.discon_analyzer.service import AnalyzerService
from src.discon_analyzer.dependencies import get_analyzer_service
from src.config import settings

router = APIRouter(
    responses={404: {"description": "Not found"}},
)

@router.post(
    "/analyze-paper", 
    response_model=PaperAnalysisResponse
)
async def analyze_paper(
    paper: PaperInput,
    analyzer_service: AnalyzerService = Depends(get_analyzer_service)
):
    """
    Analyze a single paper's discussion and conclusion sections
    """
    try:
        result = analyzer_service.analyze_paper({
            "discussion": paper.discussion,
            "conclusion": paper.conclusion
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
