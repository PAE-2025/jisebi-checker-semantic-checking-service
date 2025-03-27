from fastapi import Depends
from src.discon_analyzer.service import AnalyzerService

# Singleton instance of the analyzer service
_analyzer_service = None

def get_analyzer_service():
    """
    Dependency to get the analyzer service instance
    Uses a singleton pattern to avoid loading the NLP models multiple times
    """
    global _analyzer_service
    if _analyzer_service is None:
        _analyzer_service = AnalyzerService()
    return _analyzer_service
