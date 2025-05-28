from fastapi import HTTPException

class AnalyzerException(HTTPException):
    """Base exception for analyzer module"""
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)

class InvalidInputException(AnalyzerException):
    """Exception for invalid input data"""
    def __init__(self, detail: str = "Invalid input data"):
        super().__init__(status_code=400, detail=detail)

class AnalysisFailedException(AnalyzerException):
    """Exception for when analysis fails"""
    def __init__(self, detail: str = "Analysis failed"):
        super().__init__(status_code=500, detail=detail)
