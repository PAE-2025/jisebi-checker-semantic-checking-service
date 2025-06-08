from fastapi import FastAPI
from src.novelty_checker.router import router as novelty_router
from src.discon_analyzer.router import router as discon_analyzer_router
from src.ner.router import router as ner_router
from src.grammar_checking.router import router as grammar_router
from src.abstract_checking.router import router as abstract_router

app = FastAPI(title="Journal Search API", description="Find similar journals using Scopus API")

# Register Routers
app.include_router(novelty_router.router, prefix="/api/novelty", tags=["Search"])
app.include_router(discon_analyzer_router, prefix="/api/discon", tags=["Discon"])
app.include_router(ner_router, prefix="/api/ner", tags=["NER"])
app.include_router(grammar_router, prefix="/api/grammar", tags=["Grammar"])
app.include_router(abstract_router, prefix="/api/abstract", tags=["Abstract"])

@app.get("/")
async def root():

    return {
        "message": "Welcome to the Journal Search API", 
        "docs" : "/docs"
    }

