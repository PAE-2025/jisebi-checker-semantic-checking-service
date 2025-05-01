from fastapi import FastAPI
from routers import search

app = FastAPI(title="Journal Search API", description="Find similar journals using Scopus API")

# Register Routers
app.include_router(search.router, prefix="/api", tags=["Search"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Journal Search API"}