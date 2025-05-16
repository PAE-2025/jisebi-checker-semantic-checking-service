import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import search
from src.discon_analyzer.router import router as discon_analyzer_router
from src.ner.router import router as ner_router
from src.grammar_checking.router import router as grammar_router
from src.config import Settings
from src.core.requests.authentication_service import AuthService
from src.core.middleware.auth import AuthenticationMiddleware

settings = Settings()
# Rest of the file remains unchanged
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Journal Semantic Checking API", description="Analyze journal content for various semantic features")

# Create auth service instance
auth_service = AuthService(settings)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Authentication middleware
app.add_middleware(
    AuthenticationMiddleware,
    auth_service=auth_service,
    exclude_paths=["/docs", "/redoc", "/openapi.json", "/health", "/metrics", "/"]
)

# Register Routers
app.include_router(search.router, prefix="/api/novelty", tags=["Search"])
app.include_router(discon_analyzer_router, prefix="/api/discon", tags=["Discon"])
app.include_router(ner_router, prefix="/api/ner", tags=["NER"])
app.include_router(grammar_router, prefix="/api/grammar", tags=["Grammar"])

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Journal Semantic Checking API", 
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}