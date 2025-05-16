from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache
    
class Settings(BaseSettings):
    APP_NAME: str = "Jisebi Checker"
    DEBUG_MODE: bool = True
    SCOPUS_API_KEY: str = ""  # Berikan nilai default
    SPACY_MODEL: str = "en_core_web_sm"
    AUTH_SERVICE_URL: str = "http://localhost:8001/api/auth/validate"
    SELF_URL: str = "http://localhost:8000"
    
    # API settings
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    class Config:
        env_file = ".env"
    
@lru_cache()
def get_settings():
    return Settings()

# Ekspor settings sebagai variabel untuk backward compatibility
settings = Settings()
