from pydantic_settings import BaseSettings
    
class Settings(BaseSettings):
    APP_NAME: str = "Jisebi Checker"
    DEBUG_MODE: bool = True
    SCOPUS_API_KEY: str
    SPACY_MODEL: str = "en_core_web_sm"
    
    # API settings
    API_PREFIX: str = "/api/v1"

    class Config:
        env_file = ".env"
    
settings = Settings()
