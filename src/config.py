import json
# Load konfigurasi dari file JSON
with open("config.json") as con_file:
    config = json.load(con_file)

from pydantic_settings import BaseSettings
    
class Settings(BaseSettings):
    APP_NAME: str = "Jisebi Checker"
    DEBUG_MODE: bool = True
    SCOPUS_API_KEY = config["apikey"]
    SPACY_MODEL: str = "en_core_web_sm"
    
    # API settings
    API_PREFIX: str = "/api/v1"
    
settings = Settings()
