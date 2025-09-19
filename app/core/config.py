import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_TRANSCRIBE_SIZE: str = "turbo"  # default whisper model size
    TRANSLATION_MODEL_ID: str = "envit5-translation" #default translation model
    TRANSCRIBE_DIR: str = "transcribe"
    TRANSLATE_DIR: str = "translate"
    STORAGE_DIR: str = os.path.join(os.getcwd(), "storage", "models")

    class Config:
        env_file = ".env"

settings = Settings()