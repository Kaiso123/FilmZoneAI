import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_TRANSCRIBE_SIZE: str = "turbo"  # default whisper model size
    TRANSLATION_MODEL_ID: str = "envit5-translation" #default translation model
    TRANSCRIBE_DIR: str = "transcribe"
    TRANSLATE_DIR: str = "translate"
    STORAGE_DIR: str = os.path.join(os.getcwd(), "storage", "models")
    # GEMINI_API_KEY: str
    SECRET_KEY: str = "A_VERY_LONG_SECRET_KEY_FOR_JWT_TOKEN"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 600
    ALGORITHM: str = "HS256"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    # Config Redis cho Celery
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

    class Config:
        env_file = ".env"

settings = Settings()