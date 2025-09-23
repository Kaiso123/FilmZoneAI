from datetime import timedelta
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.security import create_access_token
from .core.config import settings
from .routers.transcribe_router import router as transcribe_router
from .routers.translate_router import router as translate_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(
    title="AI Transcription Service",
    description="API for transcribing audio and video files, and translating text and segments.",
    version="1.0.0",
)

#test
access_token = create_access_token(
    data={"sub": "test_user"}, 
    expires_delta=timedelta(minutes=30000)
)
print("Access Token:", access_token)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# mount routers
app.include_router(transcribe_router)
app.include_router(translate_router)

@app.get("/")
async def root():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
    )