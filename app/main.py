import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from routers.transcribe_router import router as transcribe_router
from routers.translate_router import router as translate_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(
    title="AI Transcription Service",
)

# mount routers
app.include_router(transcribe_router)
app.include_router(translate_router)

@app.get("/")
async def root():
    return {"status": "ok"}