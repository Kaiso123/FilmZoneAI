from pathlib import Path  
from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile, shutil, os
from ..services.transcribe_service import transcribe, transcribe2srt
from ..models.transcibeModels import TranscribeResponse, TranscribeToSRTResponse


router = APIRouter(prefix="/transcribe", tags=["transcription"])


@router.post("/audio_video")
async def transcribe_endpoint(file : UploadFile = File(...)):
    # lưu tạm
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        with open(tmp.name, "wb") as f:
            shutil.copyfileobj(file.file, f)
        try:
            result = transcribe(tmp.name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return result
    finally:
        try:
            tmp.close()
            import os
            os.remove(tmp.name)
        except Exception:
            pass

@router.post("/audio_video_2_srt")
async def transcribe_file_to_srt(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    try:
        with open(tmp.name, "wb") as f:
            shutil.copyfileobj(file.file, f)
        try:
            srt_text, segments, language, raw_segments = transcribe2srt(tmp.name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return TranscribeToSRTResponse(srt=srt_text, segments=segments, language=language, raw_segments=raw_segments)
    finally:
        try:
            tmp.close()
            os.remove(tmp.name)
        except Exception:
            pass

