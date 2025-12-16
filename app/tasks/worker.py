import logging
import os
from ..core.celery_app import celery_app
from ..services.transcribe_service import transcribe, transcribe2srt

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name="transcribe_task")
def task_transcribe(self, file_path: str, model_id: str):
    """
    Task xử lý Transcribe trả về JSON/Text.
    """
    try:
        logger.info(f"Processing transcribe task for: {file_path}")
        result = transcribe(audio_path=file_path, model_id=model_id)
        
        if os.path.exists(file_path):
            os.remove(file_path)


        return result.model_dump()
        
    except Exception as e:
        logger.error(f"Error in transcribe task: {e}")
        raise self.retry(exc=e, countdown=10, max_retries=3)

@celery_app.task(bind=True, name="transcribe_srt_task")
def task_transcribe_srt(self, file_path: str, model_id: str, vad_threshold: float):
    """
    Task xử lý Transcribe trả về SRT.
    """
    try:
        logger.info(f"Processing SRT task for: {file_path}")
        
        srt_str, segments, language, raw_segments = transcribe2srt(
            audio_path=file_path, 
            model_id=model_id,
            vad_threshold=vad_threshold
        )

        # Cleanup file
        if os.path.exists(file_path):
            os.remove(file_path)

        # Trả về dict kết quả
        return {
            "srt": srt_str,
            "language": language,
            "segments": segments,
            "raw_segments": raw_segments
        }
    except Exception as e:
        logger.error(f"Error in transcribe SRT task: {e}")
        raise self.retry(exc=e, countdown=10, max_retries=3)