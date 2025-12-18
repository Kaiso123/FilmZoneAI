import logging
import os
from ..core.celery_app import celery_app
from ..services.transcribe_service import transcribe, transcribe2srt
from ..services.translation_serivce import translate_text, translate_segments 

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
    

@celery_app.task(bind=True, name="translate_text_task")
def task_translate_text(self, texts: list[str], model_id: str):
    """
    Task dịch danh sách câu văn.
    """
    try:
        logger.info(f"Processing translate text task. Items: {len(texts)}")
        results = translate_text(inputs=texts, model_id=model_id)
        return {"translations": results}
    except Exception as e:
        logger.error(f"Error in translate text task: {e}")
        raise self.retry(exc=e, countdown=5, max_retries=3)

@celery_app.task(bind=True, name="translate_segments_task")
def task_translate_segments(self, segments: list[dict], language: str, model_id: str):
    """
    Task dịch segments (dùng cho phụ đề).
    """
    try:
        logger.info(f"Processing translate segments task. Items: {len(segments)}")
        srt_str, translated_segments = translate_segments(
            segments=segments, 
            language=language, 
            model_id=model_id
        )
        return {
            "srt": srt_str,
            "segments": translated_segments
        }
    except Exception as e:
        logger.error(f"Error in translate segments task: {e}")
        raise self.retry(exc=e, countdown=5, max_retries=3)