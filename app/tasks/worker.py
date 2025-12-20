import logging
import os
import httpx 
import json
from ..core.celery_app import celery_app
from ..services.transcribe_service import transcribe, transcribe2srt
from ..services.translation_serivce import translate_text, translate_segments 
from ..core.config import settings

logger = logging.getLogger(__name__)

def _send_webhook(payload: dict):
    """
    Hàm gửi callback về Backend khác với payload tùy chỉnh.
    """
    callback_url = settings.WEBHOOK_CALLBACK_URL
    if not callback_url:
        logger.warning("No WEBHOOK_CALLBACK_URL configured. Skipping callback.")
        return

    try:
        logger.info(f"Sending webhook to {callback_url} with payload keys: {payload.keys()}")
        with httpx.Client(timeout=10.0) as client:
            response = client.post(callback_url, json=payload)
            response.raise_for_status()
            logger.info(f"Webhook sent successfully. Response: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to send webhook: {e}")

@celery_app.task(bind=True, name="transcribe_task")
def task_transcribe(self, file_path: str, model_id: str, type: str, source_id: int):
    try:
        logger.info(f"Processing transcribe task for: {file_path} (ID: {source_id})")
        
        transcribe_res = transcribe(audio_path=file_path, model_id=model_id)
        
        key_id = "movieSourceID"
        if type == "episode":
             key_id = "episodeSourceID" 
        
        payload = {
            key_id: source_id,
            "srt": "", 
            "language": transcribe_res.language,
            "raw_segments": transcribe_res.segments, 
            "text": transcribe_res.text 
        }
        
        _send_webhook(payload)
        
        if os.path.exists(file_path):
            os.remove(file_path)

        return payload
        
    except Exception as e:
        logger.error(f"Error in transcribe task: {e}")
        raise self.retry(exc=e, countdown=10, max_retries=3)

@celery_app.task(bind=True, name="transcribe_srt_task")
def task_transcribe_srt(self, file_path: str, model_id: str, vad_threshold: float, type: str, source_id: int):
    try:
        logger.info(f"Processing SRT task for: {file_path} (ID: {source_id})")
        
        srt_str, segments, language, raw_segments = transcribe2srt(
            audio_path=file_path, 
            model_id=model_id,
            vad_threshold=vad_threshold
        )

        key_id = "movieSourceID"
        if type == "episode":
             key_id = "episodeSourceID" 

        payload = {
            key_id: source_id,
            "srt": srt_str,
            "language": language,
            "raw_segments": raw_segments if raw_segments else segments 
        }


        _send_webhook(payload)

        if os.path.exists(file_path):
            os.remove(file_path)

        return payload

    except Exception as e:
        logger.error(f"Error in transcribe SRT task: {e}")
        error_payload = {
             "movieSourceID": source_id,
             "error": str(e),
             "status": "FAILED"
        }
        _send_webhook(error_payload)
        
        raise self.retry(exc=e, countdown=10, max_retries=3)


@celery_app.task(bind=True, name="translate_text_task")
def task_translate_text(self, texts: list[str], model_id: str, type: str, source_id: str):
    """
    Task dịch danh sách câu văn.
    """
    try:
        logger.info(f"Processing translate text task. Items: {len(texts)}")
        results = translate_text(inputs=texts, model_id=model_id)
        key_id = "movieSourceID"
        if type == "episode":
            key_id = "episodeSourceID"
        payload = {
            key_id: source_id,
            "translations": results
        }
        _send_webhook(payload)
        return payload
    except Exception as e:
        logger.error(f"Error in translate text task: {e}")
        _send_webhook(self.request.id, "FAILURE", error=str(e))
        raise self.retry(exc=e, countdown=5, max_retries=3)

@celery_app.task(bind=True, name="translate_segments_task")
def task_translate_segments(self, segments: list[dict], language: str, model_id: str, type: str, source_id: str):
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
        key_id = "movieSourceID"
        if type == "episode":
            key_id = "episodeSourceID"
        payload = {
            key_id: source_id,
            "srt": srt_str,
            "segments": translated_segments
        }
        _send_webhook(payload)
        return payload
    except Exception as e:
        logger.error(f"Error in translate segments task: {e}")
        _send_webhook(self.request.id, "FAILURE", error=str(e))
        raise self.retry(exc=e, countdown=5, max_retries=3)