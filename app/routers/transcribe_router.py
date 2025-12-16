from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from celery.result import AsyncResult
import shutil
import os
import uuid
from ..core.config import settings
from ..tasks.worker import task_transcribe, task_transcribe_srt
from ..utils.file_utils import ensure_dir

router = APIRouter(prefix="/transcribe", tags=["Transcribe"])

# Đảm bảo thư mục temp tồn tại
TEMP_DIR = os.path.join(settings.STORAGE_DIR, "temp_uploads")
ensure_dir(TEMP_DIR)

def save_upload_file(upload_file: UploadFile) -> str:
    """Lưu file upload xuống đĩa và trả về đường dẫn tuyệt đối"""
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(upload_file.filename)[1]
    filename = f"{file_id}{ext}"
    file_path = os.path.join(TEMP_DIR, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return file_path

@router.post("/process")
async def create_transcribe_task(
    file: UploadFile = File(...),
    model_id: str = Form("turbo"),
    output_format: str = Form("json", description="json or srt")
):
    """
    Upload file và tạo task xử lý background.
    Trả về task_id để tracking.
    """
    try:
        # 1. Lưu file xuống đĩa (Worker cần đường dẫn file thực tế)
        file_path = save_upload_file(file)
        
        # 2. Gửi task vào Celery Queue
        if output_format == "srt":
            task = task_transcribe_srt.delay(file_path, model_id, 0.2)
        else:
            task = task_transcribe.delay(file_path, model_id)
            
        return {
            "task_id": task.id,
            "status": "processing",
            "message": "File uploaded and transcription started."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Kiểm tra trạng thái task dựa vào task_id.
    """
    task_result = AsyncResult(task_id)
    
    result = {
        "task_id": task_id,
        "status": task_result.status,
        "result": None
    }
    
    if task_result.status == 'SUCCESS':
        result["result"] = task_result.result
    elif task_result.status == 'FAILURE':
        result["error"] = str(task_result.result)
        
    return JSONResponse(content=result)