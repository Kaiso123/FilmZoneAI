from fastapi import APIRouter, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from celery.result import AsyncResult
from ..models.translateModels import TranslateRequest, TranslateSegmentsRequest
from ..tasks.worker import task_translate_text, task_translate_segments
from .dependencies import get_token_header

router = APIRouter(prefix="/translate", 
                   tags=["translation"], 
                   dependencies=[Depends(get_token_header)], 
                   responses={401: {"description": "Unauthorized"}})

@router.post("/text")
async def translate_text_endpoint(
    request: TranslateRequest,
    type: str = Form(..., description="movie | episode"),
    source_id: str = Form(..., description="movie_id or episode_id"), 
    ):
    """
    Gửi yêu cầu dịch văn bản vào hàng đợi.
    """
    try:
        # Đẩy list text vào Redis queue
        task = task_translate_text.delay(request.texts, "VietAI/envit5-translation", type, source_id)
        
        return {
            "task_id": task.id,
            "status": "processing",
            "message": "Translation task started."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/segments")
async def translate_segments_endpoint(
    request: TranslateSegmentsRequest,
    type: str = Form(..., description="movie | episode"),
    source_id: str = Form(..., description="movie_id or episode_id"),
    ):
    """
    Gửi yêu cầu dịch segments vào hàng đợi.
    """
    try:
        task = task_translate_segments.delay(
            request.segments, 
            request.language, 
            "VietAI/envit5-translation",
            type,
            source_id
        )
        
        return {
            "task_id": task.id,
            "status": "processing",
            "message": "Segment translation task started."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{task_id}")
async def get_translate_status(task_id: str):
    """
    Kiểm tra trạng thái task dịch thuật.
    """
    task_result = AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "status": task_result.status,
        "result": None
    }
    
    if task_result.status == 'SUCCESS':
        response["result"] = task_result.result
    elif task_result.status == 'FAILURE':
        response["error"] = str(task_result.result)
        
    return JSONResponse(content=response)