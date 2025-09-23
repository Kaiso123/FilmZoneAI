from fastapi import APIRouter, HTTPException, Body, Depends
from ..services.translation_serivce import translate_text, translate_segments
from ..models.translateModels import TranslateRequest, TranslateResponse, TranslateSegmentSrt, TranslateSegmentsRequest
from .dependencies import get_token_header

router = APIRouter(prefix="/translate", 
                   tags=["translation"], 
                   dependencies=[Depends(get_token_header)], 
                   responses={401: {"description": "Unauthorized"}})


@router.post("/text", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    try:
        outputs = translate_text(request.texts)
        return TranslateResponse(translations=outputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/segments", response_model=TranslateSegmentSrt)
async def translate_segmentss(request: TranslateSegmentsRequest):
    try:
        srt_str, segments = translate_segments(request.segments, request.language)
        return TranslateSegmentSrt(srt=srt_str, segments=segments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))