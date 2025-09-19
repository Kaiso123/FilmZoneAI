from pydantic import BaseModel, Field
from typing import List

class TranslateRequest(BaseModel):
    texts: List[str] = Field(..., example=["vi: Xin chào", "en: Hello world"])

class TranslateResponse(BaseModel):
    translations: List[str]

class TranslateSegmentsRequest(BaseModel):
    segments: List[dict] = Field(..., example=[{"start": 0.0, "end": 2.5, "text": "Hello world"}])
    language: str = Field(None, example="vi")
    
class TranslateSegmentSrt(BaseModel):
    srt: str
    segments: List[dict]