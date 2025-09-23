from pydantic import BaseModel, Field
from typing import List

class TranslateRequest(BaseModel):
    texts: List[str] = Field(..., example=["vi: Xin chào", "en: Hello world"],
                             description="List of texts to translate, prefixed with target language code (e.g., 'en:', 'vi:')")

class TranslateResponse(BaseModel):
    translations: List[str] = Field(..., example=["en: Hello world", "vi: Xin chào"])

class TranslateSegmentsRequest(BaseModel):
    segments: List[dict] = Field(..., example=[{"start": 0.0, "end": 2.5, "text": "Hello world"}], 
                                 description="List of segments with start, end, and text")
    language: str = Field(None, example="vi")
    
class TranslateSegmentSrt(BaseModel):
    srt: str
    segments: List[dict] = Field(...,description="List of segments with start, end, and text")