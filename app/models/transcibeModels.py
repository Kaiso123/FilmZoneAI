from pydantic import BaseModel

class TranscribeResponse(BaseModel):
    text: str
    language: str
    segments: list[dict]  

class TranscribeToSRTResponse(BaseModel):
    srt: str
    segments: list[dict]
    language: str
    raw_segments: list[dict] = None 

