from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"  # important for per-user memory

class ChatResponse(BaseModel):
    answer: str