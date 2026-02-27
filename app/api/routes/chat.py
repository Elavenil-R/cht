from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag import rag_engine
from app.services.memory import SlidingWindowMemory

router = APIRouter()

# Sliding window memory: last 10 turns (user+assistant pairs)
memory = SlidingWindowMemory(max_turns=10)

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = (getattr(req, "session_id", None) or "default").strip()

    question = (req.question or "").strip()
    if not question:
        return ChatResponse(answer="Information not available.")

    # 1) Save user message into memory
    memory.add_user(session_id, question)

    # 2) Ask RAG with history (we send memory into RAG -> LLM)
    answer = rag_engine.answer(question, history_messages=memory.get(session_id))

    # 3) Save assistant answer into memory
    memory.add_assistant(session_id, answer)

    return ChatResponse(answer=answer)