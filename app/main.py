from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.routes.chat import router as chat_router
from app.services.rag import rag_engine

BASE_DIR = Path(__file__).resolve().parent.parent  # project root (uniq_rag_bot)
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title=settings.APP_NAME)

# API
app.include_router(chat_router, prefix="/api")

# Serve frontend (only mount if folder exists)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def home():
        return FileResponse(str(STATIC_DIR / "index.html"))
else:
    @app.get("/")
    def home():
        return {"message": "static/ folder not found. Create static/index.html and static/app.js"}

@app.on_event("startup")
def _startup():
    # Do not call LM Studio here (prevents startup cancel errors on Windows)
    pass

@app.get("/health")
def health():
    return {"status": "ok"}