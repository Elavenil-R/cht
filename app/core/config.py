from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "UNIQ RAG Chatbot"
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    LM_URL: str = "http://127.0.0.1:1234/v1"
    MODEL_NAME: str = "qwen2.5-1.5b-instruct"
    EMBED_MODEL: str = "text-embedding-nomic-embed-text-v1.5"

    KNOWLEDGE_DIR: str = "knowledge"
    KNOWLEDGE_FILES: str = "uniq1.txt,uniq2.txt,uniq3.txt"
    BOT_RULES_FILE: str = "bot_rules.txt"

    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 140
    TOP_K: int = 8
    MIN_SIMILARITY: float = 0.35

    FALLBACK_TEXT: str = "Information not available."

    STORAGE_DIR: str = "storage"

    @property
    def knowledge_dir_path(self) -> Path:
        return Path(self.KNOWLEDGE_DIR)

    @property
    def storage_dir_path(self) -> Path:
        return Path(self.STORAGE_DIR)

    @property
    def knowledge_files_list(self) -> list[str]:
        return [x.strip() for x in self.KNOWLEDGE_FILES.split(",") if x.strip()]

settings = Settings()