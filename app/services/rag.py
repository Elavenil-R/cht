from __future__ import annotations
from typing import List

from app.core.config import settings
from app.services.knowledge import load_knowledge_files, load_bot_rules
from app.services.chunking import chunk_text, Chunk
from app.services.embeddings import embed_texts
from app.services.lm_client import chat_complete
from app.services.vectorstore import ChromaVectorDB


class RAGEngine:
    def __init__(self):
        self.vdb: ChromaVectorDB | None = None
        self.bot_rules: str = ""
        self._db_ready = False

    # -----------------------------
    # INITIALIZE VECTOR DATABASE
    # -----------------------------
    def initialize_db(self) -> None:
        if self._db_ready:
            return

        settings.storage_dir_path.mkdir(parents=True, exist_ok=True)

        self.bot_rules = load_bot_rules(
            settings.knowledge_dir_path,
            settings.BOT_RULES_FILE
        ).strip()

        self.vdb = ChromaVectorDB(
            persist_dir=str(settings.storage_dir_path),
            collection_name="uniq_kb",
        )

        chunks = self._load_and_chunk_all()
        print("TOTAL CHUNKS =", len(chunks))

        self.vdb.reset()

        if chunks:
            texts = [c.text for c in chunks]

            embeddings: list[list[float]] = []
            B = 64
            for i in range(0, len(texts), B):
                embeddings.extend(embed_texts(texts[i:i + B]))

            self.vdb.upsert(chunks, embeddings)

        self._db_ready = True

    # -----------------------------
    # LOAD KNOWLEDGE
    # -----------------------------
    def _load_and_chunk_all(self) -> List[Chunk]:
        files = load_knowledge_files(
            settings.knowledge_dir_path,
            settings.knowledge_files_list
        )

        all_chunks: List[Chunk] = []

        for source, text in files:
            all_chunks.extend(
                chunk_text(
                    source=source,
                    text=text,
                    chunk_size=settings.CHUNK_SIZE,
                    overlap=settings.CHUNK_OVERLAP,
                )
            )

        return all_chunks

    # =====================================================
    # ✅ MAIN ANSWER FUNCTION (NOW WITH SLIDING WINDOW)
    # =====================================================
    def answer(self, question: str, history_messages=None) -> str:

        q = (question or "").strip()
        if not q:
            return settings.FALLBACK_TEXT

        self.initialize_db()

        if self.vdb is None:
            return settings.FALLBACK_TEXT

        # ---------------------------------
        # EMBEDDING SEARCH
        # ---------------------------------
        q_emb = embed_texts([q])[0]
        hits = self.vdb.query(q_emb, top_k=settings.TOP_K) or []

        best = hits[0].similarity if hits else 0.0

        # =====================================================
        # STRICT MODE (NO CONTEXT FOUND)
        # =====================================================
        if best < settings.MIN_SIMILARITY:

            user_text = f"""
USER_MESSAGE:
{q}

RULE:
- If USER_MESSAGE is greeting / thanks / acknowledgement
  OR asks your identity/name,
  reply naturally using SYSTEM PROMPT.
- Otherwise reply exactly:
{settings.FALLBACK_TEXT}
"""

            out = chat_complete(
                system_text=self.bot_rules,
                user_text=user_text,
                history_messages=history_messages,
                max_tokens=80,
            ).strip()

            return out if out else settings.FALLBACK_TEXT

        # =====================================================
        # BUILD CONTEXT FROM VECTOR DB
        # =====================================================
        ctx_lines = []

        for h in hits:
            ctx_lines.append(h.chunk.text)
            ctx_lines.append("")

        context = "\n".join(ctx_lines).strip()

        user_text = f"""
CONTEXT:
{context}

QUESTION:
{q}

RULES:
- Answer ONLY using CONTEXT.
- If answer not present, reply exactly:
{settings.FALLBACK_TEXT}
"""

        out = chat_complete(
            system_text=self.bot_rules,
            user_text=user_text,
            history_messages=history_messages,
            max_tokens=220,
        ).strip()

        if not out:
            return settings.FALLBACK_TEXT

        if settings.FALLBACK_TEXT.lower() in out.lower():
            return settings.FALLBACK_TEXT

        return out


# GLOBAL ENGINE
rag_engine = RAGEngine()