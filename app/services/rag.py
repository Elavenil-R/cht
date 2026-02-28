from __future__ import annotations
from typing import List, Optional

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

    def initialize_db(self) -> None:
        if self._db_ready:
            return

        settings.storage_dir_path.mkdir(parents=True, exist_ok=True)

        self.bot_rules = load_bot_rules(
            settings.knowledge_dir_path, settings.BOT_RULES_FILE
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

    def _load_and_chunk_all(self) -> List[Chunk]:
        files = load_knowledge_files(settings.knowledge_dir_path, settings.knowledge_files_list)
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

    # -----------------------------
    # Helpers
    # -----------------------------
    def _last_assistant_text(self, history_messages) -> str:
        if not history_messages:
            return ""
        for m in reversed(history_messages):
            if (m or {}).get("role") == "assistant":
                t = (m or {}).get("content") or ""
                return t.strip()
        return ""

    def _classify_intent(self, user_msg: str, history_messages=None) -> str:
        """
        Returns one of:
        - "uniq_question"
        - "verification_or_feedback"
        - "casual"
        """
        sys = (
            "You are an intent classifier for a UNIQ Technologies chatbot.\n"
            "Return ONLY one label:\n"
            "uniq_question = user is asking for UNIQ Technologies info.\n"
            "verification_or_feedback = user is evaluating/correcting previous assistant answer or asking if it is correct.\n"
            "casual = greeting/thanks/acknowledgement/small talk.\n"
            "Do not output anything else."
        )

        # Keep it short. Use history if available.
        hist_preview = ""
        if history_messages:
            # take last 6 messages max for intent
            tail = history_messages[-6:]
            lines = []
            for x in tail:
                r = (x or {}).get("role")
                c = ((x or {}).get("content") or "").strip()
                if r in ("user", "assistant") and c:
                    lines.append(f"{r.upper()}: {c}")
            hist_preview = "\n".join(lines).strip()

        user = f"CHAT_TAIL:\n{hist_preview}\n\nUSER_MESSAGE:\n{user_msg}".strip()

        label = chat_complete(
            system_text=sys,
            user_text=user,
            history_messages=None,
            max_tokens=20,
            temperature=0.0,
        ).strip().lower()

        if "verification" in label:
            return "verification_or_feedback"
        if "casual" in label:
            return "casual"
        return "uniq_question"

    def _verify_last_answer(self, last_answer: str, history_messages=None) -> str:
        """
        Try to verify the last assistant answer by retrieving context for it.
        If verification is possible, confirm or correct.
        If not possible, ask user what to verify or fallback.
        """
        if not last_answer:
            return "Could you tell me which answer you want to verify?"

        if self.vdb is None:
            return settings.FALLBACK_TEXT

        # Retrieve context for the last answer itself
        ans_emb = embed_texts([last_answer])[0]
        hits = self.vdb.query(ans_emb, top_k=settings.TOP_K) or []
        best = hits[0].similarity if hits else 0.0

        # If we cannot retrieve any supporting context, we can't verify
        if best < settings.MIN_SIMILARITY:
            return "I can verify it if you share which exact statement is wrong, or ask a specific UNIQ-related question."

        ctx_lines = []
        for h in hits:
            ctx_lines.append(h.chunk.text)
            ctx_lines.append("")
        context = "\n".join(ctx_lines).strip()

        user_text = f"""
CONTEXT:
{context}

PREVIOUS_ASSISTANT_ANSWER:
{last_answer}

TASK:
- Check whether PREVIOUS_ASSISTANT_ANSWER is supported by CONTEXT.
- If supported, reply briefly: "Yes, that information is correct." (you may add 1 supporting line from CONTEXT).
- If not supported or partially wrong, reply briefly with the correct info from CONTEXT.
- If CONTEXT is insufficient, reply exactly:
{settings.FALLBACK_TEXT}
""".strip()

        out = chat_complete(
            system_text=self.bot_rules,
            user_text=user_text,
            history_messages=history_messages,
            max_tokens=160,
            temperature=0.0,
        ).strip()

        if not out:
            return settings.FALLBACK_TEXT
        if settings.FALLBACK_TEXT.lower() in out.lower():
            return settings.FALLBACK_TEXT
        return out

    # -----------------------------
    # Main
    # -----------------------------
    def answer(self, question: str, history_messages=None) -> str:
        q = (question or "").strip()
        if not q:
            return settings.FALLBACK_TEXT

        self.initialize_db()
        if self.vdb is None:
            return settings.FALLBACK_TEXT

        # 1) Intent routing (so feedback like "wrong" is handled)
        intent = self._classify_intent(q, history_messages=history_messages)

        # If user is verifying/correcting, try verifying the last assistant answer
        if intent == "verification_or_feedback":
            last_ans = self._last_assistant_text(history_messages)
            return self._verify_last_answer(last_ans, history_messages=history_messages)

        # 2) Normal strict UNIQ RAG (your original flow)
        q_emb = embed_texts([q])[0]
        hits = self.vdb.query(q_emb, top_k=settings.TOP_K) or []
        best = hits[0].similarity if hits else 0.0

        # STRICT: if not enough similarity, do NOT answer from world knowledge.
        # Allow ONLY casual / identity via system prompt.
        if best < settings.MIN_SIMILARITY:
            user_text = f"""
USER_MESSAGE:
{q}

RULE:
- If USER_MESSAGE is a greeting/thanks/acknowledgement OR asks your identity/name,
  reply naturally according to the SYSTEM PROMPT.
- Otherwise reply exactly:
{settings.FALLBACK_TEXT}
""".strip()

            out = chat_complete(
                system_text=self.bot_rules,
                user_text=user_text,
                history_messages=history_messages,
                max_tokens=80,
                temperature=0.0,
            ).strip()

            return out if out else settings.FALLBACK_TEXT

        # Build context from hits
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
- If answer is not in CONTEXT, reply exactly:
{settings.FALLBACK_TEXT}
""".strip()

        out = chat_complete(
            system_text=self.bot_rules,
            user_text=user_text,
            history_messages=history_messages,
            max_tokens=220,
            temperature=0.0,
        ).strip()

        if not out:
            return settings.FALLBACK_TEXT
        if settings.FALLBACK_TEXT.lower() in out.lower():
            return settings.FALLBACK_TEXT
        return out


rag_engine = RAGEngine()