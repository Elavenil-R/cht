import requests
from app.core.config import settings

_session = requests.Session()

def chat_complete(
    system_text: str,
    user_text: str,
    history_messages: list[dict] | None = None,
    max_tokens: int = 200,
    temperature: float = 0.2,
) -> str:
    url = f"{settings.LM_URL}/chat/completions"

    messages: list[dict] = [{"role": "system", "content": system_text}]

    # Add history (sliding window)
    if history_messages:
        for m in history_messages:
            role = (m or {}).get("role")
            content = (m or {}).get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content.strip()})

    # Current user question always last
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": settings.MODEL_NAME,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    r = _session.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return (data["choices"][0]["message"]["content"] or "").strip()