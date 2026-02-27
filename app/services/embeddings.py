import requests
from app.core.config import settings

def embed_texts(texts: list[str]) -> list[list[float]]:
    url = f"{settings.LM_URL}/embeddings"

    payload = {
        "model": settings.EMBED_MODEL,
        "input": texts
    }

    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()

    data = r.json()["data"]

    return [d["embedding"] for d in data]   