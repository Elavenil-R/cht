import re
from dataclasses import dataclass

@dataclass
class Chunk:
    source: str
    chunk_id: int
    text: str

def _clean_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # collapse excessive blank lines
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t

def chunk_text(source: str, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    text = _clean_text(text)
    if not text:
        return []

    # Split by paragraphs first, then pack into chunks
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    buf = ""
    cid = 0

    def flush(b: str):
        nonlocal cid
        b = b.strip()
        if b:
            chunks.append(Chunk(source=source, chunk_id=cid, text=b))
            cid += 1

    for p in paras:
        if not buf:
            buf = p
            continue

        if len(buf) + 2 + len(p) <= chunk_size:
            buf = buf + "\n\n" + p
        else:
            flush(buf)

            # overlap: keep last N chars from previous chunk
            tail = buf[-overlap:] if overlap > 0 else ""
            buf = (tail + "\n\n" + p).strip()

            # If still too large, hard-split
            while len(buf) > chunk_size:
                flush(buf[:chunk_size])
                tail2 = buf[chunk_size - overlap:chunk_size] if overlap > 0 else ""
                buf = (tail2 + buf[chunk_size:]).strip()

    flush(buf)
    return chunks   