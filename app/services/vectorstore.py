from dataclasses import dataclass
from typing import List, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.services.chunking import Chunk

@dataclass
class SearchHit:
    chunk: Chunk
    similarity: float  # cosine similarity in [0..1] approx

class ChromaVectorDB:
    """
    Persistent ChromaDB (local disk) using precomputed embeddings.
    We store:
      - documents: chunk text
      - embeddings: vectors
      - metadatas: source, chunk_id
      - ids: unique per chunk
    """
    def __init__(self, persist_dir: str, collection_name: str = "uniq_kb"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def reset(self) -> None:
        # delete + recreate for clean rebuild
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def upsert(self, chunks: List[Chunk], embeddings: List[list[float]]) -> None:
        ids = [f"{c.source}::chunk::{c.chunk_id}" for c in chunks]
        docs = [c.text for c in chunks]
        metas = [{"source": c.source, "chunk_id": c.chunk_id} for c in chunks]

        # Chroma needs lists
        self.collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeddings,
        )

    def query(self, query_embedding: list[float], top_k: int) -> List[SearchHit]:
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        hits: List[SearchHit] = []
        for doc, meta, dist in zip(docs, metas, dists):
            # Chroma returns distance depending on backend; commonly cosine distance (0=best)
            # Convert to similarity:
            try:
                similarity = 1.0 - float(dist)
            except Exception:
                similarity = 0.0

            chunk = Chunk(
                source=str(meta.get("source", "")),
                chunk_id=int(meta.get("chunk_id", 0)),
                text=str(doc or ""),
            )
            hits.append(SearchHit(chunk=chunk, similarity=similarity))

        # Ensure sorted by best similarity
        hits.sort(key=lambda x: x.similarity, reverse=True)
        return hits