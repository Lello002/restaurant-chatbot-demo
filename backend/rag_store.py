import json
import math
import os
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"

CHUNKS_PATH = STORAGE_DIR / "chunks.json"
EMBEDDINGS_PATH = STORAGE_DIR / "embeddings.json"

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


class RagStore:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []

    def load(self):
        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"File non trovato: {CHUNKS_PATH}. Esegui prima: python ingest_catalog.py"
            )

        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                f"File non trovato: {EMBEDDINGS_PATH}. Esegui prima: python ingest_catalog.py"
            )

        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
            self.embeddings = json.load(f)

        if len(self.chunks) != len(self.embeddings):
            raise RuntimeError(
                f"Numero chunks ({len(self.chunks)}) diverso dal numero embeddings ({len(self.embeddings)})."
            )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        cleaned_texts = [text.strip() for text in texts if text and text.strip()]

        if not cleaned_texts:
            return []

        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=cleaned_texts
        )

        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> List[float]:
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        )

        return response.data[0].embedding

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0

        for x, y in zip(a, b):
            dot += x * y
            norm_a += x * x
            norm_b += y * y

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.chunks or not self.embeddings:
            raise RuntimeError("RagStore non caricato.")

        query_embedding = self.embed_query(query)

        scored = []

        for chunk, embedding in zip(self.chunks, self.embeddings):
            score = self.cosine_similarity(query_embedding, embedding)

            result = dict(chunk)
            result["score"] = float(score)

            scored.append(result)

        scored.sort(key=lambda item: item["score"], reverse=True)

        return scored[:top_k]