import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"

CHUNKS_PATH = STORAGE_DIR / "chunks.json"
EMBEDDINGS_PATH = STORAGE_DIR / "embeddings.npy"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class RagStore:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None

    def load(self):
        if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                "Indice RAG non trovato. Esegui prima: python ingest_menu.py"
            )

        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.embeddings = np.load(EMBEDDINGS_PATH)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return np.array(embeddings, dtype=np.float32)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.embeddings is None:
            raise RuntimeError("RagStore non caricato.")

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        )

        query_embedding = np.array(query_embedding, dtype=np.float32)[0]

        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = dict(self.chunks[int(idx)])
            chunk["score"] = float(scores[int(idx)])
            results.append(chunk)

        return results