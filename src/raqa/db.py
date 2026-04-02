import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .utils import split_sentences, window_chunks
from .config import (
    DB_BASE_DIR,
    EMBEDDING_MODEL,
    CHUNK_WINDOW,
    TOP_K,
    SIMILARITY_RADIUS
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class VectorDB:
    def __init__(self, db_name: str = "default"):
        self.name = db_name
        self.db_path = DB_BASE_DIR / self.name
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.metadata = []

    def build(self, markdown_root: str):
        """Ingest markdown files, chunk, encode, and store embeddings"""
        chunks = []

        for path in Path(markdown_root).rglob("*.md"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            sentences = split_sentences(text)
            for c in window_chunks(sentences, CHUNK_WINDOW):
                chunks.append({
                    "text": c["text"],
                    "source": str(path),
                    "position": c["index"]
                })

        # Encode
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        dim = embeddings.shape[1]

        # Build FAISS index
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        self.metadata = chunks
        self.save()

    def save(self):
        faiss.write_index(self.index, str(self.db_path / "index.faiss"))
        with open(self.db_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

    def load(self):
        if not (self.db_path / "index.faiss").exists():
            raise FileNotFoundError(f"No database found at {self.db_path}. Please build it first.")
        self.index = faiss.read_index(str(self.db_path / "index.faiss"))
        with open(self.db_path / "meta.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(self, query: str, k=TOP_K):
        q_emb = self.model.encode([query])
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, k)
        return [{"score": float(s), "data": self.metadata[i]} for s, i in zip(scores[0], indices[0])]

    def nucleus_filter(self, results):
        if not results:
            return []
        best = results[0]["score"]
        filtered = [r for r in results if (best - r["score"]) <= SIMILARITY_RADIUS]
        scores = np.array([r["score"] for r in filtered])
        probs = np.exp(scores) / np.sum(np.exp(scores))
        sampled_indices = np.random.choice(len(filtered), size=min(len(filtered), 10), replace=False, p=probs)
        return [filtered[i] for i in sampled_indices]