import os
import json
from pathlib import Path

import faiss
import numpy as np
import frontmatter
from sentence_transformers import SentenceTransformer

from utils import split_sentences, window_chunks
from config import *

class VectorDB:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.metadata = []

    def ingest_markdown(self, root_dir: str):
        all_chunks = []

        for path in Path(root_dir).rglob("*.md"):
            with open(path, "r", encoding="utf-8") as f:
                post = frontmatter.load(f)

            text = post.content
            meta = post.metadata

            sentences = split_sentences(text)
            chunks = window_chunks(sentences, CHUNK_WINDOW)

            for c in chunks:
                all_chunks.append({
                    "text": c["text"],
                    "source": str(path),
                    "meta": meta,
                    "position": c["index"]
                })

        return all_chunks

    def build(self, root_dir: str):
        print("📥 Ingesting markdown...")
        chunks = self.ingest_markdown(root_dir)

        print("🧠 Encoding...")
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        dim = embeddings.shape[1]
        # self.index = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexHNSWFlat(dim, 32)

        # normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        self.metadata = chunks

        self.save()

    def save(self):
        DATA_DIR.mkdir(exist_ok=True)

        faiss.write_index(self.index, str(DATA_DIR / "index.faiss"))

        with open(DATA_DIR / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index(str(DATA_DIR / "index.faiss"))

        with open(DATA_DIR / "meta.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(self, query: str, k=TOP_K):
        q_emb = self.model.encode([query])
        faiss.normalize_L2(q_emb)

        scores, indices = self.index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "score": float(score),
                "data": self.metadata[idx]
            })

        return results

    def nucleus_filter(self, results):
        if not results:
            return []

        best = results[0]["score"]

        filtered = [
            r for r in results
            if (best - r["score"]) <= SIMILARITY_RADIUS
        ]

        # softmax sampling
        scores = np.array([r["score"] for r in filtered])
        probs = np.exp(scores) / np.sum(np.exp(scores))

        sampled_indices = np.random.choice(
            len(filtered),
            size=min(len(filtered), 10),
            replace=False,
            p=probs
        )

        return [filtered[i] for i in sampled_indices]