import json
import os
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import snippyts
from joblib import Memory
from dotenv import load_dotenv
from openai import OpenAI  # Added OpenAI import
from tqdm import tqdm

from .utils import split_sentences, window_chunks
from .config import (
    DB_BASE_DIR,
    EMBEDDING_MODEL,
    CHUNK_WINDOW,
    DB_EMBEDDINGS_CACHE,
    TOP_K,
    SIMILARITY_RADIUS
)

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "true"

if os.path.exists(DB_EMBEDDINGS_CACHE):
    EMBEDDINGS_CACHE = snippyts.from_json(DB_EMBEDDINGS_CACHE)
else:
    EMBEDDINGS_CACHE = dict([])


def remove_markdown(text: str) -> str:
    """
    Removes basic Markdown formatting from a string.
    """
    # 1. Remove images: ![alt](url) -> ""
    text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '', text)
    
    # 2. Remove links but keep the text: [link text](url) -> "link text"
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # 3. Remove bold and italics: **text**, __text__, *text*, _text_
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    
    # 4. Remove inline code: `code` -> code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # 5. Remove headers: # Header -> Header
    text = re.sub(r'^\s*#+\s+(.*)$', r'\1', text, flags=re.MULTILINE)
    
    # 6. Remove blockquotes: > quote -> quote
    text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)
    
    # 7. Remove horizontal rules: ---, ***, ___
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    
    # 8. Remove list markers: - item, * item, 1. item
    text = re.sub(r'^\s*[\-\*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Clean up excess whitespace/newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


class VectorDB:
    def __init__(self, db_name: str = "default"):
        self.name = db_name
        self.db_path = DB_BASE_DIR / self.name
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize OpenAI client instead of SentenceTransformer
        self.client = OpenAI()
        self.index = None
        self.metadata = []


    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """Fetch embeddings from OpenAI with batching, caching, and robust parsing"""
        indexed_texts = list(enumerate(texts))
        
        # FIX 1: Store the actual vector from cache, not the text string
        old_texts = [(idx, EMBEDDINGS_CACHE[text]) for idx, text in indexed_texts if text in EMBEDDINGS_CACHE]
        new_texts = [(idx, text) for idx, text in indexed_texts if text not in EMBEDDINGS_CACHE]

        all_vectors = old_texts
        batch_size = 100 
        batches = snippyts.batched(new_texts, batch_size)

        for batch in tqdm(batches, colour="green", desc="Fetching Embeddings"):
            # Ensure we don't send empty strings to the API
            clean_batch = [t if t.strip() else " " for _, t in batch]
            
            response = self.client.embeddings.create(
                input=clean_batch,
                model=EMBEDDING_MODEL
            )

            res_data = response.data if hasattr(response, 'data') else response
            
            for (idx, text), item in zip(batch, res_data):
                vector = item.embedding if hasattr(item, 'embedding') else item
                
                # FIX 2: Handle nested lists returned by some API proxies (e.g. [[val, val...]])
                if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
                    vector = vector[0]
                
                EMBEDDINGS_CACHE[text] = vector
                all_vectors.append((idx, vector))
        
        # Sort back to original order
        all_vectors.sort(key=lambda x: x[0])
        final_vectors = [vec for _, vec in all_vectors]

        # FIX 3: Validate dimensions before converting to NumPy
        if final_vectors:
            dims = [len(v) for v in final_vectors]
            if len(set(dims)) > 1:
                # If this happens, your cache likely has vectors from a different model.
                # You may need to delete your cache file.
                raise ValueError(f"Inconsistent embedding dimensions: {set(dims)}. Clear your cache.")

        # Save cache and return matrix
        snippyts.to_json(EMBEDDINGS_CACHE, DB_EMBEDDINGS_CACHE)
        return np.array(final_vectors).astype('float32')


    def build(self, markdown_root: str):
        """Ingest markdown files, chunk, encode via OpenAI, and store"""
        chunks = []

        for path in Path(markdown_root).rglob("*.md"):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                # Use the existing remove_markdown logic
                clean_text = remove_markdown(content)

            sentences = split_sentences(clean_text)
            for c in window_chunks(sentences, CHUNK_WINDOW):
                # Skip chunks that became empty after markdown removal
                if not c["text"].strip():
                    continue
                    
                chunks.append({
                    "text": c["text"],
                    "source": str(path),
                    "position": c["index"]
                })

        if not chunks:
            print("⚠️ No valid text found in markdown files.")
            return

        texts = [c["text"] for c in chunks]
        embeddings = self._get_embeddings(texts)
        
        # Build FAISS index
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
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
        # Encode query using OpenAI
        q_emb = self._get_embeddings([query])
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