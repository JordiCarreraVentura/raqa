import os
import re
from pathlib import Path
from typing import List

from config import ENV_FILE



def split_sentences(text: str) -> List[str]:
    # simple but effective
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def window_chunks(sentences: List[str], window: int = 3):
    chunks = []

    for i in range(len(sentences)):
        left = sentences[max(0, i - window): i]
        center = [sentences[i]]
        right = sentences[i + 1: i + 1 + window]

        chunk_text = " ".join(left + center + right)

        chunks.append({
            "text": chunk_text,
            "center": sentences[i],
            "index": i
        })

    return chunks


def get_openai_key() -> str:
    """
    Load the OpenAI API key from ENV_FILE, prompt user if missing.
    """
    if ENV_FILE.exists():
        key = ENV_FILE.read_text().strip()
        if key:
            return key

    # Prompt user
    print(f"🔑 OpenAI API key not found. Enter your key (it will be saved at {ENV_FILE}):")
    key = input("API Key: ").strip()

    # Save to file
    ENV_FILE.write_text(key)
    print(f"✅ Key saved at {ENV_FILE}")
    return key