import re
from typing import List


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