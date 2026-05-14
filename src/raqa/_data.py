import csv
import re
from pathlib import Path
from typing import List, Dict


def load_documents(data_dir: str = "data") -> List[Dict]:
    chunks = []
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"⚠️  Data directory '{data_dir}' not found.")
        return chunks

    for path in sorted(data_path.rglob("*")):
        if path.suffix == ".md":
            text = path.read_text(encoding="utf-8")
            clean = _remove_markdown(text)
            paras = [p.strip() for p in re.split(r'\n\s*\n', clean) if p.strip()]
            for i, para in enumerate(paras):
                chunks.append({"text": para, "source": str(path), "index": i})
        elif path.suffix == ".txt":
            text = path.read_text(encoding="utf-8")
            paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
            for i, para in enumerate(paras):
                chunks.append({"text": para, "source": str(path), "index": i})
        elif path.suffix == ".csv":
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            if rows:
                lines = [" | ".join(r) for r in rows]
                chunks.append({"text": "\n".join(lines), "source": str(path), "index": 0})

    return chunks


def _remove_markdown(text: str) -> str:
    text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'^\s*#+\s+(.*)$', r'\1', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[\-\*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
