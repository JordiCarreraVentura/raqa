# RAQA

**[R]**etrieval-**[A]**ugmented **[Q]**uestion-**[A]**nswering

Retrieval-augmented, pip-installable, CLI-based question answering over arbitrary document collections.

# Usage

## Installation

```
pip install raqa
```

**Locally**

`pip install -e .`

## Run

### BASH via Python interpreter

1. Build DB

    `python cli.py build --path ./docs`

2. Chat

    `python cli.py chat`

3. One-shot retrieval

    `python cli.py search "what is retrieval augmented generation?"`

4. Rebuild and chat

    `python cli.py rebuild-and-chat`

5. Get stats

    `python cli.py stats`

### BASH natively

```
raqa build --path ./markdown_files
raqa chat
raqa search "what is RAG?"
raqa stats
raqa rebuild-and-chat
```


## Python

### Build database

```
from db import VectorDB
from config import MARKDOWN_ROOT

db = VectorDB()
db.build(MARKDOWN_ROOT)
```

### Run

```
from agent import RAGAgent

agent = RAGAgent()
agent.chat()
```


## Build instructions

Next steps:

1. If any changes are made, update `pyproject.toml`.
2. Building the package before uploading:
    `cd raqa; python -m build`.
3. Upload the package to pypi:
    `python -m twine upload --repository {pypi|testpypi} dist/*`

## Next steps

### Real tool-calling (instead of implicit RAG)

Define OpenAI tool:

```
{
  "name": "search_docs",
  "description": "...",
  "parameters": { "query": "string" }
}
```

### Hybrid search

Combine BM25 (rank-bm25) + embeddings