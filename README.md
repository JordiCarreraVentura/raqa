# RAQA

**[R]**etrieval-**[A]**ugmented **[Q]**uestion-**[A]**nswering

# Usage

## Installation

```
pip install raqa
```

## Build database


```
from db import VectorDB
from config import MARKDOWN_ROOT

db = VectorDB()
db.build(MARKDOWN_ROOT)
```


## Run

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