# RAQA

**R**etrieval-**A**ugmented **Q**uestion-**A**nswering

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

    `python cli.py build DATABASE_NAME PATH/TO/FOLDER/WITH/MARKDOWNS`

2. Chat

    `python cli.py chat DATABASE_NAME`

3. One-shot retrieval

    `python cli.py search DATABASE_NAME "what is retrieval augmented generation?"`

4. Rebuild and chat

    `python cli.py rebuild-and-chat DATABASE_NAME PATH/TO/FOLDER/WITH/MARKDOWNS`

5. Get stats

    `python cli.py stats`

6. List databases

    `python cli.py list`


### BASH natively

```
raqa build DATABASE_NAME PATH/TO/FOLDER/WITH/MARKDOWNS
raqa chat DATABASE_NAME
raqa search DATABASE_NAME "what is RAG?"
raqa list (DATABASE_NAME)
raqa stats (DATABASE_NAME)
raqa rebuild-and-chat DATABASE_NAME PATH/TO/FOLDER/WITH/MARKDOWNS
```


## Python

### Build database

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

1. If any changes are made, update `pyproject.toml`.
2. Building the package before uploading:
    `cd raqa; python -m build`.
3. Upload the package to pypi:
    `python -m twine upload --repository {pypi|testpypi} dist/*`

Steps 2 and 3 can be done automatically by running `make publish`.

## Related or comparable projects

1. [PyRAG](https://pypi.org/project/PyRAG) difference: focuses on SingleStore.
2. [ragger-simple](https://pypi.org/project/ragger-simple) difference: uses Qdrant and requires a Qdrant API key, in addition to an LLM API key. RAQA only requires the latter and uses an open-source tech stack for the rest.

## Next steps

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