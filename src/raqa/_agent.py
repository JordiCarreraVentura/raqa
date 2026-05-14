import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, function_tool

from ._data import load_documents


load_dotenv()

_client = None
chunks = []
embeddings = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def initialize(data_dir: str = "data"):
    global chunks, embeddings
    chunks = load_documents(data_dir)
    if not chunks:
        embeddings = np.array([], dtype="float32").reshape(0, 0)
        return

    texts = [c["text"] for c in chunks]
    response = _get_client().embeddings.create(input=texts, model="text-embedding-3-small")
    embeddings = np.array([d.embedding for d in response.data], dtype="float32")


@function_tool
def search_docs(query: str) -> str:
    """Search the loaded documents for information relevant to the query."""
    if not chunks or embeddings.size == 0:
        return "No documents loaded."

    q_emb = _get_client().embeddings.create(
        input=[query], model="text-embedding-3-small"
    ).data[0].embedding
    q_vec = np.array(q_emb, dtype="float32")

    scores = embeddings @ q_vec
    top_k = min(5, len(chunks))
    indices = np.argsort(scores)[-top_k:][::-1]

    results = [f"[{chunks[i]['source']}]\n{chunks[i]['text']}" for i in indices]
    return "\n\n---\n\n".join(results)


def create_agent() -> Agent:
    return Agent(
        name="RAQA",
        instructions=(
            "You are a helpful assistant. "
            "Answer questions based on the provided documents. "
            "If the search results are not relevant, say so."
        ),
        tools=[search_docs],
    )
