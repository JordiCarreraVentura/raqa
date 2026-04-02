import typer
from typing import Optional

from db import VectorDB
from agent import RAGAgent
from config import MARKDOWN_ROOT

app = typer.Typer(help="📚 Markdown RAG CLI")


# ---------------------------
# BUILD DATABASE
# ---------------------------
@app.command()
def build(
    path: str = typer.Option(
        MARKDOWN_ROOT,
        help="Path to markdown folder"
    )
):
    """
    Build vector database from markdown files.
    """
    db = VectorDB()
    db.build(path)

    typer.secho("✅ Database built successfully.", fg=typer.colors.GREEN)


# ---------------------------
# SEARCH ONLY (DEBUG TOOL)
# ---------------------------
@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(10, help="Top K results")
):
    """
    Run retrieval without LLM (debugging).
    """
    db = VectorDB()
    db.load()

    results = db.search(query, k=k)

    typer.secho("\n🔎 Raw Results:\n", fg=typer.colors.BLUE)

    for i, r in enumerate(results):
        typer.echo(f"\n--- Result {i+1} ---")
        typer.echo(f"Score: {r['score']:.4f}")
        typer.echo(f"Source: {r['data']['source']}")
        typer.echo(r["data"]["text"][:500])


# ---------------------------
# CHAT (MAIN ENTRYPOINT)
# ---------------------------
@app.command()
def chat():
    """
    Start conversational RAG agent.
    """
    agent = RAGAgent()
    agent.chat()


# ---------------------------
# REBUILD + CHAT (CONVENIENCE)
# ---------------------------
@app.command()
def rebuild_and_chat(
    path: str = typer.Option(
        MARKDOWN_ROOT,
        help="Markdown folder"
    )
):
    """
    Rebuild database and immediately start chat.
    """
    db = VectorDB()
    db.build(path)

    typer.secho("\n🚀 Starting chat...\n", fg=typer.colors.GREEN)

    agent = RAGAgent()
    agent.chat()


# ---------------------------
# INSPECT DB
# ---------------------------
@app.command()
def stats():
    """
    Show database stats.
    """
    db = VectorDB()
    db.load()

    typer.echo("📊 Database Stats:")
    typer.echo(f"Total chunks: {len(db.metadata)}")


if __name__ == "__main__":
    app()