import os
from typing import Optional

import typer

from .db import VectorDB
from .agent import RAGAgent
from .config import (
    DB_BASE_DIR,
    MARKDOWN_ROOT
)

app = typer.Typer(help="📚 Markdown RAG CLI")


# ---------------------------
# BUILD DATABASE
# ---------------------------
@app.command()
def build(
    db_name: str = typer.Argument(..., help="Name of the database to create"),
    markdown_path: str = typer.Argument(MARKDOWN_ROOT, help="Path to markdown files")
):
    """Build a database with a user-given name"""
    db = VectorDB(db_name=db_name)
    db.build(markdown_path)
    typer.echo(f"✅ Database '{db_name}' built at {db.db_path}")

# ---------------------------
# SEARCH ONLY (DEBUG TOOL)
# ---------------------------
@app.command()
def search(
    db_name: str = typer.Argument(..., help="Database name to search within"),
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(10, help="Top K results")
):
    """
    Run retrieval without LLM (debugging).
    """
    db = VectorDB(db_name=db_name)
    db.load()

    results = db.search(query, k=k)

    typer.secho(f"\n🔎 Raw Results for '{db_name}':\n", fg=typer.colors.BLUE)

    for i, r in enumerate(results):
        typer.echo(f"\n--- Result {i+1} ---")
        typer.echo(f"Score: {r['score']:.4f}")
        typer.echo(f"Source: {r['data']['source']}")
        typer.echo(r["data"]["text"][:500])


# ---------------------------
# CHAT (MAIN ENTRYPOINT)
# ---------------------------
@app.command()
def chat(
    db_name: str = typer.Argument("default", help="Database name to use")
):
    """Start a chat using a specific database"""
    db = VectorDB(db_name=db_name)
    db.load()

    agent = RAGAgent(db=db)
    agent.chat()


# ---------------------------
# REBUILD + CHAT (CONVENIENCE)
# ---------------------------
@app.command()
def rebuild_and_chat(
    db_name: str = typer.Argument(..., help="Database name"),
    markdown_path: str = typer.Argument(..., help="Markdown folder")
):
    """
    Rebuild a named database and immediately start chat.
    """
    from .config import DB_EMBEDDINGS_CACHE
    os.remove(DB_EMBEDDINGS_CACHE)
    db = VectorDB(db_name=db_name)

    typer.echo(f"🔄 Rebuilding database '{db_name}'...")
    db.build(markdown_path)

    typer.secho("✅ Build complete. Starting chat...\n", fg=typer.colors.GREEN)

    agent = RAGAgent(db=db)
    agent.chat()


# ---------------------------
# INSPECT DB
# ---------------------------
@app.command()
def stats(
    db_name: str = typer.Argument(None, help="Database name (optional)")
):
    """
    Show stats for one or all databases.
    """
    if db_name:
        db = VectorDB(db_name)
        db.load()

        typer.echo(f"📊 Stats for '{db_name}':")
        typer.echo(f"Chunks: {len(db.metadata)}")
        typer.echo(f"Location: {db.db_path}")

    else:
        typer.echo("📊 All databases:\n")

        for db_path in DB_BASE_DIR.iterdir():
            if db_path.is_dir():
                name = db_path.name

                try:
                    db = VectorDB(name)
                    db.load()
                    typer.echo(f"• {name}: {len(db.metadata)} chunks")
                except Exception:
                    typer.echo(f"• {name}: ⚠️ corrupted or incomplete")


@app.command()
def list():
    """
    List all available databases.
    """
    # Fixed relative import for consistency
    from .config import DB_BASE_DIR

    typer.echo("📚 Available databases:\n")

    found = False
    for db_path in DB_BASE_DIR.iterdir():
        if db_path.is_dir():
            typer.echo(f"• {db_path.name}")
            found = True

    if not found:
        typer.echo("No databases found. Use `raqa build` first.")
        

if __name__ == "__main__":
    app()