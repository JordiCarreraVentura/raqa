from pathlib import Path

# User folder for raqa configs
HOME = Path.home()
RAQA_HOME = HOME / ".raqa"

# OpenAI credential file
ENV_FILE = RAQA_HOME / "env"

# Database folder (moved from project data)
DATA_DIR = RAQA_HOME / "data"

# Embeddings & chunk config
EMBEDDING_MODEL = "sentence-transformers/xlm-roberta-large-xnli"
CHUNK_WINDOW = 3
TOP_K = 50
SIMILARITY_RADIUS = 0.4

# Default markdown folder (can override via CLI)
MARKDOWN_ROOT = "./markdown_files"

# Ensure directories exist
RAQA_HOME.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)