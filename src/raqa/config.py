from pathlib import Path

# User folder for raqa configs
HOME = Path.home()
RAQA_HOME = HOME / ".raqa"
RAQA_HOME.mkdir(parents=True, exist_ok=True)

# OpenAI credential file
ENV_FILE = RAQA_HOME / "env"

# Default DB base folder (databases will be subfolders)
DB_BASE_DIR = RAQA_HOME / "databases"
DB_BASE_DIR.mkdir(parents=True, exist_ok=True)
DB_EMBEDDINGS_CACHE = RAQA_HOME / "embeddings_cache.json"

# Embeddings & chunk config
# EMBEDDING_MODEL = "joeddav/xlm-roberta-large-xnli"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_WINDOW = 1
TOP_K = 20
SIMILARITY_RADIUS = 0.7

# Default markdown folder (can override via CLI)
MARKDOWN_ROOT = "./markdown_files"

# Ensure directories exist
RAQA_HOME.mkdir(parents=True, exist_ok=True)
