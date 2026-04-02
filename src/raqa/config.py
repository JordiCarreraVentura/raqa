from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

EMBEDDING_MODEL = "sentence-transformers/xlm-roberta-large-xnli"

CHUNK_WINDOW = 3  # sentences left/right
TOP_K = 50
SIMILARITY_RADIUS = 0.4

MARKDOWN_ROOT = "./markdown_files"  # change this