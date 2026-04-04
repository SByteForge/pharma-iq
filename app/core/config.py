import os
from pathlib import Path

# All configuration in one place.
# Change behaviour via environment variables — no code changes needed.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "./chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pharma_contracts")
TOP_K = int(os.getenv("TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
DB_PATH = Path(os.getenv("DB_PATH", "./pharma_iq.db"))
MAX_TURNS = int(os.getenv("MAX_TURNS", "10"))