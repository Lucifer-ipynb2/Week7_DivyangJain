"""
config.py
---------
Central configuration for the RAG system.
All values can be overridden via environment variables (or a .env file).
"""

import os
from dotenv import load_dotenv

# Load variables from a .env file if present
load_dotenv()

# ---- OpenRouter (LLM generation) ----
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# You can swap this for any model available on https://openrouter.ai/models
# A few good free/cheap options:
#   "meta-llama/llama-3.1-8b-instruct:free"
#   "google/gemini-flash-1.5"
#   "openai/gpt-4o-mini"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

# Optional, but OpenRouter recommends sending these for analytics/rankings
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "Simple-RAG-System")

# ---- Embeddings (local, free, no API key needed) ----
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# ---- Chunking ----
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))       # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 80))  # overlap between chunks

# ---- Retrieval ----
TOP_K = int(os.getenv("TOP_K", 4))  # number of chunks to retrieve per query

# ---- Storage paths ----
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "sample_docs")
INDEX_DIR = os.getenv("INDEX_DIR", "vector_index")
