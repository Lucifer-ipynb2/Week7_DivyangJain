"""
embeddings.py
-------------
Stage 3: Embedding Creation
Converts text chunks into vector representations using a local
sentence-transformers model (free, runs on CPU, no API key required).
"""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str):
        print(f"[info] Loading embedding model '{model_name}' (first run downloads it)...")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of strings into a 2D numpy array of embeddings,
        normalized to unit length (so dot product == cosine similarity).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 20,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
