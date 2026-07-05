"""
vector_store.py
----------------
Stage 4: Vector Database
Stores chunk embeddings in a FAISS index for fast similarity search,
along with metadata (source file + chunk text) so retrieved vectors
can be mapped back to readable content.
"""

import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Tuple


class VectorStore:
    def __init__(self, dim: int = None):
        self.dim = dim
        self.index = None
        self.metadata: List[Dict] = []  # parallel list: metadata[i] <-> vector i

    def build(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Create a fresh index from embeddings + matching metadata."""
        assert embeddings.shape[0] == len(metadata), "Embeddings/metadata length mismatch"
        self.dim = embeddings.shape[1]
        # Inner product on normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        self.metadata = metadata

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[Dict, float]]:
        """Return top_k (metadata, score) pairs most similar to the query embedding."""
        if self.index is None:
            raise RuntimeError("Vector store is empty. Build or load an index first.")

        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, folder: str):
        index_path = os.path.join(folder, "index.faiss")
        meta_path = os.path.join(folder, "metadata.pkl")
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"No saved index found in '{folder}'.")
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        self.dim = self.index.d
