"""
rag_pipeline.py
----------------
Ties together every stage of the RAG system:
Ingestion -> Chunking -> Embedding -> Vector Store -> Retrieval -> Generation
"""

import os
from typing import List, Dict

import config
from document_loader import load_documents
from chunking import chunk_documents
from embeddings import EmbeddingModel
from vector_store import VectorStore
from openrouter_client import generate_answer


class RAGPipeline:
    def __init__(self):
        self.embedder = EmbeddingModel(config.EMBEDDING_MODEL_NAME)
        self.store = VectorStore()

    # ---------- Index building ----------

    def build_index(self, docs_folder: str = None, save: bool = True):
        """Run ingestion -> chunking -> embedding -> index build."""
        docs_folder = docs_folder or config.DOCS_FOLDER

        print(f"[1/4] Loading documents from '{docs_folder}'...")
        documents = load_documents(docs_folder)
        print(f"      Loaded {len(documents)} document(s).")

        print("[2/4] Chunking documents...")
        chunks = chunk_documents(documents, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        print(f"      Created {len(chunks)} chunk(s).")

        print("[3/4] Creating embeddings...")
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed(texts)

        print("[4/4] Building vector index...")
        self.store.build(embeddings, chunks)

        if save:
            self.store.save(config.INDEX_DIR)
            print(f"      Index saved to '{config.INDEX_DIR}/'.")

        print("Done. Index is ready for queries.\n")

    def load_index(self, index_dir: str = None):
        index_dir = index_dir or config.INDEX_DIR
        self.store.load(index_dir)
        print(f"[info] Loaded existing index from '{index_dir}' "
              f"({len(self.store.metadata)} chunks).")

    # ---------- Querying ----------

    def retrieve(self, question: str, top_k: int = None) -> List[Dict]:
        top_k = top_k or config.TOP_K
        query_embedding = self.embedder.embed_query(question)
        results = self.store.search(query_embedding, top_k)
        return [chunk for chunk, score in results], [score for chunk, score in results]

    def answer(self, question: str, top_k: int = None) -> Dict:
        """
        Full query pipeline: embed question -> retrieve chunks -> generate answer.
        Returns a dict with the answer and the sources used.
        """
        chunks, scores = self.retrieve(question, top_k)

        if not chunks:
            return {
                "answer": "I couldn't find any relevant information in the documents.",
                "sources": [],
            }

        answer_text = generate_answer(question, chunks)

        sources = [
            {"source": c["source"], "chunk_id": c["chunk_id"], "score": round(s, 3)}
            for c, s in zip(chunks, scores)
        ]

        return {"answer": answer_text, "sources": sources}
