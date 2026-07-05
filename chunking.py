"""
chunking.py
-----------
Stage 2: Text Chunking
Splits long documents into smaller overlapping chunks to improve retrieval
accuracy (small, focused chunks embed and match better than whole documents).
"""

from typing import List, Dict


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split `text` into overlapping chunks of roughly `chunk_size` characters.

    A simple, dependency-free recursive-ish splitter:
    1. Split on paragraphs first.
    2. Greedily pack paragraphs into chunks up to chunk_size.
    3. If a single paragraph is longer than chunk_size, hard-split it.
    4. Add `overlap` characters from the end of the previous chunk to the
       start of the next chunk, so context isn't lost at chunk boundaries.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    # Normalize whitespace, split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current = ""

    def flush(curr: str):
        if curr.strip():
            chunks.append(curr.strip())

    for para in paragraphs:
        # Hard-split very long paragraphs
        while len(para) > chunk_size:
            piece = para[:chunk_size]
            if current:
                flush(current)
                current = ""
            chunks.append(piece.strip())
            para = para[chunk_size - overlap:]

        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current} {para}".strip()
        else:
            flush(current)
            # start new chunk, carry over overlap from previous chunk
            carry_over = current[-overlap:] if overlap and current else ""
            current = f"{carry_over} {para}".strip()

    flush(current)
    return chunks


def chunk_documents(
    documents: List[Dict[str, str]], chunk_size: int, overlap: int
) -> List[Dict[str, str]]:
    """
    Chunk a list of loaded documents.

    Returns:
        List of dicts: [{"source": filename, "chunk_id": int, "text": chunk}, ...]
    """
    all_chunks = []
    for doc in documents:
        pieces = chunk_text(doc["text"], chunk_size, overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "source": doc["source"],
                "chunk_id": i,
                "text": piece,
            })
    return all_chunks
