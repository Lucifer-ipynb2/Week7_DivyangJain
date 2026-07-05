"""
document_loader.py
-------------------
Stage 1: Document Ingestion
Loads PDFs and text files from a folder and converts them into raw text.
"""

import os
from typing import List, Dict
from pypdf import PdfReader


def load_pdf(path: str) -> str:
    """Extract raw text from a PDF file."""
    text_parts = []
    reader = PdfReader(path)
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts)


def load_txt(path: str) -> str:
    """Read raw text from a .txt or .md file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_documents(folder: str) -> List[Dict[str, str]]:
    """
    Load every supported document (.pdf, .txt, .md) inside `folder`.

    Returns:
        List of dicts: [{"source": filename, "text": raw_text}, ...]
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Documents folder not found: {folder}")

    documents = []
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            continue

        ext = filename.lower().split(".")[-1]
        try:
            if ext == "pdf":
                text = load_pdf(path)
            elif ext in ("txt", "md"):
                text = load_txt(path)
            else:
                continue  # skip unsupported file types

            if text.strip():
                documents.append({"source": filename, "text": text})
            else:
                print(f"[warn] No extractable text found in: {filename}")
        except Exception as e:
            print(f"[warn] Failed to load {filename}: {e}")

    if not documents:
        raise ValueError(
            f"No readable .pdf/.txt/.md files found in '{folder}'. "
            "Add some documents and try again."
        )

    return documents
