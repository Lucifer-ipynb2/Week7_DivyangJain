"""
openrouter_client.py
---------------------
Stage 6: Answer Generation
Sends the retrieved context + user question to an LLM via the OpenRouter API
(OpenAI-compatible /chat/completions endpoint) and returns the generated answer.
"""

import requests
from typing import List, Dict

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    OPENROUTER_SITE_URL,
    OPENROUTER_APP_NAME,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context from the user's documents. "
    "If the answer is not contained in the context, say you don't know "
    "based on the provided documents instead of making something up. "
    "Keep answers concise and cite which source(s) you used when relevant."
)


def build_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    """Assemble the retrieved chunks into a context block for the LLM."""
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"[Source {i}: {chunk['source']} | chunk {chunk['chunk_id']}]\n{chunk['text']}"
        )
    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = (
        f"Context from documents:\n\n{context_text}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Answer the question using only the context above."
    )
    return prompt


def generate_answer(question: str, retrieved_chunks: List[Dict]) -> str:
    """Call the OpenRouter chat completions API and return the model's answer."""
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file "
            "(see .env.example) or export it as an environment variable."
        )

    user_prompt = build_prompt(question, retrieved_chunks)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Optional but recommended by OpenRouter:
        "HTTP-Referer": OPENROUTER_SITE_URL,
        "X-Title": OPENROUTER_APP_NAME,
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter API error ({response.status_code}): {response.text}"
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected OpenRouter response format: {data}") from e
