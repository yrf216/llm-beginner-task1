from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    text = str(text or "")
    if not text:
        return []

    step = chunk_size - overlap
    chunks: List[str] = []
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        if not chunk:
            continue
        chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks
