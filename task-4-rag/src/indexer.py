from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embedding_backend import encode_texts
from .kb import build_chunks


@dataclass
class ChunkIndex:
    chunks: list[dict]
    embeddings: np.ndarray | None
    faiss_index: object | None


def build_index(chunk_size: int = 256, overlap: int = 32) -> ChunkIndex:
    chunks = build_chunks(chunk_size=chunk_size, overlap=overlap)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = encode_texts(texts)
    if embeddings is None:
        return ChunkIndex(chunks=chunks, embeddings=None, faiss_index=None)

    import faiss

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return ChunkIndex(chunks=chunks, embeddings=embeddings, faiss_index=index)
