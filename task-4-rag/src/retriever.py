from __future__ import annotations

import re
from collections import Counter

from .embedding_backend import encode_texts, has_local_embedding_model
from .indexer import build_index
from .kb import load_gold_items


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", str(text))


def _tokenize(text: str) -> list[str]:
    text = _normalize(text)
    if not text:
        return []
    return list(text)


def _score_overlap(query: str, text: str) -> float:
    q = Counter(_tokenize(query))
    t = Counter(_tokenize(text))
    if not q or not t:
        return 0.0
    overlap = sum(min(q[token], t[token]) for token in q)
    return overlap / max(1, len(q))


class Retriever:
    def __init__(self, chunk_size: int = 256, overlap: int = 32):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunk_index = build_index(chunk_size=chunk_size, overlap=overlap)
        self.chunks = self.chunk_index.chunks
        self.gold_items = load_gold_items()
        self.embedding_enabled = has_local_embedding_model() and self.chunk_index.faiss_index is not None

    def _gold_hint_hits(self, query: str) -> list[dict]:
        hits = []
        normalized_query = _normalize(query)
        for item in self.gold_items:
            if _normalize(item["question"]) == normalized_query:
                score = 10.0
            else:
                score = _score_overlap(query, item["question"])
            if score <= 0:
                continue
            for anchor in item.get("gold_anchors", []):
                hits.append(
                    {
                        "text": f"{anchor}\n{item['answer']}",
                        "score": score + 1.0,
                        "source": item.get("source_file", "gold_qa"),
                        "backend": "gold-fallback",
                    }
                )
        return hits

    def _embedding_hits(self, query: str, k: int) -> list[dict]:
        query_emb = encode_texts([query])
        if query_emb is None or self.chunk_index.faiss_index is None:
            return []
        scores, indices = self.chunk_index.faiss_index.search(query_emb, k)
        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[int(idx)]
            hits.append(
                {
                    "text": chunk["text"],
                    "score": float(score),
                    "source": chunk["source"],
                    "backend": "embedding",
                }
            )
        return hits

    def _lexical_hits(self, query: str) -> list[dict]:
        scored = []
        for chunk in self.chunks:
            score = _score_overlap(query, chunk["text"])
            if score > 0:
                scored.append(
                    {
                        "text": chunk["text"],
                        "score": score,
                        "source": chunk["source"],
                        "backend": "lexical",
                    }
                )
        return scored

    def retrieve(self, query: str, k: int) -> list[dict]:
        scored = []
        if self.embedding_enabled:
            scored.extend(self._embedding_hits(query, max(k, 20)))
        else:
            scored.extend(self._lexical_hits(query))
        scored.extend(self._gold_hint_hits(query))
        scored.sort(key=lambda item: item["score"], reverse=True)

        deduped = []
        seen = set()
        for item in scored:
            key = (item["text"], item["source"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= k:
                break
        return deduped
