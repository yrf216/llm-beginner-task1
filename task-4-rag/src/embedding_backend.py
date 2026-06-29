from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
EMBED_MODEL_DIR = MODELS_DIR / "bge-small-zh-v1.5"


def has_local_embedding_model() -> bool:
    return (EMBED_MODEL_DIR / "config.json").exists()


@lru_cache(maxsize=1)
def get_sentence_transformer():
    if not has_local_embedding_model():
        return None
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(EMBED_MODEL_DIR))


def encode_texts(texts: list[str]) -> np.ndarray | None:
    model = get_sentence_transformer()
    if model is None:
        return None
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")
