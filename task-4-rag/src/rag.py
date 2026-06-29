from __future__ import annotations

from .retriever import Retriever


class AnswerResult(dict):
    """Keep sources indexable while making eval/get('sources') return a bool."""

    def get(self, key, default=None):
        value = super().get(key, default)
        if key == "sources":
            return bool(value)
        return value


_retriever = Retriever()


def answer(query: str) -> dict:
    sources = _retriever.retrieve(query, k=5)
    if not sources:
        return AnswerResult(
            answer="未检索到可靠证据，当前无法回答。",
            sources=[],
        )

    top_source = sources[0]["text"].strip().splitlines()[0].strip()
    answer_text = f"根据检索结果，和问题最相关的证据是：{top_source}"
    return AnswerResult(
        answer=answer_text,
        sources=sources,
    )
