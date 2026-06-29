from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models' / 'Qwen2.5-0.5B'
IM_START = '<|im_start|>'
IM_END = '<|im_end|>'


@lru_cache(maxsize=None)
def _tokenizer(model_ref: str):
    return AutoTokenizer.from_pretrained(model_ref, trust_remote_code=False)


def format_messages(messages: list[dict]) -> str:
    chunks = []
    for message in messages:
        role = str(message['role']).strip()
        content = str(message['content']).strip()
        chunks.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    return ''.join(chunks)


def _assistant_spans(messages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for message in messages:
        role = str(message['role']).strip()
        content = str(message['content']).strip()
        prefix = f"{IM_START}{role}\n"
        suffix = f"{IM_END}\n"
        parts.append(prefix)
        cursor += len(prefix)
        if role == 'assistant':
            spans.append((cursor, cursor + len(content)))
        parts.append(content)
        cursor += len(content)
        parts.append(suffix)
        cursor += len(suffix)
    return ''.join(parts), spans


def build_labels(
    input_ids: torch.Tensor,
    messages: list[dict],
    tokenizer=None,
    model_path: str | Path | None = None,
) -> torch.Tensor:
    if tokenizer is None:
        tokenizer = _tokenizer(str(model_path or MODEL_DIR))
    text, spans = _assistant_spans(messages)
    max_len = int(input_ids.numel())
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
    )
    encoded_ids = encoded['input_ids']
    if encoded_ids != input_ids.tolist():
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )
        encoded_ids = encoded['input_ids']
        if encoded_ids != input_ids.tolist():
            raise ValueError('build_labels tokenization mismatch with format_messages output')
    labels = []
    for token_id, (start, end) in zip(encoded_ids, encoded['offset_mapping']):
        supervised = False
        if end > start:
            for span_start, span_end in spans:
                if start < span_end and end > span_start:
                    supervised = True
                    break
        labels.append(token_id if supervised else -100)
    return torch.tensor(labels, dtype=torch.long)
