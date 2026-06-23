from __future__ import annotations

from collections import Counter

import torch

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"


def normalize_text(text: str) -> str:
    text = "" if text is None else str(text)
    return " ".join(text.strip().split())


class SimpleCharTokenizer:
    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {idx: token for token, idx in token_to_id.items()}
        self.pad_id = token_to_id[PAD_TOKEN]
        self.unk_id = token_to_id[UNK_TOKEN]
        self.cls_id = token_to_id[CLS_TOKEN]

    @classmethod
    def build(
        cls,
        texts,
        min_freq: int = 1,
        max_vocab_size: int | None = None,
    ) -> "SimpleCharTokenizer":
        counter = Counter()
        for text in texts:
            counter.update(list(normalize_text(text)))

        specials = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN]
        vocab = {token: idx for idx, token in enumerate(specials)}
        for token, freq in counter.most_common():
            if freq < min_freq or token in vocab:
                continue
            vocab[token] = len(vocab)
            if max_vocab_size is not None and len(vocab) >= max_vocab_size:
                break
        return cls(vocab)

    def encode(self, text: str, max_length: int) -> torch.Tensor:
        chars = list(normalize_text(text))
        chars = chars[: max_length - 1]
        ids = [self.cls_id] + [self.token_to_id.get(ch, self.unk_id) for ch in chars]
        ids += [self.pad_id] * (max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def tokens(self, text: str, max_length: int) -> list[str]:
        chars = list(normalize_text(text))
        chars = chars[: max_length - 1]
        return [CLS_TOKEN] + chars

    def __len__(self) -> int:
        return len(self.token_to_id)
