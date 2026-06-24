from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class BPETokenizer:
    def __init__(
        self,
        merges: list[list[str]],
        vocab: dict[str, int],
        token_to_bytes: dict[int, bytes] | None = None,
    ):
        self.merges = [tuple(pair) for pair in merges]
        self.vocab = {token: int(idx) for token, idx in vocab.items()}
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}
        if token_to_bytes:
            first_value = next(iter(token_to_bytes.values()))
            if isinstance(first_value, str):
                self.token_to_bytes = {int(k): bytes.fromhex(v) for k, v in token_to_bytes.items()}
            else:
                self.token_to_bytes = {int(k): v for k, v in token_to_bytes.items()}
        else:
            self.token_to_bytes = {idx: bytes.fromhex(token) for token, idx in self.vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 384,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        data = [bytes([b]) for b in text.encode("utf-8")]
        vocab = {bytes([i]).hex(): i for i in range(256)}
        merges: list[list[str]] = []

        while len(vocab) < vocab_size:
            pair_counts = Counter()
            for a, b in zip(data, data[1:]):
                pair_counts[(a, b)] += 1

            if not pair_counts:
                break
            (left, right), freq = pair_counts.most_common(1)[0]
            if freq < min_frequency:
                break

            merged = left + right
            merged_hex = merged.hex()
            if merged_hex in vocab:
                break

            merges.append([left.hex(), right.hex()])
            vocab[merged_hex] = len(vocab)

            new_data: list[bytes] = []
            i = 0
            while i < len(data):
                if i < len(data) - 1 and data[i] == left and data[i + 1] == right:
                    new_data.append(merged)
                    i += 2
                else:
                    new_data.append(data[i])
                    i += 1
            data = new_data

        token_to_bytes = {idx: bytes.fromhex(token_hex) for token_hex, idx in vocab.items()}
        return cls(merges=merges, vocab=vocab, token_to_bytes=token_to_bytes)

    def _apply_merges(self, tokens: list[bytes]) -> list[bytes]:
        if not tokens:
            return tokens

        while True:
            best_index = None
            best_rank = None
            best_pair = None
            for i in range(len(tokens) - 1):
                pair = (tokens[i].hex(), tokens[i + 1].hex())
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = i
                    best_pair = pair

            if best_index is None or best_pair is None:
                break

            merged = bytes.fromhex(best_pair[0]) + bytes.fromhex(best_pair[1])
            tokens = tokens[:best_index] + [merged] + tokens[best_index + 2 :]
        return tokens

    def encode(self, text: str) -> list[int]:
        tokens = [bytes([b]) for b in text.encode("utf-8")]
        tokens = self._apply_merges(tokens)
        ids: list[int] = []
        for token in tokens:
            token_hex = token.hex()
            if token_hex in self.vocab:
                ids.append(self.vocab[token_hex])
            else:
                ids.extend(self.vocab[bytes([b]).hex()] for b in token)
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.token_to_bytes[int(idx)] for idx in ids)
        return raw.decode("utf-8", errors="strict")

    def save_pretrained(self, path: str | Path) -> None:
        payload = {
            "merges": self.merges,
            "vocab": self.vocab,
            "token_to_bytes": {str(idx): data.hex() for idx, data in self.token_to_bytes.items()},
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            merges=payload["merges"],
            vocab=payload["vocab"],
            token_to_bytes=payload.get("token_to_bytes"),
        )
