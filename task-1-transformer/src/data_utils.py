from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .tokenizer import SimpleCharTokenizer


def load_split(path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    return frame[["text", "label"]].dropna().reset_index(drop=True)


@dataclass
class EncodedBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor


class ReviewDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: SimpleCharTokenizer,
        max_length: int,
    ):
        self.input_ids = torch.stack([tokenizer.encode(text, max_length) for text in texts])
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.labels[index]
