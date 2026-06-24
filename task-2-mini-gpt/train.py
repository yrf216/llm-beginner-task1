from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from src.model import MiniGPT, MiniGPTConfig, save_checkpoint
from src.tokenizer import BPETokenizer


class TextChunkDataset(Dataset):
    def __init__(self, token_ids: list[int], block_size: int, stride: int | None = None):
        stride = stride or max(1, block_size // 2)
        self.inputs = []
        self.targets = []
        for i in range(0, len(token_ids) - block_size - 1, stride):
            chunk = token_ids[i : i + block_size + 1]
            if len(chunk) < block_size + 1:
                continue
            self.inputs.append(torch.tensor(chunk[:-1], dtype=torch.long))
            self.targets.append(torch.tensor(chunk[1:], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--ckpt-dir", type=Path, default=Path("ckpt"))
    ap.add_argument("--vocab-size", type=int, default=384)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--ffn-hidden-dim", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model: MiniGPT, loader: DataLoader, device: torch.device):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
            total_nll += loss.item()
            total_tokens += y.numel()
    avg_nll = total_nll / max(1, total_tokens)
    ppl = math.exp(avg_nll)
    return avg_nll, ppl


def main():
    args = parse_args()
    set_seed(args.seed)

    train_text = (args.data_dir / "train.txt").read_text(encoding="utf-8")
    dev_text = (args.data_dir / "dev.txt").read_text(encoding="utf-8")

    tokenizer_corpus = train_text + "\n" + dev_text
    tokenizer = BPETokenizer.train(tokenizer_corpus, vocab_size=args.vocab_size)
    train_ids = tokenizer.encode(train_text)
    dev_ids = tokenizer.encode(dev_text)

    train_set = TextChunkDataset(train_ids, block_size=args.max_seq_len, stride=max(1, args.max_seq_len // 4))
    dev_set = TextChunkDataset(dev_ids, block_size=args.max_seq_len, stride=max(1, args.max_seq_len // 2))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size)

    config = MiniGPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_hidden_dim=args.ffn_hidden_dim,
        dropout=args.dropout,
    )
    model = MiniGPT(config)
    device = torch.device("cpu")
    model.to(device)

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.ckpt_dir / "tokenizer.json"
    tokenizer.save_pretrained(tokenizer_path)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = make_scheduler(optimizer, total_steps=total_steps, warmup_steps=int(total_steps * args.warmup_ratio))

    best_ppl = float("inf")
    best_metrics = {}
    ckpt_path = args.ckpt_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            if step % 20 == 0 or step == len(train_loader):
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"train_nll={total_loss / max(1, total_tokens):.4f}"
                )

        dev_nll, dev_ppl = evaluate(model, dev_loader, device)
        train_nll = total_loss / max(1, total_tokens)
        print(
            f"epoch={epoch} train_nll={train_nll:.4f} "
            f"dev_nll={dev_nll:.4f} dev_ppl={dev_ppl:.2f}"
        )

        if dev_ppl < best_ppl:
            best_ppl = dev_ppl
            best_metrics = {
                "epoch": epoch,
                "train_nll": round(train_nll, 4),
                "dev_nll": round(dev_nll, 4),
                "dev_ppl": round(dev_ppl, 2),
            }
            save_checkpoint(ckpt_path, model, config, tokenizer_path, best_metrics)
            print(f"saved best checkpoint to {ckpt_path}")

    print(f"best_metrics={best_metrics}")


if __name__ == "__main__":
    main()
