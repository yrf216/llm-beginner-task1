from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data_utils import ReviewDataset, load_split
from src.model import TransformerClassifier, TransformerConfig, save_checkpoint
from src.tokenizer import SimpleCharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("ckpt") / "best.pt")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=6000)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--ffn-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_items += labels.size(0)

    return total_loss / max(total_items, 1), total_correct / max(total_items, 1)


def make_scheduler(optimizer: AdamW, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_df = load_split(args.data_dir / "train.parquet")
    dev_df = load_split(args.data_dir / "validation.parquet")

    tokenizer = SimpleCharTokenizer.build(
        train_df["text"].tolist(),
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
    )
    train_set = ReviewDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    dev_set = ReviewDataset(
        texts=dev_df["text"].tolist(),
        labels=dev_df["label"].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size)

    config = TransformerConfig(
        vocab_size=len(tokenizer),
        max_length=args.max_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_hidden_dim=args.ffn_hidden_dim,
        dropout=args.dropout,
        pad_id=tokenizer.pad_id,
    )
    model = TransformerClassifier(config)
    device = torch.device("cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    total_steps = max(len(train_loader) * args.epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = make_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    best_acc = 0.0
    best_metrics = {}
    epochs_without_improvement = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"train={len(train_set)} dev={len(dev_set)} vocab={len(tokenizer)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_items = 0

        for step, (input_ids, labels) in enumerate(train_loader, start=1):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_items += labels.size(0)

            if step % 50 == 0 or step == len(train_loader):
                avg_loss = total_loss / total_items
                avg_acc = total_correct / total_items
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"train_loss={avg_loss:.4f} train_acc={avg_acc:.4f}"
                )

        train_loss = total_loss / total_items
        train_acc = total_correct / total_items
        dev_loss, dev_acc = evaluate(model, dev_loader, device)
        print(
            f"epoch={epoch} done train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"dev_loss={dev_loss:.4f} dev_acc={dev_acc:.4f}"
        )

        if dev_acc > best_acc:
            best_acc = dev_acc
            best_metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "dev_loss": round(dev_loss, 4),
                "dev_acc": round(dev_acc, 4),
            }
            save_checkpoint(args.output, model, tokenizer, metrics=best_metrics)
            print(f"saved best checkpoint to {args.output} with dev_acc={dev_acc:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("early stopping triggered")
                break

    print(f"best_metrics={best_metrics}")


if __name__ == "__main__":
    main()
