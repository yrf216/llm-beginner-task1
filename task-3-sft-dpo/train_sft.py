from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.bootstrap import MIN_MODEL_MAX_LENGTH, SFT_DEMO_PATH, ensure_local_assets
from src.chat import build_labels, format_messages
from src.lora import inject_lora, save_lora_adapter
from src.runtime import (
    DEFAULT_MODEL_DIR,
    DEFAULT_SFT_DATA_PATH,
    model_load_kwargs,
    require_real_model,
    resolve_data_path,
    resolve_device,
    resolve_dtype,
)


class SFTDataset(Dataset):
    def __init__(self, tokenizer, model_path: Path, records: list[dict], max_length: int):
        self.samples = []
        for record in records:
            text = format_messages(record['messages'])
            encoded = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
            )
            input_ids = encoded.input_ids[0]
            attention_mask = encoded.attention_mask[0]
            labels = build_labels(input_ids, record['messages'], tokenizer=tokenizer, model_path=model_path)
            self.samples.append(
                {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(tokenizer, batch):
    pad_id = tokenizer.pad_token_id
    max_len = max(item['input_ids'].size(0) for item in batch)
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        pad_len = max_len - item['input_ids'].size(0)
        input_ids.append(F.pad(item['input_ids'], (0, pad_len), value=pad_id))
        attention_mask.append(F.pad(item['attention_mask'], (0, pad_len), value=0))
        labels.append(F.pad(item['labels'], (0, pad_len), value=-100))
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels),
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-path', type=Path, default=DEFAULT_SFT_DATA_PATH)
    ap.add_argument('--model-path', type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument('--output-dir', type=Path, default=Path('ckpt/sft'))
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--max-length', type=int, default=MIN_MODEL_MAX_LENGTH)
    ap.add_argument('--lora-r', type=int, default=8)
    ap.add_argument('--lora-alpha', type=float, default=16.0)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    ap.add_argument('--dtype', type=str, default='auto', choices=['auto', 'float32', 'float16', 'bfloat16'])
    ap.add_argument('--gradient-accumulation-steps', type=int, default=8)
    ap.add_argument('--allow-bootstrap', action='store_true')
    ap.add_argument('--allow-demo-data', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.allow_bootstrap or args.allow_demo_data:
        ensure_local_assets()
    args.data_path = resolve_data_path(args.data_path, SFT_DEMO_PATH, allow_demo_data=args.allow_demo_data)
    require_real_model(args.model_path, allow_bootstrap=args.allow_bootstrap)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        **model_load_kwargs(device, dtype),
    )
    inject_lora(
        model,
        ['q_proj', 'v_proj'],
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    model.train()

    records = json.loads(args.data_path.read_text(encoding='utf-8'))
    dataset = SFTDataset(tokenizer, args.model_path, records, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(tokenizer, batch),
    )

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)

    last_loss = None
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            (loss / args.gradient_accumulation_steps).backward()
            if step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item()
        if len(loader) % args.gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        last_loss = epoch_loss / max(1, len(loader))
        print(f'epoch={epoch} loss={last_loss:.4f}')

    metadata = {
        'task': 'sft',
        'loss': round(float(last_loss), 4) if last_loss is not None else None,
        'records': len(records),
        'model_path': str(args.model_path),
        'data_path': str(args.data_path),
        'device': str(device),
        'dtype': str(dtype),
    }
    save_lora_adapter(model, args.output_dir, extra_metadata=metadata)
    print(f'saved SFT adapter to {args.output_dir}')


if __name__ == '__main__':
    main()
