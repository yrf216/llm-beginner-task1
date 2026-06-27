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

from src.bootstrap import MIN_MODEL_MAX_LENGTH, ensure_local_assets
from src.chat import format_messages
from src.lora import inject_lora, load_lora_adapter, save_lora_adapter


def masked_log_prob(model, input_ids, attention_mask, prompt_lens: torch.Tensor):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    target = input_ids[:, 1:]
    token_log_probs = F.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    mask = attention_mask[:, 1:].float()
    positions = torch.arange(mask.size(1), device=mask.device).unsqueeze(0)
    prompt_offsets = (prompt_lens.to(mask.device) - 1).clamp_min(0).unsqueeze(1)
    mask = mask * (positions >= prompt_offsets).float()
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (token_log_probs * mask).sum(dim=1) / denom


class DPODataset(Dataset):
    def __init__(self, tokenizer, records: list[dict], max_length: int):
        self.samples = []
        for record in records:
            prompt = format_messages(record['messages'])
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            chosen_text = prompt + f"<|im_start|>assistant\n{record['chosen']}<|im_end|>\n"
            rejected_text = prompt + f"<|im_start|>assistant\n{record['rejected']}<|im_end|>\n"
            chosen = tokenizer(chosen_text, return_tensors='pt', truncation=True, max_length=max_length)
            rejected = tokenizer(rejected_text, return_tensors='pt', truncation=True, max_length=max_length)
            prompt_len = min(len(prompt_ids), chosen.input_ids.size(1))
            self.samples.append(
                {
                    'chosen_ids': chosen.input_ids[0],
                    'chosen_mask': chosen.attention_mask[0],
                    'rejected_ids': rejected.input_ids[0],
                    'rejected_mask': rejected.attention_mask[0],
                    'prompt_len': prompt_len,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _pad(batch_tensors, pad_value):
    max_len = max(t.size(0) for t in batch_tensors)
    return torch.stack([F.pad(t, (0, max_len - t.size(0)), value=pad_value) for t in batch_tensors])


def collate_fn(tokenizer, batch):
    return {
        'chosen_ids': _pad([x['chosen_ids'] for x in batch], tokenizer.pad_token_id),
        'chosen_mask': _pad([x['chosen_mask'] for x in batch], 0),
        'rejected_ids': _pad([x['rejected_ids'] for x in batch], tokenizer.pad_token_id),
        'rejected_mask': _pad([x['rejected_mask'] for x in batch], 0),
        'prompt_len': torch.tensor([x['prompt_len'] for x in batch], dtype=torch.long),
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-path', type=Path, default=Path('data/dpo_demo.json'))
    ap.add_argument('--model-path', type=Path, default=Path('models/Qwen2.5-0.5B'))
    ap.add_argument('--sft-adapter', type=Path, default=Path('ckpt/sft'))
    ap.add_argument('--output-dir', type=Path, default=Path('ckpt/dpo'))
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--beta', type=float, default=0.1)
    ap.add_argument('--max-length', type=int, default=MIN_MODEL_MAX_LENGTH)
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    assets = ensure_local_assets()
    if not args.data_path.exists():
        args.data_path = assets['dpo_data']
    if not args.model_path.exists():
        args.model_path = assets['model_dir']

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    policy = AutoModelForCausalLM.from_pretrained(str(args.model_path))
    reference = AutoModelForCausalLM.from_pretrained(str(args.model_path))
    inject_lora(policy, ['q_proj', 'v_proj'], r=8, alpha=16)
    if args.sft_adapter.exists():
        load_lora_adapter(policy, args.sft_adapter)
        inject_lora(reference, ['q_proj', 'v_proj'], r=8, alpha=16)
        load_lora_adapter(reference, args.sft_adapter)
    else:
        inject_lora(reference, ['q_proj', 'v_proj'], r=8, alpha=16)
    for param in reference.parameters():
        param.requires_grad = False
    reference.eval()
    policy.train()

    records = json.loads(args.data_path.read_text(encoding='utf-8'))
    dataset = DPODataset(tokenizer, records, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(tokenizer, batch),
    )

    optimizer = AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)
    device = torch.device('cpu')
    policy.to(device)
    reference.to(device)

    last_loss = None
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            chosen_pi = masked_log_prob(policy, batch['chosen_ids'], batch['chosen_mask'], batch['prompt_len'])
            rejected_pi = masked_log_prob(policy, batch['rejected_ids'], batch['rejected_mask'], batch['prompt_len'])
            with torch.no_grad():
                chosen_ref = masked_log_prob(reference, batch['chosen_ids'], batch['chosen_mask'], batch['prompt_len'])
                rejected_ref = masked_log_prob(reference, batch['rejected_ids'], batch['rejected_mask'], batch['prompt_len'])
            preference_logits = (chosen_pi - rejected_pi) - (chosen_ref - rejected_ref)
            loss = -F.logsigmoid(args.beta * preference_logits).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        last_loss = epoch_loss / max(1, len(loader))
        print(f'epoch={epoch} dpo_loss={last_loss:.4f}')

    metadata = {
        'task': 'dpo',
        'loss': round(float(last_loss), 4) if last_loss is not None else None,
        'records': len(records),
        'beta': args.beta,
    }
    save_lora_adapter(policy, args.output_dir, extra_metadata=metadata)
    print(f'saved DPO adapter to {args.output_dir}')


if __name__ == '__main__':
    main()
