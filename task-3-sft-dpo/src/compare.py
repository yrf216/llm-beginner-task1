from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chat import format_messages
from src.bootstrap import ensure_local_assets
from src.lora import inject_lora, load_lora_adapter


def build_prompt(user_prompt: str) -> str:
    return format_messages([{'role': 'user', 'content': user_prompt}]) + '<|im_start|>assistant\n'


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return full[len(prompt):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter-dir', type=Path, default=Path('ckpt/sft'))
    ap.add_argument('--prompt', type=str, default='什么是 LoRA？')
    args = ap.parse_args()

    assets = ensure_local_assets()
    model_path = assets['model_dir']
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    prompt = build_prompt(args.prompt)

    base = AutoModelForCausalLM.from_pretrained(str(model_path))
    tuned = AutoModelForCausalLM.from_pretrained(str(model_path))
    inject_lora(tuned, ['q_proj', 'v_proj'], r=8, alpha=16)
    if args.adapter_dir.exists():
        load_lora_adapter(tuned, args.adapter_dir)

    print('PROMPT:')
    print(prompt)
    print('\nBASE:')
    print(generate_text(base, tokenizer, prompt))
    print('\nTUNED:')
    print(generate_text(tuned, tokenizer, prompt))


if __name__ == '__main__':
    main()
