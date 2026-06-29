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
from src.lora import inject_lora, load_lora_adapter
from src.runtime import DEFAULT_MODEL_DIR, model_load_kwargs, resolve_device, resolve_dtype


def build_prompt(user_prompt: str) -> str:
    return format_messages([{'role': 'user', 'content': user_prompt}]) + '<|im_start|>assistant\n'


def generate_text(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument('--adapter-dir', type=Path, default=Path('ckpt/sft'))
    ap.add_argument('--prompt', action='append', dest='prompts')
    ap.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    ap.add_argument('--dtype', type=str, default='auto', choices=['auto', 'float32', 'float16', 'bfloat16'])
    ap.add_argument('--max-new-tokens', type=int, default=128)
    return ap.parse_args()


def main():
    args = parse_args()
    prompts = args.prompts or ['什么是 LoRA？', '请用两句话介绍深度学习。']
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))

    base = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        **model_load_kwargs(device, dtype),
    )
    tuned = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        **model_load_kwargs(device, dtype),
    )
    inject_lora(tuned, ['q_proj', 'v_proj'], r=8, alpha=16, dropout=0.05)
    if args.adapter_dir.exists():
        load_lora_adapter(tuned, args.adapter_dir)

    base.to(device).eval()
    tuned.to(device).eval()

    for prompt_text in prompts:
        prompt = build_prompt(prompt_text)
        print(f'PROMPT: {prompt_text}')
        print('BASE:')
        print(generate_text(base, tokenizer, prompt, device, args.max_new_tokens))
        print('TUNED:')
        print(generate_text(tuned, tokenizer, prompt, device, args.max_new_tokens))
        print('-' * 80)


if __name__ == '__main__':
    main()
