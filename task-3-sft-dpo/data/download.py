from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset, load_from_disk
from huggingface_hub import snapshot_download

from src.bootstrap import DPO_DEMO_PATH, SFT_DEMO_PATH, ensure_local_assets
from src.runtime import (
    DEFAULT_DPO_DATA_PATH,
    DEFAULT_MODEL_DIR,
    DEFAULT_SFT_DATA_PATH,
    is_real_qwen_model,
)

DEFAULT_MODEL_ID = 'Qwen/Qwen2.5-0.5B'
DEFAULT_SFT_SOURCE = 'OpenMOSS-Team/moss-003-sft-data'
DEFAULT_DPO_SOURCE = 'argilla/ultrafeedback-binarized-preferences-cleaned'


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-id', type=str, default=DEFAULT_MODEL_ID)
    ap.add_argument('--model-path', type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument('--skip-model', action='store_true')
    ap.add_argument('--prepare-demo', action='store_true')
    ap.add_argument('--sft-source', type=str, default=DEFAULT_SFT_SOURCE)
    ap.add_argument('--sft-split', type=str, default='train')
    ap.add_argument('--sft-output', type=Path, default=DEFAULT_SFT_DATA_PATH)
    ap.add_argument('--max-sft-samples', type=int, default=20000)
    ap.add_argument('--skip-sft', action='store_true')
    ap.add_argument('--dpo-source', type=str, default=DEFAULT_DPO_SOURCE)
    ap.add_argument('--dpo-split', type=str, default='train')
    ap.add_argument('--dpo-output', type=Path, default=DEFAULT_DPO_DATA_PATH)
    ap.add_argument('--max-dpo-samples', type=int, default=10000)
    ap.add_argument('--skip-dpo', action='store_true')
    return ap.parse_args()


def normalize_role(role: str) -> str:
    role = str(role).strip().lower()
    mapping = {
        'human': 'user',
        'user': 'user',
        'instruction': 'user',
        'question': 'user',
        'gpt': 'assistant',
        'assistant': 'assistant',
        'bot': 'assistant',
        'system': 'system',
    }
    return mapping.get(role, role)


def clean_chat_text(text: str) -> str:
    cleaned = str(text)
    markers = [
        '<|Human|>:',
        '<|MOSS|>:',
        '<|Inner Thoughts|>:',
        '<|Commands|>:',
        '<|Results|>:',
        '<eoh>',
        '<eom>',
        '<eot>',
        '<eoc>',
        '<eor>',
    ]
    for marker in markers:
        cleaned = cleaned.replace(marker, '')
    return cleaned.strip()


def extract_content(message) -> str:
    if isinstance(message, str):
        return message.strip()
    if isinstance(message, dict):
        for key in ('content', 'value', 'text', 'response', 'output'):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ''


def normalize_messages(raw_messages) -> list[dict]:
    messages = []
    if isinstance(raw_messages, dict):
        turn_keys = sorted(
            [key for key in raw_messages.keys() if str(key).startswith('turn_')],
            key=lambda key: int(str(key).split('_')[-1]),
        )
        for turn_key in turn_keys:
            turn = raw_messages.get(turn_key)
            if not isinstance(turn, dict):
                continue
            human = clean_chat_text(turn.get('Human', ''))
            assistant = clean_chat_text(turn.get('MOSS', ''))
            if human:
                messages.append({'role': 'user', 'content': human})
            if assistant:
                messages.append({'role': 'assistant', 'content': assistant})
        return messages

    for item in raw_messages or []:
        if isinstance(item, dict) and 'human' in item and 'assistant' in item:
            human = str(item['human']).strip()
            assistant = str(item['assistant']).strip()
            if human:
                messages.append({'role': 'user', 'content': human})
            if assistant:
                messages.append({'role': 'assistant', 'content': assistant})
            continue
        if isinstance(item, dict):
            role = normalize_role(item.get('role') or item.get('from') or item.get('speaker'))
            content = extract_content(item)
            if role in {'user', 'assistant', 'system'} and content:
                messages.append({'role': role, 'content': content})
    return messages


def load_any_dataset(source: str, split: str):
    path = Path(source)
    if path.exists():
        if path.is_dir() and (path / 'dataset_info.json').exists():
            return load_from_disk(str(path))
        suffix = path.suffix.lower()
        if suffix in {'.json', '.jsonl'}:
            return load_dataset('json', data_files=str(path), split='train')
        if suffix == '.parquet':
            return load_dataset('parquet', data_files=str(path), split='train')
        raise ValueError(f'Unsupported local dataset format: {path}')
    return load_dataset(source, split=split, streaming=True)


def normalize_sft_record(example: dict) -> dict | None:
    raw_messages = (
        example.get('messages')
        or example.get('conversations')
        or example.get('conversation')
        or example.get('chat')
    )
    if raw_messages:
        messages = normalize_messages(raw_messages)
        if len(messages) >= 2 and any(m['role'] == 'assistant' for m in messages):
            system = example.get('meta_instruction') or example.get('system') or example.get('system_prompt')
            if isinstance(system, str) and system.strip():
                messages = [{'role': 'system', 'content': system.strip()}] + messages
            return {'messages': messages}

    instruction = example.get('instruction') or example.get('prompt') or example.get('question')
    output = example.get('output') or example.get('response') or example.get('answer')
    if isinstance(instruction, str) and isinstance(output, str) and instruction.strip() and output.strip():
        system = example.get('system') or example.get('system_prompt')
        messages = []
        if isinstance(system, str) and system.strip():
            messages.append({'role': 'system', 'content': system.strip()})
        messages.append({'role': 'user', 'content': instruction.strip()})
        messages.append({'role': 'assistant', 'content': output.strip()})
        return {'messages': messages}
    return None


def split_preference_transcripts(chosen, rejected) -> tuple[list[dict], str, str] | None:
    chosen_messages = normalize_messages(chosen)
    rejected_messages = normalize_messages(rejected)
    if not chosen_messages or not rejected_messages:
        return None
    prefix = []
    idx = 0
    while idx < len(chosen_messages) and idx < len(rejected_messages) and chosen_messages[idx] == rejected_messages[idx]:
        prefix.append(chosen_messages[idx])
        idx += 1
    chosen_suffix = chosen_messages[idx:]
    rejected_suffix = rejected_messages[idx:]
    if not chosen_suffix or not rejected_suffix:
        return None
    if chosen_suffix[-1]['role'] != 'assistant' or rejected_suffix[-1]['role'] != 'assistant':
        return None
    return prefix, chosen_suffix[-1]['content'], rejected_suffix[-1]['content']


def normalize_dpo_record(example: dict) -> dict | None:
    split_from_transcripts = split_preference_transcripts(
        example.get('chosen'),
        example.get('rejected'),
    )
    if split_from_transcripts is not None:
        prompt_messages, chosen_text, rejected_text = split_from_transcripts
        if prompt_messages:
            return {
                'messages': prompt_messages,
                'chosen': chosen_text,
                'rejected': rejected_text,
            }

    raw_messages = example.get('messages') or example.get('prompt_messages')
    if raw_messages:
        prompt_messages = normalize_messages(raw_messages)
    else:
        prompt = example.get('prompt') or example.get('instruction') or example.get('question')
        prompt_messages = [{'role': 'user', 'content': str(prompt).strip()}] if isinstance(prompt, str) and prompt.strip() else []

    chosen = example.get('chosen_response') or example.get('chosen')
    rejected = example.get('rejected_response') or example.get('rejected')
    chosen_text = extract_content(chosen)
    rejected_text = extract_content(rejected)
    if prompt_messages and chosen_text and rejected_text:
        return {
            'messages': prompt_messages,
            'chosen': chosen_text,
            'rejected': rejected_text,
        }
    return None


def convert_dataset(dataset, normalizer, limit: int) -> list[dict]:
    records = []
    for example in dataset:
        normalized = normalizer(example)
        if normalized is None:
            continue
        records.append(normalized)
        if len(records) >= limit:
            break
    if not records:
        raise RuntimeError('No usable records were found after dataset normalization.')
    return records


def save_records(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')


def maybe_download_model(model_id: str, model_path: Path) -> None:
    if (model_path / 'config.json').exists() and is_real_qwen_model(model_path):
        print(f'model already present: {model_path}')
        return
    if model_path.exists():
        print(f'replacing local placeholder model at {model_path}')
        shutil.rmtree(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'downloading model {model_id} -> {model_path}')
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_path),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def main():
    args = parse_args()

    if args.prepare_demo:
        assets = ensure_local_assets()
        print('prepared local smoke-test assets:')
        print(f"- demo model: {assets['model_dir']}")
        print(f"- demo sft: {assets['sft_data']}")
        print(f"- demo dpo: {assets['dpo_data']}")

    if not args.skip_model:
        maybe_download_model(args.model_id, args.model_path)

    if not args.skip_sft:
        print(f'loading SFT dataset from {args.sft_source} [{args.sft_split}]')
        sft_dataset = load_any_dataset(args.sft_source, args.sft_split)
        sft_records = convert_dataset(sft_dataset, normalize_sft_record, args.max_sft_samples)
        save_records(sft_records, args.sft_output)
        print(f'saved {len(sft_records)} SFT records -> {args.sft_output}')
    else:
        print(f'skipped SFT conversion; demo fallback remains at {SFT_DEMO_PATH}')

    if not args.skip_dpo:
        print(f'loading DPO dataset from {args.dpo_source} [{args.dpo_split}]')
        dpo_dataset = load_any_dataset(args.dpo_source, args.dpo_split)
        dpo_records = convert_dataset(dpo_dataset, normalize_dpo_record, args.max_dpo_samples)
        save_records(dpo_records, args.dpo_output)
        print(f'saved {len(dpo_records)} DPO records -> {args.dpo_output}')
    else:
        print(f'skipped DPO conversion; demo fallback remains at {DPO_DEMO_PATH}')

    print('\nnext steps:')
    print('1. run train_sft.py against data/sft_train.json and the real model directory')
    print('2. run train_dpo.py against data/dpo_train.json using the SFT adapter as initialization')
    print('3. keep --allow-bootstrap / --allow-demo-data only for local smoke tests')


if __name__ == '__main__':
    main()
