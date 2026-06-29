from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
MODEL_DIR = ROOT / 'models' / 'Qwen2.5-0.5B'
SFT_DEMO_PATH = DATA_DIR / 'sft_demo.json'
DPO_DEMO_PATH = DATA_DIR / 'dpo_demo.json'
SPECIAL_TOKENS = ['<unk>', '<pad>', '<bos>', '<eos>', '<|im_start|>', '<|im_end|>']
MIN_MODEL_MAX_LENGTH = 1024


def demo_sft_records() -> list[dict]:
    return [
        {
            'messages': [
                {'role': 'system', 'content': '你是 MOSS，一个简洁可靠的中文助手。'},
                {'role': 'user', 'content': '请用两句话介绍深度学习。'},
                {'role': 'assistant', 'content': '深度学习是机器学习的一个分支，使用多层神经网络学习数据表示。它擅长处理图像、语音和文本等复杂任务。'},
            ]
        },
        {
            'messages': [
                {'role': 'system', 'content': '你是 MOSS，一个乐于解释概念的中文助手。'},
                {'role': 'user', 'content': '什么是 LoRA？'},
                {'role': 'assistant', 'content': 'LoRA 是一种参数高效微调方法，它在冻结原模型参数的同时，只训练少量低秩矩阵。这样可以显著减少显存占用和可训练参数量。'},
            ]
        },
        {
            'messages': [
                {'role': 'user', 'content': '请写一句关于春天的短诗。'},
                {'role': 'assistant', 'content': '春风轻拂千山醒，细雨无声润百花。'},
            ]
        },
        {
            'messages': [
                {'role': 'user', 'content': '请解释一下什么是 DPO。'},
                {'role': 'assistant', 'content': 'DPO 是 Direct Preference Optimization，用偏好对数据直接优化策略模型。它不需要单独训练奖励模型，就能利用 chosen 和 rejected 回复进行对齐训练。'},
            ]
        },
    ]


def demo_dpo_records() -> list[dict]:
    return [
        {
            'messages': [
                {'role': 'system', 'content': '你是 MOSS，一个注重准确性的中文助手。'},
                {'role': 'user', 'content': '请解释一下 Transformer 中的 attention。'},
            ],
            'chosen': 'Attention 会让模型在处理当前 token 时，根据相关性聚合上下文中的其他 token 信息，从而更好地建模长距离依赖。',
            'rejected': 'Attention 就是一个很厉害的东西，反正用了效果会更好。',
        },
        {
            'messages': [
                {'role': 'user', 'content': '请写一句鼓励学习机器学习的话。'},
            ],
            'chosen': '持续练习和复盘会让你逐步理解模型背后的原理，不要被一时的困难吓退。',
            'rejected': '学不会也没关系，随便看看就行。',
        },
    ]


def _write_json(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def ensure_demo_datasets() -> tuple[Path, Path]:
    if not SFT_DEMO_PATH.exists():
        _write_json(SFT_DEMO_PATH, demo_sft_records())
    if not DPO_DEMO_PATH.exists():
        _write_json(DPO_DEMO_PATH, demo_dpo_records())
    return SFT_DEMO_PATH, DPO_DEMO_PATH


def _tokenizer_corpus() -> list[str]:
    texts = []
    for record in demo_sft_records():
        for message in record['messages']:
            texts.append(message['role'])
            texts.append(message['content'])
    for record in demo_dpo_records():
        for message in record['messages']:
            texts.append(message['role'])
            texts.append(message['content'])
        texts.append(record['chosen'])
        texts.append(record['rejected'])
    texts.extend(SPECIAL_TOKENS)
    return texts


def _needs_rebuild() -> bool:
    tokenizer_json = MODEL_DIR / 'tokenizer.json'
    config_json = MODEL_DIR / 'config.json'
    if not tokenizer_json.exists() or not config_json.exists():
        return True
    try:
        config = json.loads(config_json.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return True
    return int(config.get('max_position_embeddings', 0)) < MIN_MODEL_MAX_LENGTH


def ensure_local_model() -> Path:
    if _needs_rebuild() and MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer_json = MODEL_DIR / 'tokenizer.json'
    config_json = MODEL_DIR / 'config.json'
    if tokenizer_json.exists() and config_json.exists() and not _needs_rebuild():
        return MODEL_DIR

    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=512,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(_tokenizer_corpus(), trainer=trainer)
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token='<unk>',
        pad_token='<pad>',
        bos_token='<bos>',
        eos_token='<eos>',
        additional_special_tokens=['<|im_start|>', '<|im_end|>'],
    )
    hf_tokenizer.model_max_length = MIN_MODEL_MAX_LENGTH
    hf_tokenizer.padding_side = 'right'
    hf_tokenizer.save_pretrained(str(MODEL_DIR))

    torch.manual_seed(42)
    config = Qwen2Config(
        vocab_size=hf_tokenizer.vocab_size,
        hidden_size=96,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=MIN_MODEL_MAX_LENGTH,
        bos_token_id=hf_tokenizer.bos_token_id,
        eos_token_id=hf_tokenizer.eos_token_id,
        pad_token_id=hf_tokenizer.pad_token_id,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(str(MODEL_DIR))
    return MODEL_DIR


def ensure_local_assets() -> dict[str, Path]:
    sft_path, dpo_path = ensure_demo_datasets()
    model_path = ensure_local_model()
    return {
        'model_dir': model_path,
        'sft_data': sft_path,
        'dpo_data': dpo_path,
    }
