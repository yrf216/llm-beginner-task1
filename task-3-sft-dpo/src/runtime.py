from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_MODEL_DIR = ROOT / 'models' / 'Qwen2.5-0.5B'
DEFAULT_MODEL_DIR = ROOT / 'local_models' / 'Qwen2.5-0.5B'
DEFAULT_SFT_DATA_PATH = ROOT / 'data' / 'sft_train.json'
DEFAULT_DPO_DATA_PATH = ROOT / 'data' / 'dpo_train.json'

_DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}


def resolve_device(device: str) -> torch.device:
    if device != 'auto':
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype != 'auto':
        return _DTYPE_MAP[dtype]
    if device.type == 'cuda':
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model_config(model_ref: str | Path) -> dict | None:
    model_path = Path(model_ref)
    if not model_path.exists():
        return None
    config_path = model_path / 'config.json'
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text(encoding='utf-8'))


def is_bootstrap_model(model_ref: str | Path) -> bool:
    config = load_model_config(model_ref)
    if not config:
        return False
    return int(config.get('num_hidden_layers', 0)) <= 4 and int(config.get('hidden_size', 0)) <= 256


def is_real_qwen_model(model_ref: str | Path) -> bool:
    config = load_model_config(model_ref)
    if not config:
        return False
    return int(config.get('num_hidden_layers', 0)) >= 20 and int(config.get('hidden_size', 0)) >= 512


def require_real_model(model_ref: str | Path, allow_bootstrap: bool) -> None:
    model_path = Path(model_ref)
    if not model_path.exists():
        return
    if is_bootstrap_model(model_path) and not allow_bootstrap:
        raise RuntimeError(
            'Detected the local bootstrap smoke-test model instead of the real Qwen2.5-0.5B weights. '
            'Download the real checkpoint into local_models/Qwen2.5-0.5B, or rerun with --allow-bootstrap '
            'for a local smoke test.'
        )


def resolve_data_path(requested: Path, demo_path: Path, allow_demo_data: bool) -> Path:
    if requested.exists():
        return requested
    if allow_demo_data:
        return demo_path
    raise FileNotFoundError(
        f'Missing training data at {requested}. Run python data/download.py to prepare real data, '
        'or rerun with --allow-demo-data for the bundled smoke-test dataset.'
    )


def model_load_kwargs(device: torch.device, dtype: torch.dtype) -> dict:
    if device.type == 'cpu':
        return {}
    return {'torch_dtype': dtype}
