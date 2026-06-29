from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be positive")
        self.base_layer = base_layer
        self.r = int(r)
        self.alpha = float(alpha)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scaling = self.alpha / self.r
        weight = base_layer.weight
        self.lora_A = nn.Parameter(
            torch.empty(
                self.r,
                base_layer.in_features,
                device=weight.device,
                dtype=weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base_layer.out_features,
                self.r,
                device=weight.device,
                dtype=weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        for param in self.base_layer.parameters():
            param.requires_grad = False

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        dropped = self.dropout(x).to(self.lora_A.dtype)
        update = (dropped @ self.lora_A.t()) @ self.lora_B.t()
        return base + update.to(base.dtype) * self.scaling

    def merged_linear(self) -> nn.Linear:
        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
            device=self.base_layer.weight.device,
            dtype=self.base_layer.weight.dtype,
        )
        delta = (self.lora_B @ self.lora_A) * self.scaling
        merged.weight.data.copy_(self.base_layer.weight.data + delta.to(self.base_layer.weight.dtype))
        if self.base_layer.bias is not None:
            merged.bias.data.copy_(self.base_layer.bias.data)
        return merged


def _freeze_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _replace_module(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parent = root
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def inject_lora(
    model: nn.Module,
    target_modules: Iterable[str],
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
):
    target_modules = set(target_modules)
    _freeze_model(model)
    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit('.', 1)[-1]
        if leaf not in target_modules:
            continue
        _replace_module(model, name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        replaced.append(name)
    if not replaced:
        raise ValueError(f"No target modules matched: {sorted(target_modules)}")
    model._lora_config = {
        'target_modules': sorted(target_modules),
        'r': int(r),
        'alpha': float(alpha),
        'dropout': float(dropout),
    }
    return model


def _merge_recursive(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, name, child.merged_linear())
        else:
            _merge_recursive(child)


def merge_lora(model: nn.Module):
    _merge_recursive(model)
    return model


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return state


def save_lora_adapter(model: nn.Module, output_dir: str | Path, extra_metadata: dict | None = None) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(getattr(model, '_lora_config', {}))
    if not config:
        raise ValueError('Model does not contain LoRA config')
    payload = {
        'config': config,
        'state_dict': lora_state_dict(model),
        'metadata': extra_metadata or {},
    }
    adapter_path = output_dir / 'adapter.pt'
    torch.save(payload, adapter_path)
    (output_dir / 'adapter_config.json').write_text(
        json.dumps({'config': config, 'metadata': extra_metadata or {}}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return adapter_path


def load_lora_adapter(model: nn.Module, adapter_dir: str | Path):
    adapter_dir = Path(adapter_dir)
    payload = torch.load(adapter_dir / 'adapter.pt', map_location='cpu')
    config = payload['config']
    if not any(isinstance(module, LoRALinear) for module in model.modules()):
        inject_lora(
            model,
            target_modules=config['target_modules'],
            r=config['r'],
            alpha=config['alpha'],
            dropout=config.get('dropout', 0.0),
        )
    state = payload['state_dict']
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_A.data.copy_(state[f"{name}.lora_A"].to(module.lora_A.dtype))
            module.lora_B.data.copy_(state[f"{name}.lora_B"].to(module.lora_B.dtype))
    model._lora_config = config
    return model
