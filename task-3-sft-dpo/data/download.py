from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap import ensure_local_assets


def main():
    assets = ensure_local_assets()
    print('已准备 task-3 离线 demo 资源：')
    print(f"- model: {assets['model_dir']}")
    print(f"- sft data: {assets['sft_data']}")
    print(f"- dpo data: {assets['dpo_data']}")
    print('如需替换为真实 Qwen2.5-0.5B 和 MOSS 数据，可直接覆盖上述目录后继续运行 train_sft.py / train_dpo.py。')


if __name__ == '__main__':
    main()
