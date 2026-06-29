# Task 3: SFT + DPO

This task now supports two paths:

- real-setting training with the actual `Qwen2.5-0.5B` checkpoint and external SFT / DPO data
- local smoke testing with the tiny bootstrap checkpoint already stored in `models/Qwen2.5-0.5B`

The tracked `models/Qwen2.5-0.5B` directory is only for smoke tests and local self-check. Real weights are downloaded into `local_models/Qwen2.5-0.5B` so the repository stays clean.

## Setup

```bash
pip install -r requirements.txt
python data/download.py
```

Default behavior:

- download `Qwen/Qwen2.5-0.5B` into `local_models/Qwen2.5-0.5B/`
- normalize `OpenMOSS-Team/moss-003-sft-data` into `data/sft_train.json`
- normalize `argilla/ultrafeedback-binarized-preferences-cleaned` into `data/dpo_train.json`

If you only want the local smoke-test path:

```bash
python data/download.py --prepare-demo --skip-model --skip-sft --skip-dpo
```

## Training

### SFT

```bash
python train_sft.py \
  --model-path local_models/Qwen2.5-0.5B \
  --data-path data/sft_train.json \
  --device cuda
```

Common options:

- `--batch-size 1 --gradient-accumulation-steps 8`
- `--dtype bfloat16`
- `--lora-r 8 --lora-alpha 16 --lora-dropout 0.05`

### DPO

```bash
python train_dpo.py \
  --model-path local_models/Qwen2.5-0.5B \
  --data-path data/dpo_train.json \
  --sft-adapter ckpt/sft \
  --device cuda
```

## Smoke Test

```bash
python data/download.py --prepare-demo --skip-model --skip-sft --skip-dpo
python train_sft.py --allow-bootstrap --allow-demo-data --model-path models/Qwen2.5-0.5B --data-path data/sft_demo.json --epochs 1 --gradient-accumulation-steps 1
python train_dpo.py --allow-bootstrap --allow-demo-data --model-path models/Qwen2.5-0.5B --data-path data/dpo_demo.json --epochs 1 --gradient-accumulation-steps 1
```

## Data Format

`data/sft_train.json`:

```json
[
  {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain LoRA."},
      {"role": "assistant", "content": "LoRA is a parameter-efficient fine-tuning method."}
    ]
  }
]
```

`data/dpo_train.json`:

```json
[
  {
    "messages": [
      {"role": "user", "content": "Explain Transformer attention."}
    ],
    "chosen": "Attention lets the model aggregate relevant context.",
    "rejected": "Attention is just a useful trick."
  }
]
```

## Compare

```bash
python src/compare.py --model-path local_models/Qwen2.5-0.5B --adapter-dir ckpt/sft --device cuda
```

## Self-Check

```bash
python eval/run.py
```

The official self-check still validates only:

- LoRA trainable parameter ratio
- assistant-only loss masking
- existence of `ckpt/sft`

It does not prove real downstream quality. For submission, you still need to compare base / SFT / DPO outputs manually and document the real experiment setup.
