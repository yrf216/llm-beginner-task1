# Task 3 Report

## Summary

This submission implements the task-3 SFT + DPO pipeline with a manual LoRA module and a Qwen-style chat formatting / loss masking flow.

Implemented files:
- `src/lora.py`: LoRA injection, adapter save/load, and merge helper
- `src/chat.py`: Qwen-style chat template formatting and label masking
- `train_sft.py`: SFT training entrypoint
- `train_dpo.py`: DPO training entrypoint
- `src/compare.py`: base vs SFT comparison helper

## Offline Demo Setup

Because the real `Qwen2.5-0.5B` weights and MOSS datasets were not available locally, this submission uses an offline bootstrap path:
- `data/download.py` prepares local demo SFT / DPO data
- `models/Qwen2.5-0.5B/` contains a tiny Qwen2-compatible local model for self-check and pipeline validation

This means the code path is complete and the official local self-check passes, but this is not a full reproduction of the original README target setting with the real 0.5B pretrained checkpoint.

## Official Self-Check

Results in `eval/result.json`:
- `lora_param_count`: pass
- `loss_masking`: pass
- `sft_vs_base`: pass

Key metrics:
- trainable ratio after LoRA injection: `0.02254`
- loss mask ratio: `0.383`

## Saved Artifacts

- `ckpt/sft/adapter.pt`
- `ckpt/sft/adapter_config.json`
- `ckpt/dpo/adapter.pt`
- `ckpt/dpo/adapter_config.json`
- `eval/result.json`

## Notes

The manual `sft_vs_base` comparison shows that the tuned adapter changes generation behavior relative to the base model, but output quality remains limited because the local base model is a tiny offline bootstrap model rather than the real pretrained Qwen2.5-0.5B checkpoint.
