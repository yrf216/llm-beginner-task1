# Task 3 Report

## Summary

Task 3 now supports both:

- a real-setting path for `Qwen2.5-0.5B` plus external SFT / DPO datasets
- a local smoke-test fallback for the bundled bootstrap model and demo data

The implementation remains manual:

- `src/lora.py`: LoRA injection, adapter save/load, merge helper
- `src/chat.py`: Qwen-style chat formatting and assistant-only loss masking
- `train_sft.py`: SFT entrypoint with real-model-first defaults
- `train_dpo.py`: DPO entrypoint with real-model-first defaults
- `data/download.py`: model download plus dataset normalization into local JSON training files
- `src/compare.py`: base vs tuned generation comparison helper
- `src/runtime.py`: split between tracked bootstrap assets and untracked real-model assets

## Real-Setting Upgrade

Compared with the earlier offline-only version, task 3 now:

- defaults to `local_models/Qwen2.5-0.5B/` plus `data/sft_train.json` / `data/dpo_train.json`
- keeps `models/Qwen2.5-0.5B/` as the tracked bootstrap checkpoint for smoke tests and official self-check
- refuses to silently train on the bootstrap checkpoint unless `--allow-bootstrap` is passed
- refuses to silently fall back to demo data unless `--allow-demo-data` is passed
- supports device and dtype selection for practical GPU runs
- converts common Hugging Face dataset schemas into the local SFT / DPO JSON formats expected by the training scripts

## Verified State

Verified locally:

- official self-check still passes
- upgraded scripts compile
- smoke-test fallback path still works
- real dataset normalization works
- real `Qwen2.5-0.5B` download path works
- real-model forward path exposed and fixed a LoRA dtype mismatch bug

Not fully verified here:

- end-to-end real SFT or DPO completion on the real model

Reason:

- this environment has no GPU, and CPU training for `Qwen2.5-0.5B` is too slow to complete in a practical time budget

## Local Self-Check

Results in `eval/result.json`:

- `lora_param_count`: pass
- `loss_masking`: pass
- `sft_vs_base`: pass

Key smoke-test metrics:

- trainable ratio after LoRA injection: `0.02254`
- loss mask ratio: `0.383`

## Notes

At this point the remaining blocker is compute, not missing code. On a GPU-equipped machine, the next step is to run `train_sft.py`, then `train_dpo.py`, then attach qualitative comparisons from `src/compare.py`.
