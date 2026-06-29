# Base vs SFT Comparison

This file should be regenerated after running task 3 on the real `Qwen2.5-0.5B` checkpoint in `local_models/Qwen2.5-0.5B`.

Recommended command:

```bash
python src/compare.py --model-path local_models/Qwen2.5-0.5B --adapter-dir ckpt/sft --device cuda
```

Suggested prompts:

- `什么是 LoRA？`
- `请用两句话介绍深度学习。`
- `解释一下 DPO 和 SFT 的区别。`

Record:

- base output
- SFT output
- if available, DPO-tuned output
- a short note on whether the tuned model is more helpful, more specific, or more aligned with the instruction

The old bootstrap-model comparison is no longer the main result. For submission, the useful comparison is the one produced from the real checkpoint and real datasets.
