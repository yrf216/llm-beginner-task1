# Base vs SFT Comparison

Prompt 1:
`什么是 LoRA？`

Base output:
A mostly invalid / unreadable response with replacement characters.

SFT output:
A different but still low-quality response dominated by repeated punctuation.

Prompt 2:
`请用两句话介绍深度学习。`

Base output:
Again mostly unreadable bootstrap-model output.

SFT output:
A visibly different response pattern, but still not semantically useful.

Interpretation:
- The tuned adapter changes the model output distribution, which is enough for the task-3 local self-check path.
- Output quality is still weak because this repository uses a tiny offline Qwen2-compatible bootstrap model instead of the real `Qwen2.5-0.5B` pretrained checkpoint.
- Replacing `models/Qwen2.5-0.5B/` with the real pretrained model and swapping in real MOSS SFT / DPO data is the next step if you want a submission suitable for stronger human evaluation.
