# Task 2 Report

## Summary

This submission implements a decoder-only mini-GPT in PyTorch for classical Chinese poetry modeling.

- `BPETokenizer`: byte-level BPE with manual merge training, encode/decode, and JSON save/load
- `CausalSelfAttention`: multi-head causal attention with RoPE and KV cache support
- `MiniGPT`: stacked decoder blocks with tied token embedding / LM head and autoregressive generation
- `train.py`: next-token prediction training loop with AdamW, cosine learning-rate schedule, and gradient clipping

## Training Setup

- Dataset: Tang poetry quick-start corpus (`poetry`)
- Vocabulary size: 384
- Max sequence length: 64
- Model dimension: 128
- Attention heads: 4
- Best checkpoint metric: dev perplexity `47.62`

## Official Self-Check

Results in `eval/result.json`:

- `tokenizer_roundtrip`: pass
- `kv_cache_equivalence`: pass
- `perplexity_on_dev`: pass
- `perplexity`: `47.61` on poetry dev set

## Included Artifacts

- `src/`: tokenizer, RoPE, attention, sampling, and model implementation
- `train.py`: training entrypoint
- `ckpt/tokenizer.json`: trained BPE tokenizer
- `ckpt/best.pt`: best model checkpoint
- `eval/result.json`: official self-check output
