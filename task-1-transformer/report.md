# Task 1 Report

## Summary

This submission implements a Transformer encoder classifier from scratch for Chinese sentiment classification on ChnSentiCorp.

- `scaled_dot_product_attention`: manual implementation with padding mask and causal mask support
- `MultiHeadAttention`: manual QKV projection, head split/merge, output projection
- `TransformerBlock`: pre-LN attention + FFN + residual connections
- `TransformerClassifier`: character-level tokenizer, positional embedding, encoder stack, pooled classifier head

## Training Setup

- Dataset: ChnSentiCorp
- Tokenization: character-level with `[CLS]`, `[PAD]`, `[UNK]`
- Max length: 128
- Model size: `d_model=96`, `num_heads=4`, `num_layers=3`, `ffn_hidden_dim=256`
- Dropout: 0.15
- Batch size: 64
- Optimizer: AdamW
- Learning rate: 3e-4 with warmup + cosine decay

## Result

Official self-check result:

- `attention_correctness`: pass
- `causal_mask`: pass
- `classifier_accuracy`: pass
- `dev accuracy`: `0.845`

See:

- `eval/result.json`
- `ckpt/best.pt`
- `figures/positive.png`
- `figures/negative.png`
- `figures/long_negative.png`

## Observations

The model converged steadily on CPU and reached the best dev accuracy at epoch 7. Padding mask handling was important for stable classification performance. The attention heatmaps show that some heads focus on sentiment-bearing phrases and local context around negative or contrastive expressions.
