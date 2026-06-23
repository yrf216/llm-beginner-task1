from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib import rcParams

from src.model import load_for_eval
from src.tokenizer import SimpleCharTokenizer


rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("ckpt") / "best.pt")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenize_fn = load_for_eval(str(args.ckpt))
    payload = torch.load(args.ckpt, map_location="cpu")
    config = model.config
    tokenizer = SimpleCharTokenizer(payload["vocab"])

    input_ids = tokenize_fn(args.text).unsqueeze(0)
    with torch.no_grad():
        logits, attentions = model(input_ids, return_attention=True)

    pred = int(logits.argmax(dim=-1).item())
    probs = torch.softmax(logits, dim=-1)[0]
    tokens = tokenizer.tokens(args.text, config.max_length)
    attn = attentions[args.layer][0, args.head, : len(tokens), : len(tokens)].cpu().numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn, cmap="viridis")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)
    ax.set_title(
        f"layer={args.layer} head={args.head} pred={pred} "
        f"p_neg={probs[0]:.3f} p_pos={probs[1]:.3f}"
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
