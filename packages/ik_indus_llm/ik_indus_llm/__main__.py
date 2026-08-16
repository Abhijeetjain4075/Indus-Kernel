"""CLI: generate text from a trained checkpoint.

Usage:
    python -m llm_from_scratch.generate --prompt "Once upon a time" --max_new_tokens 500
"""

import argparse

from .generate import generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--checkpoint", type=str, default="checkpoints/ckpt_final.pt")
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=200)
    args = p.parse_args()

    text = generate(
        prompt=args.prompt,
        checkpoint=args.checkpoint,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(text)


if __name__ == "__main__":
    main()
