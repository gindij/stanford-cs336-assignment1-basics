from typing import Optional
import argparse
import os
import torch
import json

from cs336_basics.bpe import BPETokenizer
from cs336_basics.optimizer.adamw import AdamW
from cs336_basics.transformer.functional import softmax
from cs336_basics.transformer.model import TransformerLM
from cs336_basics.utils import load_checkpoint


def decode(
    prompt: str,
    tokenizer: BPETokenizer,
    model: TransformerLM,
    max_tokens_allowed: int,
    temp: float = 2.0,
    p: Optional[float] = None,
    eot_token: str = "<|endoftext|>",
    device: str = "cpu",
) -> str:
    eot_token_id = tokenizer.btoi[eot_token.encode("utf-8")]
    context_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).view(1, -1)
    tokens_generated = 0
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        while context_tokens[-1, -1].item() != eot_token_id and tokens_generated < max_tokens_allowed:
            if context_tokens.shape[1] < model.context_length:
                context = context_tokens.to(device)
            else:
                context = context_tokens[:, -model.context_length :].to(device)
            logits = model(context).squeeze(0)
            next_token_logits = logits[-1, :] / temp

            if p is not None and p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > p

                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float("-inf")

            dist = softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(dist, 1).unsqueeze(0)
            context_tokens = torch.concat([context_tokens, next_token], dim=-1)
            tokens_generated += 1
        return tokenizer.decode(context_tokens[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from a model checkpoint.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to start generation.")
    parser.add_argument("--max-tokens-allowed", type=int, default=256, help="Maximum number of tokens to generate.")
    parser.add_argument("--temp", type=float, default=2.0, help="Temperature for sampling.")
    parser.add_argument("--p", type=float, default=None, help="Top-p sampling probability.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu, cuda, mps).")
    parser.add_argument(
        "--tokenizer-dir", type=str, default="ts_tokenizer_new", help="Directory containing tokenizer files."
    )

    args = parser.parse_args()

    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    tok = BPETokenizer.from_files(
        vocab_path=os.path.join(args.tokenizer_dir, "vocab.json"),
        merges_path=os.path.join(args.tokenizer_dir, "merges.txt"),
        special_tokens=["<|endoftext|>"],
    )
    model = TransformerLM(
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        attn_pdrop=config["attn_pdrop"],
        residual_pdrop=config["residual_pdrop"],
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
    )
    optimizer = AdamW(model.parameters(), lr=0.001)
    load_checkpoint(src=args.checkpoint_path, model=model, optimizer=optimizer, map_location=torch.device(args.device))

    print(
        decode(
            prompt=args.prompt,
            tokenizer=tok,
            max_tokens_allowed=args.max_tokens_allowed,
            model=model,
            device=args.device,
            temp=args.temp,
            p=args.p,
        )
    )
