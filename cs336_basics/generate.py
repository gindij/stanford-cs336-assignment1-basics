from typing import Optional
import torch

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


tok = BPETokenizer.from_files(
    vocab_path="ts_tokenizer_new/vocab.json",
    merges_path="ts_tokenizer_new/merges.txt",
    special_tokens=["<|endoftext|>"],
)
prompt = "hello darkness my old friend"
max_tokens_allowed = 1000
model = TransformerLM(
    num_heads=16,
    num_layers=4,
    d_model=512,
    d_ff=2048,
    attn_pdrop=0.2,
    residual_pdrop=0.2,
    vocab_size=10000,
    context_length=256,
)
optimizer = AdamW(model.parameters(), lr=0.001)
load_checkpoint(src="checkpoints/00099.pt", model=model, optimizer=optimizer)

print(
    decode(
        prompt=prompt,
        tokenizer=tok,
        max_tokens_allowed=max_tokens_allowed,
        model=model,
        device="mps",
        p=0.1,
    )
)
