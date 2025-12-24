from typing import Optional
import torch

from cs336_basics.bpe import BPETokenizer
from cs336_basics.transformer.functional import softmax
from cs336_basics.transformer.model import TransformerLM


def decode(
    prompt: str,
    tokenizer: BPETokenizer,
    model: TransformerLM,
    max_tokens_allowed: int,
    temp: float = 1.0,
    p: Optional[float] = None,
    eot_token: str = "<|endoftext|>",
    device: str = "cpu",
) -> str:
    eot_token_id = tokenizer.btoi[eot_token.encode("utf-8")]
    context_tokens = torch.Tensor(tokenizer.encode(prompt)).view(1, -1).int()
    tokens_generated = 0
    model.eval()
    with torch.no_grad():
        while context_tokens[-1, -1].item() != eot_token_id and tokens_generated < max_tokens_allowed:
            if context_tokens.shape[1] < model.context_length:
                context = context_tokens.to(device)
            else:
                context = context_tokens[:, -model.context_length :].to(device)
            logits = model(context).squeeze(0)
            dist = softmax(logits[-1, :], dim=-1)
            next_token = torch.multinomial(dist, 1).unsqueeze(0)
            context_tokens = torch.concat([context_tokens, next_token], dim=-1)
            tokens_generated += 1
        return tokenizer.decode(context_tokens[0].tolist())


tok = BPETokenizer.from_files(
    vocab_path="tokenizer_outputs/vocab.json",
    merges_path="tokenizer_outputs/merges.txt",
    special_tokens=["<|endoftext|>"],
)
prompt = "hello darkness my old friend"
max_tokens_allowed = 1000
model = TransformerLM(
    d_model=128,
    num_heads=2,
    d_ff=128,
    attn_pdrop=0.2,
    residual_pdrop=0.2,
    vocab_size=32000,
    context_length=128,
    num_layers=2,
)

print(
    decode(
        prompt=prompt,
        tokenizer=tok,
        max_tokens_allowed=max_tokens_allowed,
        model=model,
    )
)
