import argparse
import os
import numpy as np

from cs336_basics.bpe import BPETokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="owt")
parser.add_argument("--tokenizer-dir", type=str, default="tokenizer_outputs")
parser.add_argument("--token-output-dir", type=str, default="tokens")
parser.add_argument("--chunk-size", type=int, default=1000000)
parser.add_argument("--num-chunks", type=int, default=100)
args = parser.parse_args()


tokenizer = BPETokenizer.from_files(
    vocab_path=f"{args.tokenizer_dir}/vocab.json",
    merges_path=f"{args.tokenizer_dir}/merges.txt",
)

if not os.path.exists(args.token_output_dir):
    os.makedirs(args.token_output_dir)

lines_per_batch = 100

with open(f"data/{args.dataset}_train.txt", mode="r", encoding="utf-8") as input_file:
    lines = []
    chunk = []
    chunk_idx = 0
    tokens_read = 0
    for line in input_file:
        if len(lines) <= lines_per_batch:
            lines.append(line.strip())
            continue
        tokens = list(tokenizer.encode_iterable("\n".join(lines)))
        chunk.extend(tokens)
        print(f"chunk size: {len(chunk)}")
        if len(chunk) >= args.chunk_size:
            print(f"writing chunk {chunk_idx}")
            token_array = np.array(tokens[: args.chunk_size])
            np.save(
                file=os.path.join(
                    args.token_output_dir,
                    args.dataset + f"_tokens_{str(chunk_idx).zfill(6)}.npy",
                ),
                arr=token_array,
            )
            tokens_read += len(token_array)
            chunk = tokens[args.chunk_size :]
            chunk_idx += 1
            if chunk_idx == args.num_chunks:
                break
        lines = []
