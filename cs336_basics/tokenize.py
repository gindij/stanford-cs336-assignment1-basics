import argparse
import os
import numpy as np
import tqdm

from cs336_basics.bpe import BPETokenizer


def tokenize(arguments: argparse.Namespace):
    tokenizer = BPETokenizer.from_files(
        vocab_path=f"{arguments.tokenizer_dir}/vocab.json",
        merges_path=f"{arguments.tokenizer_dir}/merges.txt",
    )

    if not os.path.exists(arguments.token_output_dir):
        os.makedirs(arguments.token_output_dir)

    lines_per_batch = 100

    for set_name in ["train", "valid"]:
        with open(f"data/{arguments.dataset}_{set_name}.txt", mode="r", encoding="utf-8") as input_file:
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
                if len(chunk) >= arguments.chunk_size:
                    print(f"writing chunk {chunk_idx}")
                    token_array = np.array(chunk[: arguments.chunk_size])
                    np.save(
                        file=os.path.join(
                            arguments.token_output_dir,
                            arguments.dataset + f"_tokens_{set_name}_{str(chunk_idx).zfill(6)}.npy",
                        ),
                        arr=token_array,
                    )
                    tokens_read += len(token_array)
                    chunk = chunk[arguments.chunk_size :]
                    chunk_idx += 1
                    if chunk_idx == getattr(arguments, f"num_{set_name}_chunks"):
                        break
                lines = []


def combine(arguments: argparse.Namespace):

    for set_name in ["train", "valid"]:
        tok_file_names = [fname for fname in os.listdir(arguments.token_output_dir) if set_name in fname]
        tokens = []
        for fname in tqdm.tqdm(tok_file_names):
            tok_file_path = os.path.join(arguments.token_output_dir, fname)
            new_tokens = list(np.load(tok_file_path, mmap_mode="r"))
            tokens.extend(new_tokens)
            os.remove(tok_file_path)
        output_path = os.path.join(arguments.token_output_dir, f"{set_name}.npy")
        tokens = np.array(tokens)
        np.save(output_path, tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="tokenize-and-combine")

    parser.add_argument("--dataset", type=str, default="ts")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer_outputs")
    parser.add_argument("--token-output-dir", type=str, default="tokens")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--num-train-chunks", type=int, default=30)
    parser.add_argument("--num-valid-chunks", type=int, default=5)

    args = parser.parse_args()
    assert args.mode in {"tokenize-and-combine", "combine-only"}
    if args.mode == "tokenize-and-combine":
        tokenize(args)
    combine(args)
