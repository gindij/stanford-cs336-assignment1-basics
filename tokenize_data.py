import argparse
import os
import time

import numpy as np

from cs336_basics.bpe import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--tokenizer-dir", type=str)

    args = parser.parse_args()

    tokenizer = BPETokenizer.from_files(
        vocab_path=os.path.join(args.tokenizer_dir, "vocab.json"),
        merges_path=os.path.join(args.tokenizer_dir, "merges.txt"),
    )
    tokens = []
    with open(args.data_path, "r", encoding="utf-8") as infile:
        start_time = time.time()
        for token in tokenizer.encode_iterable(infile):
            if (len(tokens) + 1) % 10_000 == 0:
                end_time = time.time()
                print(
                    f"read {len(tokens) + 1} tokens ({end_time - start_time:.2f}s)",
                    end="\r",
                )
                start_time = end_time
            tokens.append(token)
    np.save(args.output_path, np.array(tokens))


if __name__ == "__main__":
    main()
