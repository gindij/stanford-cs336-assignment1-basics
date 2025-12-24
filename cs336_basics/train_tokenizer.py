import os
import argparse

from cs336_basics.bpe import BPETokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--vocab-size", type=int, default=32000)
parser.add_argument("--dataset", type=str, default="owt")
parser.add_argument("--tokenizer-output-dir", type=str, default="tokenizer_outputs")
args = parser.parse_args()


tokenizer = BPETokenizer(
    vocab_size=args.vocab_size,
    special_tokens=["<|endoftext|>"],
)

if not os.path.exists(args.tokenizer_output_dir):
    os.makedirs(args.tokenizer_output_dir)

tokenizer.train(f"data/{args.dataset}_train.txt")
tokenizer.save(
    vocab_path=f"{args.tokenizer_output_dir}/vocab.json",
    merges_path=f"{args.tokenizer_output_dir}/merges.txt",
)
