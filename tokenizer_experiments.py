import cProfile
import time
import typing as T
from cs336_basics.bpe import BPETokenizer


def get_documents(filepath: str, n: int) -> T.List[str]:
    chunks = []
    current_chunk = ""
    chunks_found = 0

    with open(filepath, "r", encoding="utf-8") as file:
        while chunks_found < n:
            # Read a small chunk of the file
            data = file.read(8192)
            if not data:  # End of file
                if current_chunk:
                    chunks.append(current_chunk)
                break

            current_chunk += data

            # Look for separators in the current buffer
            while "<|endoftext|>" in current_chunk and chunks_found < n:
                chunk, remainder = current_chunk.split("<|endoftext|>", 1)
                chunks.append(chunk)
                current_chunk = remainder
                chunks_found += 1

    return chunks


def compute_stats(docs: T.List[str], tokenizer: BPETokenizer) -> T.Dict[str, float]:
    num_docs = len(docs)
    bpt = []
    bps = []
    for doc in docs:
        start_time = time.time()
        tokens = list(tokenizer.encode(doc))
        duration = time.time() - start_time
        byts = doc.encode(encoding="utf-8")
        bpt.append(len(byts) / len(tokens))
        bps.append(len(byts) / duration)
    return {
        "bytes_per_sec": sum(bps) / num_docs,
        "bytes_per_tok": sum(bpt) / num_docs,
    }


def run_experiments():
    ts_tokenizer = BPETokenizer.from_files(
        vocab_path="tokenizer_outputs/ts/vocab.json",
        merges_path="tokenizer_outputs/ts/merges.txt",
        special_tokens=["<|endoftext|>"],
    )
    owt_tokenizer = BPETokenizer.from_files(
        vocab_path="tokenizer_outputs/owt/vocab.json",
        merges_path="tokenizer_outputs/owt/merges.txt",
        special_tokens=["<|endoftext|>"],
    )

    num_docs = 100

    start = time.time()
    ts_docs = get_documents(
        filepath="data/TinyStoriesV2-GPT4-valid.txt",
        n=num_docs,
    )
    print(f"Loading TinyStories docs took: {time.time() - start:.2f}s")

    start = time.time()
    owt_docs = get_documents(filepath="data/owt_valid.txt", n=num_docs)
    print(f"Loading OWT docs took: {time.time() - start:.2f}s")

    start = time.time()
    ts_ts_stats = compute_stats(ts_docs, ts_tokenizer)
    print(f"TinyStories tokenizer on TinyStories docs took: {time.time() - start:.2f}s")

    start = time.time()
    owt_owt_stats = compute_stats(owt_docs, owt_tokenizer)
    print(f"OWT tokenizer on OWT docs took: {time.time() - start:.2f}s")

    print("experiment (a)")
    print(
        f"tinystories bytes/token ({num_docs} docs):",
        ts_ts_stats["bytes_per_tok"],
    )
    print(
        f"owt bytes/token ({num_docs} docs):",
        owt_owt_stats["bytes_per_tok"],
    )

    print()

    start = time.time()
    ts_owt_stats = compute_stats(owt_docs, ts_tokenizer)
    print(f"TinyStories tokenizer on OWT docs took: {time.time() - start:.2f}s")

    start = time.time()
    owt_ts_stats = compute_stats(ts_docs, owt_tokenizer)
    print(f"OWT tokenizer on TinyStories docs took: {time.time() - start:.2f}s")

    print("experiment (b)")
    print(
        f"tinystories on owt bytes/token ({num_docs} docs):",
        ts_owt_stats["bytes_per_tok"],
    )
    print(
        f"owt on tinystories bytes/token ({num_docs} docs):",
        owt_ts_stats["bytes_per_tok"],
    )

    print()

    print("experiment (c)")
    owt_bps = owt_owt_stats["bytes_per_sec"]
    print(owt_bps)
    pile_size = 825 * 1_000_000_000
    print(
        "to decode 825GB of text, it would take",
        pile_size / owt_bps,
        "seconds",
    )


if __name__ == "__main__":
    # cProfile.run("run_experiments()")
    run_experiments()
    # ts_tokenizer = BPETokenizer.from_files(
    #     vocab_path="tokenizer_outputs/ts/vocab.json",
    #     merges_path="tokenizer_outputs/ts/merges.txt",
    #     special_tokens=["<|endoftext|>"],
    # )
    # print(list(ts_tokenizer.encode_iterable("there was a little boy <|endoftext|> whose name was jack")))
