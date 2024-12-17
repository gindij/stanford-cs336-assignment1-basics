import base64 as b64
import collections
import json
import regex as re
import sys
from typing import Dict, Generator, Iterable, List, Optional, Set, Tuple

import tqdm


def get_int_pair_counts(
    token_count_dict: Dict[Tuple, int]
) -> Dict[Tuple[int, int], int]:
    bp_counts = collections.defaultdict(int)
    for w, wc in token_count_dict.items():
        for pair in zip(w, w[1:]):
            bp_counts[pair] += wc
    return bp_counts


def apply_merge_to_word(
    word: Tuple[int, ...], pair_to_merge: Tuple[int, int], new_ix: int
) -> Tuple[int]:
    if len(word) == 1:
        return word
    new_word = []
    i = 0
    while i < len(word):
        if (
            i < len(word) - 1
            and word[i] == pair_to_merge[0]
            and word[i + 1] == pair_to_merge[1]
        ):
            new_word.append(new_ix)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def replace_pairs_by_location(
    word: Tuple[int], locations: Set[int], new_ix: int
) -> Tuple[int]:
    new_word = []
    i = 0
    while i < len(word):
        if i in locations:
            new_word.append(new_ix)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


class PairIndex:

    def __init__(self):

        self._word_counts = collections.defaultdict(int)
        self._pair2count = collections.defaultdict(int)
        self._pair2words = collections.defaultdict(set)
        self._word2pairlocs = collections.defaultdict(
            lambda: collections.defaultdict(set)
        )

    @classmethod
    def from_file(
        cls, input_path: str, chunk_size: int, regex_pattern: str
    ) -> "PairIndex":
        pos_start = 0
        pair_index = cls()
        with open(input_path, "r", encoding="utf-8") as file:
            chunks_read = 0
            while True:
                file.seek(pos_start)
                data = file.read(chunk_size)

                if not data:
                    break

                matches = list(re.finditer(regex_pattern, data))

                if len(matches) > 1:
                    pair_index.add_words(
                        [
                            tuple(match.group(0).encode("utf-8"))
                            for match in matches[:-1]
                        ]
                    )
                    last_match = matches[-2]
                elif len(matches) == 1:
                    pair_index.add_words([tuple(matches[0].group(0).encode("utf-8"))])
                    last_match = matches[0]
                else:
                    break
                last_char_position = last_match.span()[1]
                step_size = len(data[:last_char_position].encode(encoding="utf-8"))
                pos_start += step_size
                chunks_read += 1

                print(f"{chunks_read = }", end="\r")
                print()
        return pair_index

    def _add_word(self, word: Tuple[int, ...], count: int):
        self._word_counts[word] += count
        for i, pair in enumerate(zip(word, word[1:])):
            self._pair2count[pair] += count
            self._pair2words[pair].add(word)
            self._word2pairlocs[word][pair].add(i)

    def add_words(self, words: List[Tuple[int, ...]]):
        for word in words:
            self._add_word(word, 1)

    def _remove_word(self, word: Tuple[int, ...], count: int):
        for pair in zip(word, word[1:]):
            if pair in self._pair2count:
                self._pair2count[pair] -= count
                if self._pair2count[pair] == 0:
                    del self._pair2count[pair]
            if pair in self._pair2words and word in self._pair2words[pair]:
                self._pair2words[pair].remove(word)
                if len(self._pair2words[pair]) == 0:
                    del self._pair2words[pair]
        del self._word2pairlocs[word]

    @property
    def pair_counts(self) -> Dict[Tuple[int, int], int]:
        return self._pair2count

    def apply_merge(self, merge: Tuple[int, int], new_ix: int):
        # get the words that had the merge in them
        impacted_words = self._pair2words[merge].copy()
        for word in impacted_words:
            count = self._word_counts[word]
            merged_word = replace_pairs_by_location(
                word, self._word2pairlocs[word][merge], new_ix
            )
            self._remove_word(word, count)
            self._add_word(merged_word, count)


class BPETokenizer:

    def __init__(
        self,
        vocab_size: int,
        special_tokens: Optional[List[str]] = None,
        pretokenization_pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        file_chunk_size: int = 1_000_000,
    ):
        self._file_chunk_size = file_chunk_size
        self._vocab_size = vocab_size
        self._special_tokens = special_tokens or []
        self._pretok_regex = pretokenization_pattern

        self._base_vocab = {i: bytes([i]) for i in range(256)}
        for i, special_token in enumerate(self._special_tokens):
            self._base_vocab[i + 256] = special_token.encode("utf-8")
        self._base_vocab_size = len(self._base_vocab)
        self._reverse_vocab = {v: k for k, v in self._base_vocab.items()}

        self._new_vocab = {}
        self._reverse_new_vocab = {}

        self._merges = None
        self._vocab = None
        self._reverse_vocab = None

    @classmethod
    def from_vocab_and_merges(
        cls,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ) -> "BPETokenizer":
        tokenizer = cls(vocab_size=len(vocab), special_tokens=special_tokens)
        tokenizer._merges = merges
        tokenizer._vocab = vocab
        tokenizer._reverse_vocab = {v: k for k, v in vocab.items()}
        return tokenizer

    def _int2bytes(self, ix: int) -> bytes:
        if ix in self._base_vocab:
            return self._base_vocab[ix]
        ix_left, ix_right = self._new_vocab[ix]
        return self._int2bytes(ix_left) + self._int2bytes(ix_right)

    def _pair2bytes(self, pair: Tuple[int, int]) -> bytes:
        return self._int2bytes(pair[0]) + self._int2bytes(pair[1])

    def _assemble_byte_vocab(self, merges: List[Tuple[int, int]]) -> Dict[int, bytes]:
        assert self._merges is not None
        vocab = self._base_vocab.copy()
        for pair in tqdm.tqdm(merges):
            ix = self._reverse_new_vocab[pair]
            byts = self._pair2bytes(pair)
            vocab[ix] = byts
        return vocab

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str) -> "BPETokenizer":
        with open(vocab_path, "r", encoding="utf-8") as file:
            vocab = {int(k): b64.b64decode(v) for k, v in json.load(file).items()}
        with open(merges_path, "r", encoding="utf-8") as file:
            merges = []
            for line in file:
                tup = tuple(line.strip().split("\t"))
                merges.append((b64.b64decode(tup[0]), b64.b64decode(tup[1])))
        return BPETokenizer.from_vocab_and_merges(vocab, merges)

    @property
    def vocab(self) -> Dict[int, bytes]:
        if self._vocab is None:
            raise ValueError("tokenizer must be trained to access vocab")
        return self._vocab

    def _assemble_byte_merges(
        self, merges: List[Tuple[int, int]]
    ) -> List[Tuple[bytes, bytes]]:
        byte_merges = []
        for p1, p2 in tqdm.tqdm(merges):
            byte_merges.append((self._int2bytes(p1), self._int2bytes(p2)))
        return byte_merges

    @property
    def merges(self) -> List[Tuple[bytes, bytes]]:
        if self._merges is None:
            raise ValueError("tokenizer must be trained to access merges")
        return self._merges

    def _pretokenize(self, text: str) -> List[str]:
        if not any(stok in text for stok in self._special_tokens):
            return re.findall(self._pretok_regex, text)
        special_token_regex = (
            "("
            + "|".join(
                [
                    re.escape(stok)
                    for stok in sorted(self._special_tokens, key=len, reverse=True)
                ]
            )
            + ")"
        )
        special_parts = re.split(special_token_regex, text)
        pretokens = []
        for part in special_parts:
            if part in self._special_tokens:
                pretokens.append(part)
            else:
                pretokens.extend(re.findall(self._pretok_regex, part))
        return pretokens

    def _get_pretoken_counts(self, words: List[str]) -> Dict[Tuple, int]:
        counts = collections.defaultdict(int)
        for word in tqdm.tqdm(words):
            byte_tuple = tuple(bytes(word, encoding="utf-8"))  # Tuple[int]
            counts[byte_tuple] += 1
        return counts

    def _most_common_pair(
        self, counts: Dict[Tuple[int, int], int]
    ) -> Tuple[Tuple[int, int], int]:
        max_count = -sys.maxsize
        most_frequent_pair = (-1, -1)
        for pair, count in counts.items():

            if count > max_count:
                max_count = count
                most_frequent_pair = pair
                continue

            if count == max_count:
                # if the counts are equal, we have to resolve the pieces of the pair
                # separately and compare them for lexical order
                ip_pair = (self._int2bytes(pair[0]), self._int2bytes(pair[1]))
                max_ip_pair = (
                    self._int2bytes(most_frequent_pair[0]),
                    self._int2bytes(most_frequent_pair[1]),
                )
                if ip_pair > max_ip_pair:
                    max_count = count
                    most_frequent_pair = pair

        return (most_frequent_pair, max_count)

    def train(self, input_path: str):
        pair_index = PairIndex.from_file(
            input_path,
            chunk_size=self._file_chunk_size,
            regex_pattern=self._pretok_regex,
        )
        num_merges = self._vocab_size - self._base_vocab_size
        merges_int = []
        for i in tqdm.tqdm(range(num_merges)):
            most_frequent_pair, most_frequent_count = self._most_common_pair(
                pair_index.pair_counts
            )

            if most_frequent_count == 1:
                break

            merges_int.append(most_frequent_pair)

            new_idx = self._base_vocab_size + i
            self._new_vocab[new_idx] = most_frequent_pair
            self._reverse_new_vocab[most_frequent_pair] = new_idx

            pair_index.apply_merge(most_frequent_pair, new_idx)

        self._merges = self._assemble_byte_merges(merges_int)
        self._vocab = self._assemble_byte_vocab(merges_int)
        self._reverse_vocab = {v: k for k, v in self._vocab.items()}

    def save(self, vocab_path: str, merges_path: str):
        assert self._merges is not None
        assert self._vocab is not None
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(
                {i: b64.b64encode(bs).decode("ascii") for i, bs in self._vocab.items()},
                file,
            )
        with open(merges_path, "w", encoding="utf-8") as file:
            for merge in self._merges:
                left = b64.b64encode(merge[0]).decode("ascii")
                right = b64.b64encode(merge[1]).decode("ascii")
                file.write(f"{left}\t{right}\n")

    def encode(self, text: str) -> List[int]:
        assert self._merges is not None
        assert self._reverse_vocab is not None
        str_pretokens = self._pretokenize(text)
        pretokens = []
        for i, pretoken in enumerate(str_pretokens):
            if pretoken in self._special_tokens:
                idxs = [self._reverse_vocab[bytes(pretoken, "utf-8")]]
            else:
                byte_list = [bytes([b]) for b in bytes(pretoken, "utf-8")]
                idxs = [self._reverse_vocab[byt] for byt in byte_list]
            pretokens.append(idxs)
        int_merges = [
            (
                self._reverse_vocab[merge[0]],
                self._reverse_vocab[merge[1]],
            )
            for merge in self._merges
        ]
        for i, pretoken in enumerate(pretokens):
            merged_token = pretoken
            if len(merged_token) > 1:
                for j, merge in enumerate(int_merges):
                    merged_token = apply_merge_to_word(merged_token, merge, 256 + j)
            pretokens[i] = merged_token
        ids = []
        for merged_pretoken in pretokens:
            ids.extend(merged_pretoken)
        return ids

    def encode_iterable(self, iterable: Iterable[str]):
        for s in iterable:
            ids = self.encode(s)
            yield from ids

    def decode(self, ids: List[int]) -> str:
        assert self._vocab is not None
        print(ids, [self._vocab[id_] for id_ in ids])
        return b"".join([self._vocab[id_] for id_ in ids]).decode("utf-8")
