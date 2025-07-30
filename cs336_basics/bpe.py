import base64 as b64
import collections
import json
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import regex as re
import tqdm

GPT2_PRETOK_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def replace_pairs_by_location(
    word: Tuple[bytes, ...], locations: Set[int], merge: Tuple[bytes, bytes]
) -> Tuple[bytes, ...]:
    """
    Replace all occurrences of a pair in a word with a new integer, based on
    the locations where the pair occurs in the word.

    :param word: The tuple of integers representing a word.
    :param locations: The set of locations where the pair occurs in the word.
    :param merge: The pair of bytes replace with the merged pair.
    :return: The new tuple of integers representing the word with the pair
        replaced.
    """
    new_word = []
    i = 0
    while i < len(word):
        if i in locations:
            new_word.append(merge[0] + merge[1])
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    if not all(isinstance(x, bytes) for x in new_word):
        print(word, locations, merge, new_word)
    return tuple(new_word)


class PairIndex:
    """
    A class representing an index of pairs of integers in a BPE tokenizer. This class
    is used to track the and update frequency of pairs of integers in a corpus of words
    as merges are applied (and new token ids are created).
    """

    def __init__(self):

        self._word_counts = collections.defaultdict(int)
        self._pair2count = collections.defaultdict(int)
        self._pair2words = collections.defaultdict(set)
        self._word2pairlocs = collections.defaultdict(lambda: collections.defaultdict(set))

    @classmethod
    def from_file(cls, input_path: str, chunk_size: int, regex_pattern: str) -> "PairIndex":
        """
        Create a PairIndex object from a file of text data. The file is read in chunks
        in case the input is too large to fit in memory.

        :param input_path: The input file path.
        :param chunk_size: The size of the chunks to read from the file.
        :param regex_pattern: The regex pattern to use to find words in the text.
        :return: The PairIndex containing byte pair counts computed from the data in
            the input file.
        """
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
                    words = (tuple(bytes([b]) for b in match.group(0).encode("utf-8")) for match in matches[:-1])
                    pair_index.add_words(words)
                    last_match = matches[-2]
                elif len(matches) == 1:
                    word = tuple(bytes([b]) for b in matches[0].group(0).encode("utf-8"))
                    pair_index.add_word(word, 1)
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

    def add_word(self, word: Tuple[bytes, ...], count: int):
        self._word_counts[word] += count
        for i, pair in enumerate(zip(word, word[1:])):
            self._pair2count[pair] += count
            self._pair2words[pair].add(word)
            self._word2pairlocs[word][pair].add(i)

    def add_words(self, words: Iterable[Tuple[bytes, ...]]):
        """
        Add a list of words to the PairIndex.

        :param words: The words to add
        """
        for word in words:
            self.add_word(word, 1)

    def _remove_word(self, word: Tuple[bytes, ...], count: int):
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
    def pair_counts(self) -> Dict[Tuple[bytes, bytes], int]:
        return self._pair2count

    def apply_merge(self, merge: Tuple[bytes, bytes], new_ix: int):
        """
        Apply a merge operation to the PairIndex. This involves finding
        replacing all words impacted by the merge in the internal state of
        the object.

        :param merge: The pair of integers to merge.
        :param new_ix: The integer to replace the pair with.
        """
        impacted_words = self._pair2words[merge].copy()
        for word in impacted_words:
            count = self._word_counts[word]
            merged_word = replace_pairs_by_location(word, self._word2pairlocs[word][merge], merge)
            self._remove_word(word, count)
            self.add_word(merged_word, count)

    def most_common_pair(self) -> Tuple[Tuple[bytes, bytes], int]:
        max_count = -sys.maxsize
        most_frequent_pair = None
        for pair, count in self.pair_counts.items():

            if most_frequent_pair is None:
                most_frequent_pair = pair
                max_count = count
                continue

            if count > max_count:
                max_count = count
                most_frequent_pair = pair
                continue

            if count == max_count:
                # if the counts are equal, we have to resolve the pieces of
                # the pair separately and compare them for lexical order
                if pair > most_frequent_pair:
                    max_count = count
                    most_frequent_pair = pair

        assert most_frequent_pair is not None
        return (most_frequent_pair, max_count)


class BPETokenizer:

    BASE_VOCAB_SIZE = 256

    def __init__(
        self,
        vocab_size: int,
        special_tokens: Optional[List[str]] = None,
        pretokenization_pattern: str = GPT2_PRETOK_REGEX,
        file_chunk_size: int = 1_000_000,
    ):
        self._file_chunk_size = file_chunk_size
        self._vocab_size = vocab_size
        self._special_tokens = special_tokens or []
        self._pretok_regex = pretokenization_pattern
        self._special_token_regex = (
            "(" + "|".join([re.escape(stok) for stok in sorted(self._special_tokens, key=len, reverse=True)]) + ")"
        )

        self._merges = None
        self._vocab = None
        self._reverse_vocab = None
        self._cache = {}

    @classmethod
    def from_vocab_and_merges(
        cls,
        vocab: Dict[int, bytes],
        merges: Dict[Tuple[bytes, bytes], int],
        special_tokens: Optional[List[str]] = None,
    ) -> "BPETokenizer":
        """
        Create a BPETokenizer object from a vocabulary mapping and a list of merge operations.

        :param vocab: The vocabulary mapping.
        :param merges: The list of merge operations.
        :param special_tokens: The special tokens to consider during encoder
            operations, defaults to None
        :return: A BPETokenizer object.
        """
        tokenizer = cls(vocab_size=len(vocab), special_tokens=special_tokens)
        tokenizer._merges = merges
        tokenizer._vocab = vocab
        tokenizer._reverse_vocab = {v: k for k, v in vocab.items()}
        return tokenizer

    @classmethod
    def from_files(
        cls,
        vocab_path: str,
        merges_path: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "BPETokenizer":
        """
        Create a BPETokenizer object from files containing a vocabulary mapping and a list
        of merge operations. (See from_vocab_and_merges.)

        :param vocab_path: The path to the vocabulary (json) file.
        :param merges_path: The path to the merges (txt) file.
        :param special_tokens: The special tokens to initialize with.
        :return: A tokenizer initialized from sets of merges and a vocabulary
        """
        with open(merges_path, "r", encoding="utf-8") as file:
            i, merges = 0, {}
            for line in file:
                tup = tuple(line.strip().split("\t"))
                merges[((b64.b64decode(tup[0]), b64.b64decode(tup[1])))] = i
                i += 1
        with open(vocab_path, "r", encoding="utf-8") as file:
            vocab = {int(k): b64.b64decode(v) for k, v in json.load(file).items()}
        return BPETokenizer.from_vocab_and_merges(vocab, merges, special_tokens)

    @property
    def vocab(self) -> Dict[int, bytes]:
        if self._vocab is None:
            raise ValueError("tokenizer must be trained to access vocab")
        return self._vocab

    @property
    def merges(self) -> List[Tuple[bytes, bytes]]:
        if self._merges is None:
            raise ValueError("tokenizer must be trained to access merges")
        return self._merges

    def _pretokenize(self, text: str) -> List[bytes]:
        if not any(stok in text for stok in self._special_tokens):
            strs = re.findall(self._pretok_regex, text)
            return [s.encode("utf-8") for s in strs]
        special_parts = re.split(self._special_token_regex, text)
        pretokens = []
        for part in special_parts:
            if part in self._special_tokens:
                pretokens.append(part)
            else:
                pretokens.extend(re.findall(self._pretok_regex, part))
        return [pretok.encode("utf-8") for pretok in pretokens]

    def train(self, input_path: str):
        """
        Train the BPE tokenizer on the data in a text file. To find each subsequent merge,
        we find the current most common pair of bytes in the data and merge them. We continue
        this process until we have reached the desired vocabulary size (or until no
        pair occurs more than once).

        :param input_path: The path with the training data.
        """
        pair_index = PairIndex.from_file(
            input_path,
            chunk_size=self._file_chunk_size,
            regex_pattern=self._pretok_regex,
        )
        num_merges = self._vocab_size - (BPETokenizer.BASE_VOCAB_SIZE + len(self._special_tokens))
        merges = []
        vocab = {i: bytes([i]) for i in range(BPETokenizer.BASE_VOCAB_SIZE)}
        for i in tqdm.tqdm(range(num_merges)):
            most_frequent_pair, most_frequent_count = pair_index.most_common_pair()
            if most_frequent_count == 1:
                break

            assert isinstance(most_frequent_pair[0], bytes) and isinstance(
                most_frequent_pair[1], bytes
            ), f"must be a pair of bytes, got: {most_frequent_pair}"

            merges.append(most_frequent_pair)
            new_idx = BPETokenizer.BASE_VOCAB_SIZE + i
            vocab[new_idx] = most_frequent_pair[0] + most_frequent_pair[1]
            pair_index.apply_merge(most_frequent_pair, new_idx)

        for i, special_token in enumerate(self._special_tokens):
            vocab[i + len(vocab)] = special_token.encode("utf-8")

        self._merges = merges
        self._vocab = vocab
        self._reverse_vocab = {v: k for k, v in vocab.items()}

    def save(self, vocab_path: str, merges_path: str):
        """
        Save the vocabulary and merge operations to files.

        :param vocab_path: The path to save the vocabulary to (json).
        :param merges_path: The path to save the merges to (txt).
        """
        assert self._merges is not None
        assert self._vocab is not None

        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(
                {i: b64.b64encode(bs).decode("ascii") for i, bs in self.vocab.items()},
                file,
            )

        with open(merges_path, "w", encoding="utf-8") as file:
            for merge in self._merges:
                left = b64.b64encode(merge[0]).decode("ascii")
                right = b64.b64encode(merge[1]).decode("ascii")
                file.write(f"{left}\t{right}\n")

    def _has_pair(self, word: Tuple[bytes, ...], pair: Tuple[bytes, bytes]) -> bool:
        for p in zip(word, word[1:]):
            if p == pair:
                return True
        return False

    def _merge_word(
        self,
        word: Tuple[bytes, ...],
        pair_to_merge: Tuple[bytes, bytes],
    ) -> Tuple[bytes, ...]:
        """
        Apply a merge to a given word.
        Example: word=["a", "b", "c"], pair_to_merge=("b", "c") would output
            ["a", "bc"]

        :param word: The word as a tuple of bytes.
        :param pair_to_merge: The pair to merge. For example,
        :return: The word after applying the merge as a tuple of bytes
        """
        if len(word) == 1:
            return word

        pair_found = self._has_pair(word, pair_to_merge)
        if not pair_found:
            return word

        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair_to_merge[0] and word[i + 1] == pair_to_merge[1]:
                new_word.append(pair_to_merge[0] + pair_to_merge[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def _apply_merges(
        self,
        merge_dict: Dict[Tuple[bytes, bytes], int],
        token: bytes,
    ) -> Tuple[bytes, ...]:
        """
        Apply all applicable merges to a given word. For each word, we find the merges
        that match the word's pairs and apply them. We know there are no more merges to
        apply when we can no longer find merges that match pairs in the word.

        :param merge_dict: A dictionary mapping byte pairs to merge indices. (We use the indices
            to indicate the "priority" of a merge. Lower index means more important to apply.)
        :param token: The token to apply the merges to.
        :return: The token bytes strings after all merges have been applied.
        """
        token_bytes = tuple(bytes([b]) for b in token)
        while len(token_bytes) >= 2:
            # merge the pair with the lowest merge index
            pairs = list(zip(token_bytes, token_bytes[1:]))
            pair = min(pairs, key=lambda p: merge_dict.get(p, float("inf")))
            if pair not in merge_dict:
                break  # cant merge anything else
            token_bytes = self._merge_word(token_bytes, pair)
        return token_bytes

    def encode(self, text: str) -> List[int]:
        """
        Return the token ids corresponding to an input string.

        :param text: The input string to compute ids for.
        :return: The list of token ids.
        """
        assert self._merges is not None
        assert self._vocab is not None
        assert self._reverse_vocab is not None

        pretokens = self._pretokenize(text)

        merge_dict = {p: i for i, p in enumerate(self._merges)}
        token_ids = []
        for pretoken in pretokens:
            if pretoken.decode("utf-8") in self._special_tokens:
                token_ids.append(self._reverse_vocab[pretoken])
            else:
                if pretoken in self._cache:
                    merged = self._cache[pretoken]
                else:
                    merged = self._apply_merges(merge_dict, pretoken)
                    self._cache[pretoken] = merged
                token_ids.extend(self._reverse_vocab[byte_string] for byte_string in merged)
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]):
        """
        Encode an iterable of strings, yielding the tokens
        one at a time.

        :param iterable: The iterable of strings to encode.
        :yield: One token id at a time.
        """
        for s in iterable:
            yield from self.encode(s)

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token ids to a string.

        :param ids: The token ids to decode.
        :return: The string corresponding to the input ids
        """
        assert self._vocab is not None
        return b"".join([self._vocab[id_] for id_ in ids]).decode("utf-8")
