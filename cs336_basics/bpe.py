import base64 as b64
import collections
import json
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import regex as re
import tqdm

GPT2_PRETOK_REGEX = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def apply_merge_to_word(
    word: Tuple[int, ...], pair_to_merge: Tuple[int, int], new_ix: int
) -> Tuple[int, ...]:
    """
    Apply a merge operation to a word. Here, a word is a tuple of integers, where each
    The merge operation is applied to each occurrence of the pair in the word.

    :param word: The tuple of integers representing the word.
    :param pair_to_merge: The pair of integers to merge.
    :param new_ix: The integer to replace the pair with (if found).
    :return: A new tuple of integers representing the word with the merge applied.
    """
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
    """
    Replace all occurrences of a pair in a word with a new integer, based on the locations
    where the pair occurs in the word.

    :param word: The tuple of integers representing a word.
    :param locations: The set of locations where the pair occurs in the word.
    :param new_ix: The integer to replace the pair with (if found).
    :return: The new tuple of integers representing the word with the pair replaced.
    """
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


class Merge:
    """
    A class representing a merge operation in a BPE tokenizer. The merge operation is
    represented by a pair of integers or bytes.
    """

    def __init__(self, left: Union[int, bytes], right: Union[int, bytes]):
        tleft = type(left)
        tright = type(right)
        assert tleft == tright
        assert tleft in {int, bytes}
        self._left = left
        self._right = right

    @classmethod
    def from_ints(cls, left: int, right: int) -> "Merge":
        """
        Create a Merge object from a pair of integers.

        :param left: The left integer in the pair.
        :param right: The right integer in the pair.
        :return: A Merge object representing the pair of integers.
        """
        return cls(left, right)

    @classmethod
    def from_bytes(cls, left: bytes, right: bytes) -> "Merge":
        """
        Create a Merge object from a pair of bytes.

        :param left: The left byte in the pair.
        :param right: The right byte in the pair.
        :return: A Merge object representing the pair of bytes.
        """
        return cls(left, right)

    def as_int_tuple(
        self, reverse_vocab: Optional[Dict[bytes, int]] = None
    ) -> Tuple[int, int]:
        """
        Get the pair of integers represented by the Merge object.

        :param reverse_vocab: A reverse vocabulary to use if the merge was instantiated
            with bytes, defaults to None
        :return: A tuple of integers representing the merge.
        """
        if isinstance(self._left, int) and isinstance(self._right, int):
            return (self._left, self._right)
        assert reverse_vocab is not None
        return (
            reverse_vocab[self._left],
            reverse_vocab[self._right],
        )

    def as_bytes_tuple(
        self, vocab: Optional[Dict[int, bytes]] = None
    ) -> Tuple[bytes, bytes]:
        """
        Get the pair of bytes represented by the Merge object.

        :param vocab: A forward vocabulary mapping to use if the merge was instantiated
            with ints, defaults to None
        :return: A tuple of bytes representing the merge.
        """
        if isinstance(self._left, bytes) and isinstance(self._right, bytes):
            return (self._left, self._right)
        assert vocab is not None
        return (vocab[self._left], vocab[self._right])


def int2bytes(
    ix: int, base: Dict[int, bytes], new: Dict[int, Tuple[int, int]]
) -> bytes:
    """
    Recursively convert an integer to a sequence of bytes based on a
    base vocabulary and a new vocabulary.

    :param ix: The index to convert.
    :param base: The base vocabulary mapping. This is used for the
        base case for the conversion.
    :param new: The new vocabulary mapping. This is used for the
        inductive case.
    :return: A sequence of bytes representing the integer.
    """
    if ix in base:
        return base[ix]
    ix_left, ix_right = new[ix]
    return int2bytes(ix_left, base, new) + int2bytes(ix_right, base, new)


def pair2bytes(
    pair: Tuple[int, int],
    base: Dict[int, bytes],
    new: Dict[int, Tuple[int, int]],
) -> bytes:
    """
    Convert a pair of integers to a sequence of bytes based on a base and
    new vocabulary.

    :param pair: The pair to convert.
    :param base: The base vocabulary mapping. See int2bytes.
    :param new: The new vocabulary mapping. See int2bytes.
    :return: The sequence of bytes representing the pair.
    """
    return int2bytes(pair[0], base, new) + int2bytes(pair[1], base, new)


def ipair2bpair(
    pair: Tuple[int, int],
    base_vocab: Dict[int, bytes],
    new_vocab: Dict[int, Tuple[int, int]],
) -> Tuple[bytes, bytes]:
    return (
        int2bytes(pair[0], base_vocab, new_vocab),
        int2bytes(pair[1], base_vocab, new_vocab),
    )


class Vocab:
    """
    A class representing a bidirectional vocabulary mapping in a BPE tokenizer.
    """

    def __init__(self, vocab: Dict[int, bytes]):
        self._forward_mapping = vocab
        self._inverse_mapping = {v: k for k, v in vocab.items()}

    def __getitem__(self, key: Union[int, bytes]):
        if isinstance(key, bytes):
            return self._inverse_mapping[key]
        return self._forward_mapping[key]

    def __len__(self):
        return len(self._forward_mapping)

    @classmethod
    def from_base_and_new(
        cls,
        base_vocab: Dict[int, bytes],
        new_vocab: Dict[int, Tuple[int, int]],
        merges: List[Merge],
    ) -> "Vocab":
        """
        Create a Vocab object from a base vocabulary, a new vocabulary, and a list of
        merge operations.

        :param base_vocab: The base vocabulary mapping. This maps integers directly to bytes.
        :param new_vocab: The new vocabulary mapping. This maps integers to pairs of integers
            that were merged to create the new integer.
        :param merges: The merges that were applied to create the new vocabulary from the base
            vocabulary.
        :return: A Vocab object representing the new vocabulary.
        """
        reverse_new_vocab = {v: k for k, v in new_vocab.items()}
        vocab = base_vocab.copy()
        for merge in tqdm.tqdm(merges):
            pair = merge.as_int_tuple()
            ix = reverse_new_vocab[pair]
            byts = pair2bytes(pair, base_vocab, new_vocab)
            vocab[ix] = byts
        return cls(vocab=vocab)

    def add_mapping(self, ix: int, word: str):
        self._forward_mapping[ix] = word.encode("utf-8")
        self._inverse_mapping[word.encode("utf-8")] = ix

    def convert_strings_to_int_tuples(
        self, strs: List[str], special_tokens: List[str]
    ) -> List[Tuple[int, ...]]:
        """
        Convert a list of strings to a list of tuples of integers based on the vocabulary
        and special tokens.

        :param strs: The strings to convert.
        :param special_tokens: The special tokens to handle in the conversion. (These
            are considered as single tokens.)
        :return: A list of tuples of integers representing the strings.
        """
        int_tups = []
        for s in strs:
            if s in special_tokens:
                idxs = [self._inverse_mapping[bytes(s, "utf-8")]]
            else:
                byte_list = [bytes([b]) for b in bytes(s, "utf-8")]
                idxs = [self._inverse_mapping[byt] for byt in byte_list]
            int_tups.append(idxs)
        return int_tups

    def save(self, path: str):
        """
        Save a vocabulary mapping to a json file.

        :param path: The path to save the vocabulary mapping to.
        """
        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    i: b64.b64encode(bs).decode("ascii")
                    for i, bs in self._forward_mapping.items()
                },
                file,
            )

    @property
    def forward_mapping(self) -> Dict[int, bytes]:
        return self._forward_mapping

    @property
    def inverse_mapping(self) -> Dict[bytes, int]:
        return self._inverse_mapping


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
        self._word2pairlocs = collections.defaultdict(
            lambda: collections.defaultdict(set)
        )

    @classmethod
    def from_file(
        cls, input_path: str, chunk_size: int, regex_pattern: str
    ) -> "PairIndex":
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
        """
        Add a list of words to the PairIndex.

        :param words: The words to add
        """
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
            merged_word = replace_pairs_by_location(
                word, self._word2pairlocs[word][merge], new_ix
            )
            self._remove_word(word, count)
            self._add_word(merged_word, count)


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
            "("
            + "|".join(
                [
                    re.escape(stok)
                    for stok in sorted(self._special_tokens, key=len, reverse=True)
                ]
            )
            + ")"
        )

        self._merges = None
        self._vocab = None

    @classmethod
    def from_vocab_and_merges(
        cls,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
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
        tokenizer._merges = [Merge.from_bytes(*byte_merge) for byte_merge in merges]
        tokenizer._vocab = Vocab(vocab)
        return tokenizer

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str) -> "BPETokenizer":
        """
        Create a BPETokenizer object from files containing a vocabulary mapping and a list
        of merge operations. (See from_vocab_and_merges.)

        :param vocab_path: The path to the vocabulary (json) file.
        :param merges_path: The path to the merges (txt) file.
        :return: _description_
        """
        with open(merges_path, "r", encoding="utf-8") as file:
            merges = []
            for line in file:
                tup = tuple(line.strip().split("\t"))
                merges.append(
                    Merge.from_bytes(b64.b64decode(tup[0]), b64.b64decode(tup[1]))
                )
        with open(vocab_path, "r", encoding="utf-8") as file:
            vocab = {int(k): b64.b64decode(v) for k, v in json.load(file).items()}
        return BPETokenizer.from_vocab_and_merges(vocab, merges)

    @property
    def vocab(self) -> Dict[int, bytes]:
        if self._vocab is None:
            raise ValueError("tokenizer must be trained to access vocab")
        return self._vocab.forward_mapping

    @property
    def merges(self) -> List[Merge]:
        if self._merges is None:
            raise ValueError("tokenizer must be trained to access merges")
        return [merge.as_bytes_tuple(self.vocab) for merge in self._merges]

    def _pretokenize(self, text: str) -> List[str]:
        if not any(stok in text for stok in self._special_tokens):
            return re.findall(self._pretok_regex, text)
        special_parts = re.split(self._special_token_regex, text)
        pretokens = []
        for part in special_parts:
            if part in self._special_tokens:
                pretokens.append(part)
            else:
                pretokens.extend(re.findall(self._pretok_regex, part))
        return pretokens

    def _most_common_pair(
        self,
        counts: Dict[Tuple[int, int], int],
        base_vocab: Dict[int, bytes],
        new_vocab: Dict[int, Tuple[int, int]],
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
                ip_pair = ipair2bpair(pair, base_vocab, new_vocab)
                max_ip_pair = ipair2bpair(most_frequent_pair, base_vocab, new_vocab)
                if ip_pair > max_ip_pair:
                    max_count = count
                    most_frequent_pair = pair

        return (most_frequent_pair, max_count)

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
        num_merges = self._vocab_size - (
            BPETokenizer.BASE_VOCAB_SIZE + len(self._special_tokens)
        )
        merges = []
        base_vocab = {i: bytes([i]) for i in range(BPETokenizer.BASE_VOCAB_SIZE)}
        new_vocab = {}
        for i in tqdm.tqdm(range(num_merges)):
            most_frequent_pair, most_frequent_count = self._most_common_pair(
                pair_index.pair_counts, base_vocab, new_vocab
            )

            if most_frequent_count == 1:
                break

            merges.append(Merge.from_ints(*most_frequent_pair))

            new_idx = BPETokenizer.BASE_VOCAB_SIZE + i
            new_vocab[new_idx] = most_frequent_pair

            pair_index.apply_merge(most_frequent_pair, new_idx)

        self._merges = merges
        self._vocab = Vocab.from_base_and_new(base_vocab, new_vocab, merges)

        for i, special_token in enumerate(self._special_tokens):
            self._vocab.add_mapping(i + len(self._vocab), special_token)

    def save(self, vocab_path: str, merges_path: str):
        """
        Save the vocabulary and merge operations to files.

        :param vocab_path: The path to save the vocabulary to (json).
        :param merges_path: The path to save the merges to (txt).
        """
        assert self._merges is not None
        assert self._vocab is not None

        self._vocab.save(vocab_path)

        with open(merges_path, "w", encoding="utf-8") as file:
            for merge in self._merges:
                merge_byte_pair = merge.as_byte_tuple()
                left = b64.b64encode(merge_byte_pair.left).decode("ascii")
                right = b64.b64encode(merge_byte_pair.right).decode("ascii")
                file.write(f"{left}\t{right}\n")

    def _apply_all_merges(self, merges, token):
        merged_token = token
        for j, merge in enumerate(merges):
            merged_token = apply_merge_to_word(merged_token, merge, 256 + j)
        return merged_token

    def encode(self, text: str) -> List[int]:
        """
        Return the token ids corresponding to an input string.

        :param text: The input string to compute ids for.
        :return: The list of token ids.
        """
        assert self._merges is not None
        assert self._vocab is not None
        pretokens = self._vocab.convert_strings_to_int_tuples(
            self._pretokenize(text), self._special_tokens
        )
        int_merges = [
            merge.as_int_tuple(self._vocab.inverse_mapping) for merge in self._merges
        ]
        for i, pretoken in enumerate(pretokens):
            pretokens[i] = self._apply_all_merges(int_merges, pretoken)
        return [id_ for merged_pretoken in pretokens for id_ in merged_pretoken]

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
