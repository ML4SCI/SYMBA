"""Frozen vocabulary encoding and generated-sequence validation."""

from typing import Dict, List

import numpy as np


class Vocab:
    """Token <-> ID mapping with encode/decode helpers."""

    def __init__(self, symbol2id: Dict[str, int]):
        self.symbol2id = dict(symbol2id)
        self.id2symbol = {v: k for k, v in self.symbol2id.items()}

    @property
    def size(self) -> int:
        return len(self.symbol2id)

    @property
    def pad_id(self) -> int:
        return self.symbol2id.get("<pad>", 0)

    @property
    def bos_id(self) -> int:
        return self.symbol2id.get("<bos>", 1)

    @property
    def eos_id(self) -> int:
        return self.symbol2id.get("<eos>", self.size - 1)

    def encode_token(self, token: str) -> int:
        if token not in self.symbol2id:
            raise ValueError(f"token {token!r} is outside the frozen vocabulary")
        return self.symbol2id[token]

    def decode_id(self, idx: int) -> str:
        return self.id2symbol.get(idx, "<unk>")

    def encode_sequence(self, tokens: List[str], add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.encode_token(t) for t in tokens)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode_generated(self, ids: List[int]) -> List[str]:
        """Decode one complete generated expression, failing closed.

        Generated sequences must contain exactly one leading BOS and one
        terminal EOS.  Padding, repeated BOS/EOS, and unknown ids are invalid
        inside the expression body.
        """
        sequence = list(ids)
        if len(sequence) < 3 or sequence[0] != self.bos_id or sequence[-1] != self.eos_id:
            raise ValueError("generated sequence must have leading BOS and terminal EOS")
        body = sequence[1:-1]
        special_ids = {self.pad_id, self.bos_id, self.eos_id}
        if any(index in special_ids or index not in self.id2symbol for index in body):
            raise ValueError("special or unknown token inside generated expression")
        return [self.id2symbol[index] for index in body]

    def pad_sequence(self, ids: List[int], max_len: int) -> np.ndarray:
        if len(ids) > max_len:
            raise ValueError(f"sequence length {len(ids)} exceeds fixed limit {max_len}")
        arr = np.full(max_len, self.pad_id, dtype=np.int64)
        n = len(ids)
        arr[:n] = ids
        return arr

# Default decoder vocab matching Aryamaan's decoder_vocab file
DEFAULT_DECODER_TOKENS = [
    "<pad>", "<bos>",
    "E", "pi",
    "s+", "s-",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "add", "mul", "pow", "exp", "ln", "abs", "sign",
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "acot", "asec", "acsc",
    "sinh", "cosh", "tanh", "coth", "sech", "csch",
    "asinh", "acosh", "atanh", "acoth", "asech", "acsch",
    "s_1", "s_2", "s_3", "s_4", "s_5", "s_6", "s_7", "s_8", "s_9", "s_10", "s_11",
    "<eos>",
]


def get_default_decoder_vocab() -> Vocab:
    return Vocab({tok: i for i, tok in enumerate(DEFAULT_DECODER_TOKENS)})
