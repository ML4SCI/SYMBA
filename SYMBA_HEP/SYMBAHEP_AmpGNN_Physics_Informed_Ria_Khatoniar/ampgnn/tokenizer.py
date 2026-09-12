import re
from collections import Counter
from typing import List, Dict

TOK_SPECIALS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[CTX]"]

_re_token = re.compile(
    r"""
    \(-?\d+\)             |
    \^\(-?\d+\)           |
    [-+]?\d+/\d+          |
    \d+/\d+               |
    \^\d+                 |
    s_\d+                 |
    M[ix]?\d+             |
    [A-Za-z_]+            |
    [-+*/^()*,]           |
    \d+
    """, re.VERBOSE
)


_re_prefactor = re.compile(
    r"""
    (?:
        [-+]?\d+(?:/\d+)?(?:\*i)?
      |
        i
    )
    (?=\*|$)
    """,
    re.VERBOSE,
)

def tokenize_expr(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    tokens: List[str] = []
    pos = 0
    n = len(text)

    while pos < n:

        if text[pos].isspace():
            pos += 1
            continue


        prev = tokens[-1] if tokens else None
        is_block_start = prev is None or prev in ("+", "-", "(")

        if is_block_start:
            m_pref = _re_prefactor.match(text, pos)
            if m_pref:
                tokens.append(m_pref.group(0))
                pos = m_pref.end()
                continue

        m = _re_token.match(text, pos)
        if not m:
            raise ValueError(f"Cannot tokenize at position {pos}: {text[pos:pos+20]!r}")
        tokens.append(m.group(0))
        pos = m.end()


    return [t for t in tokens if t != "*"]


class Vocab:
    def __init__(self, stoi: Dict[str,int]):
        self.stoi = dict(stoi)
        self.itos = {i:s for s,i in self.stoi.items()}
        self.pad = self.stoi["[PAD]"]
        self.bos = self.stoi["[BOS]"]
        self.eos = self.stoi["[EOS]"]
        self.unk = self.stoi["[UNK]"]
        self.ctx = self.stoi["[CTX]"]

    @classmethod
    def build(cls, sequences: List[List[str]], min_freq: int = 1) -> "Vocab":
        cnt = Counter()
        for seq in sequences:
            cnt.update(seq)
        stoi = {s:i for i,s in enumerate(TOK_SPECIALS)}
        for tok, n in cnt.items():
            if n >= min_freq and tok not in stoi:
                stoi[tok] = len(stoi)
        return cls(stoi)

    def encode(self, seq: List[str]) -> List[int]:
        return [self.stoi.get(t, self.unk) for t in seq]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos.get(i, "[UNK]") for i in ids]
