"""Adapter from our point-cloud dataset to the upstream SymbolicGPT interface.

Upstream (``mojivalipour/symbolicgpt``) consumes, per example,

    (inputs, outputs, points, numVars)

where ``inputs``/``outputs`` are a **character-level** encoding of the equation
skeleton shifted by one, ``points`` is a ``(numVars + numYs, numPoints)``
tensor, and ``numVars`` is the variable count.  Our pipeline instead produces
prefix token IDs and a ``(numPoints, max_vars + 1)`` point cloud.

This module bridges the two *without touching either side's semantics*:

* ``prefix_to_infix`` / ``infix_to_prefix`` convert between our prefix token
  strings and a fully parenthesised infix skeleton string — the representation
  SymbolicGPT is character-level over.  The pair is exactly invertible on every
  prefix string our tokenizer can emit (see ``check_roundtrip``), so a
  SymbolicGPT prediction can be scored by our existing prefix-based evaluator.
* ``SymbolicGPTDataset`` wraps a ``PointCloudDataset`` and emits upstream's
  4-tuple, transposing the cloud to upstream's channel-first layout.

Nothing here reimplements model, trainer, or metric logic: the baseline stays
an independent upstream run over our data.
"""

import re

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Prefix <-> infix skeleton
# ---------------------------------------------------------------------------

# Prefix operator tokens -> infix operator text.  Every binary application is
# emitted fully parenthesised, `(A op B)`, so the string needs no precedence
# rules to be read back unambiguously.
BINARY_OPS = {'add': '+', 'mul': '*', 'pow': '**'}

UNARY_FUNCS = (
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
    'exp', 'log', 'tanh', 'sinh', 'cosh',
)

# Non-numeric leaves.  'C' is SymbolicGPT's own fittable-constant placeholder,
# and it is also our prefix vocabulary's, so it passes through unchanged.
SYMBOL_LEAVES = ('C', 'pi', 'e')

# Structural numerics -> literal text.  Negatives are parenthesised so they can
# never fuse with a neighbouring operator (`x1*-1` is avoided; `(x1*(-1))` is
# what gets emitted).
NUMERIC_TEXT = {
    'two': '2', 'three': '3', 'four': '4',
    'half': '0.5', 'neghalf': '(-0.5)',
    'neg1': '(-1)', 'neg2': '(-2)', 'neg3': '(-3)', 'neg4': '(-4)',
}
# Reverse map, keyed on the bare number text (parentheses stripped).
TEXT_NUMERIC = {v.strip('()'): k for k, v in NUMERIC_TEXT.items()}

_VAR_RE = re.compile(r'x[1-9][0-9]*$')
_IDENT_RE = re.compile(r'[A-Za-z][A-Za-z0-9]*')
_NUMBER_RE = re.compile(r'-?[0-9]+(?:\.[0-9]+)?')


def prefix_to_infix(prefix: str) -> str:
    """Convert one of our prefix token strings to an infix skeleton string.

    Args:
        prefix: space-separated prefix tokens, e.g. ``'add mul x1 C sin x1'``.

    Returns:
        Fully parenthesised infix text, e.g. ``'((x1*C)+sin(x1))'``.

    Raises:
        ValueError: on an unknown or truncated token sequence.
    """
    tokens = prefix.split()

    def _rec(i):
        if i >= len(tokens):
            raise ValueError(f'truncated prefix at position {i}: {prefix!r}')
        tok = tokens[i]

        if tok in BINARY_OPS:
            left, i = _rec(i + 1)
            right, i = _rec(i)
            return f'({left}{BINARY_OPS[tok]}{right})', i
        if tok in UNARY_FUNCS:
            arg, i = _rec(i + 1)
            return f'{tok}({arg})', i
        if tok == 'neg':
            arg, i = _rec(i + 1)
            return f'(-{arg})', i
        if tok in NUMERIC_TEXT:
            return NUMERIC_TEXT[tok], i + 1
        if tok in SYMBOL_LEAVES or _VAR_RE.match(tok):
            return tok, i + 1

        raise ValueError(f'unknown prefix token {tok!r} in {prefix!r}')

    text, end = _rec(0)
    if end != len(tokens):
        raise ValueError(f'trailing prefix tokens after a complete expression: '
                         f'{tokens[end:]}')
    return text


def infix_to_prefix(infix: str) -> str:
    """Parse an infix skeleton string back into our prefix token string.

    The inverse of :func:`prefix_to_infix`.  Model output that does not fit the
    grammar raises, which is the honest outcome: our evaluator then counts the
    prediction as unparseable, exactly as it would an unparseable prefix string.

    Args:
        infix: infix text as produced by :func:`prefix_to_infix`.

    Returns:
        Space-separated prefix token string.

    Raises:
        ValueError: on any character, token, or structure outside the grammar.
    """
    s = infix.strip()
    if not s:
        raise ValueError('empty expression')

    def _expect(i, ch):
        if i >= len(s) or s[i] != ch:
            got = s[i] if i < len(s) else 'end of string'
            raise ValueError(f'expected {ch!r} at position {i}, got {got!r}')
        return i + 1

    def _number(i):
        """Return (token, next_index) for a numeric literal at *i*, or None."""
        m = _NUMBER_RE.match(s, i)
        if not m:
            return None
        tok = TEXT_NUMERIC.get(m.group(0))
        if tok is None:
            raise ValueError(f'numeric literal {m.group(0)!r} at position {i} '
                             f'is outside the vocabulary')
        return tok, m.end()

    def _rec(i):
        if i >= len(s):
            raise ValueError(f'truncated expression at position {i}')

        if s[i] == '(':
            i += 1
            # `(-...)` is either a negative numeric literal or a negation.
            if s[i:i + 1] == '-':
                num = _number(i)
                if num is not None and s[num[1]:num[1] + 1] == ')':
                    return [num[0]], num[1] + 1
                arg, i = _rec(i + 1)
                return ['neg'] + arg, _expect(i, ')')

            left, i = _rec(i)
            # '**' must be tested before '*'.
            if s[i:i + 2] == '**':
                op, i = 'pow', i + 2
            elif s[i:i + 1] == '+':
                op, i = 'add', i + 1
            elif s[i:i + 1] == '*':
                op, i = 'mul', i + 1
            else:
                got = s[i] if i < len(s) else 'end of string'
                raise ValueError(f'expected a binary operator at position {i}, '
                                 f'got {got!r}')
            right, i = _rec(i)
            return [op] + left + right, _expect(i, ')')

        m = _IDENT_RE.match(s, i)
        if m:
            name = m.group(0)
            if name in UNARY_FUNCS:
                i = _expect(m.end(), '(')
                arg, i = _rec(i)
                return [name] + arg, _expect(i, ')')
            if name in SYMBOL_LEAVES or _VAR_RE.match(name):
                return [name], m.end()
            raise ValueError(f'unknown identifier {name!r} at position {i}')

        num = _number(i)
        if num is not None:
            return [num[0]], num[1]

        raise ValueError(f'unexpected character {s[i]!r} at position {i}')

    tokens, end = _rec(0)
    if end != len(s):
        raise ValueError(f'trailing text after a complete expression: '
                         f'{s[end:]!r}')
    return ' '.join(tokens)


def check_roundtrip(prefix_strings, sample: int = 0) -> dict:
    """Verify ``infix_to_prefix(prefix_to_infix(p)) == p`` over *prefix_strings*.

    Args:
        prefix_strings: iterable of prefix token strings.
        sample: if > 0, check only the first *sample* strings.

    Returns:
        Dict with 'n_checked', 'n_failed', and up to 5 'failures'
        as ``(prefix, infix_or_error)`` pairs.
    """
    strings = list(prefix_strings)
    if sample > 0:
        strings = strings[:sample]

    failures = []
    for p in strings:
        try:
            text = prefix_to_infix(p)
            back = infix_to_prefix(text)
            if back != p:
                failures.append((p, text))
        except Exception as exc:            # noqa: BLE001 - reported, not raised
            failures.append((p, f'{type(exc).__name__}: {exc}'))

    return {
        'n_checked': len(strings),
        'n_failed': len(failures),
        'failures': failures[:5],
    }


# ---------------------------------------------------------------------------
# Character vocabulary
# ---------------------------------------------------------------------------

# Upstream's specials: '_' padding, '<' SOS, '>' EOS.  'T' and ':' are in
# upstream's vocabulary too (test marker / STR_VAR separator); they are kept so
# the vocabulary construction stays byte-for-byte the upstream recipe.
SPECIAL_CHARS = ['_', 'T', '<', '>', ':']


def build_char_vocab(equation_strings) -> list[str]:
    """Upstream's character vocabulary recipe, over our equation strings.

    Upstream builds it as ``sorted(set(text) + ['_','T','<','>',':'])`` over the
    concatenated raw data file; the only change here is the source of *text*.
    """
    chars = set()
    for eq in equation_strings:
        chars.update(eq)
    return sorted(set(chars) | set(SPECIAL_CHARS))


# ---------------------------------------------------------------------------
# Dataset adapter
# ---------------------------------------------------------------------------

class SymbolicGPTDataset(Dataset):
    """Serve a ``PointCloudDataset`` through upstream SymbolicGPT's interface.

    Emits upstream's ``(inputs, outputs, points, numVars)`` 4-tuple:

    * ``inputs`` / ``outputs``: ``'<' + skeleton + '>'`` encoded character-wise
      and shifted by one, padded to *block_size* with upstream's ``'_'``.
    * ``points``: our normalised cloud transposed to upstream's channel-first
      ``(numVars + numYs, numPoints)`` layout, which is what ``tNet`` consumes.
    * ``numVars``: variable count, read off the skeleton the way upstream does.

    The point cloud itself — how many points, the sampling support, and the
    per-column normalisation — is entirely the wrapped dataset's business, so
    train examples keep resampling and val/test clouds stay deterministic
    exactly as in our own experiments.

    Args:
        base: a ``PointCloudDataset`` (or ``MultiViewPointCloudDataset``);
            ``base[i]['points']`` and ``base.samples[i]`` are used.
        tokenizer: the ``PrefixTokenizer`` that produced ``base``'s token IDs.
        stoi: character -> ID map (from :func:`build_char_vocab`).
        block_size: upstream context length; sequences are padded to it and
            truncated at it.
        padding_token: upstream's padding character.
    """

    def __init__(self, base, tokenizer, stoi: dict, block_size: int,
                 padding_token: str = '_'):
        self.base = base
        self.block_size = block_size

        # Upstream's CharDataset attribute names.  `Trainer` reads `.itos`, and
        # the model-construction / decoding code reads `.vocab_size`,
        # `.block_size` and `.paddingID`, so the adapter has to answer to them.
        self.stoi = stoi
        self.itos = {i: c for c, i in stoi.items()}
        self.vocab_size = len(stoi)
        self.paddingToken = padding_token
        self.paddingID = stoi[padding_token]

        # Ground-truth prefix strings, decoded the same way the evaluation path
        # decodes them, so `prefix_strings[i]` is the GT that our evaluator will
        # be handed for prediction i.
        self.prefix_strings = [
            tokenizer.decode(s['input_ids'].tolist()) for s in base.samples
        ]
        self.equation_strings = [prefix_to_infix(p) for p in self.prefix_strings]

    def __len__(self):
        return len(self.base)

    def max_encoded_length(self) -> int:
        """Longest ``'<' + skeleton + '>'`` length, i.e. the block size needed."""
        return max(len(eq) + 2 for eq in self.equation_strings)

    def __getitem__(self, idx):
        item = self.base[idx]
        eq = self.equation_strings[idx]

        # Upstream's encoding: '<' is SOS, '>' is EOS, next-token shift, then
        # right-pad both sides with the padding ID and clip to block_size.
        dix = [self.stoi[c] for c in '<' + eq + '>']
        inputs = dix[:-1]
        outputs = dix[1:]
        pad = [self.paddingID] * max(self.block_size - len(inputs), 0)
        inputs = (inputs + pad)[:self.block_size]
        outputs = (outputs + pad)[:self.block_size]

        # (numPoints, numVars + numYs) -> upstream's channel-first layout.
        points = item['points'].transpose(0, 1).contiguous()

        num_vars = 0
        for m in re.finditer(r'x([0-9]+)', eq):
            num_vars = max(num_vars, int(m.group(1)))

        return (
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(outputs, dtype=torch.long),
            points,
            torch.tensor(num_vars, dtype=torch.long),
        )


def decode_prediction(ids, itos: dict, padding_token: str = '_') -> str:
    """Turn generated character IDs into a skeleton string, upstream's way.

    Mirrors the post-processing in upstream ``symbolicGPT.py``: join the
    characters, drop padding, cut at the first ``'>'`` (EOS), strip the leading
    ``'<'``.
    """
    text = ''.join(itos[int(i)] for i in ids)
    text = text.strip(padding_token).split('>')[0]
    return text.strip('<').strip('>')
