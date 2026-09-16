"""
Prefix tokenizer and SymPy <-> prefix conversion for SYMBA symbolic regression.

40-token vocabulary matching SymPy elementary functions:
  special(4) + variables(9) + structural numerics(9) +
  constants(3) + operators(15) = 40

The structural numerics cover every exponent the SYMBA_Reg_Data_Gen generator
can produce — {2, 3, 4, -1, -2, -3, -4, 1/2} — so no exponent collapses to the
fittable 'C' token.  That matters because 'C' is not identifiable: x**3 and
x**4 would both tokenize to `pow x1 C`, making them indistinguishable to the
decoder and unrecoverable by exact match.  Literals outside this set still
fall back to 'C'.
"""

from typing import Optional

import sympy as sp


# ============================================================================
# Prefix conversion constants
# ============================================================================

SPECIFIC_NUMERICS = {
    sp.Integer(-1):     'neg1',
    sp.Integer(-2):     'neg2',
    sp.Integer(-3):     'neg3',
    sp.Integer(-4):     'neg4',
    sp.Rational(1, 2):  'half',
    sp.Rational(-1, 2): 'neghalf',
    sp.Integer(2):      'two',
    sp.Integer(3):      'three',
    sp.Integer(4):      'four',
}

FUNC_MAP = {
    sp.sin:  'sin',  sp.cos:  'cos',  sp.tan:  'tan',
    sp.asin: 'asin', sp.acos: 'acos', sp.atan: 'atan',
    sp.exp:  'exp',  sp.log:  'log',
    sp.tanh: 'tanh', sp.sinh: 'sinh', sp.cosh: 'cosh',
}

BINARY_OPS = {
    'add': lambda a, b: a + b,
    'mul': lambda a, b: a * b,
    'pow': lambda a, b: a ** b,
}

UNARY_OPS = {
    'neg':  lambda a: -a,
    'sin':  sp.sin,  'cos':  sp.cos,  'tan':  sp.tan,
    'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
    'exp':  sp.exp,  'log':  sp.log,
    'tanh': sp.tanh, 'sinh': sp.sinh, 'cosh': sp.cosh,
}

NUMERIC_VALUES = {
    'neg1':    sp.Integer(-1),
    'neg2':    sp.Integer(-2),
    'neg3':    sp.Integer(-3),
    'neg4':    sp.Integer(-4),
    'half':    sp.Rational(1, 2),
    'neghalf': sp.Rational(-1, 2),
    'two':     sp.Integer(2),
    'three':   sp.Integer(3),
    'four':    sp.Integer(4),
}


# ============================================================================
# SymPy -> prefix
# ============================================================================

def sympy_to_prefix(expr, var_map: dict[str, str]) -> Optional[str]:
    """Convert a SymPy expression to a space-separated prefix string.

    var_map maps original variable names to numbered tokens, e.g.
    {"theta": "x1", "sigma": "x2"}.  Constants become 'C' unless they
    are structural (SPECIFIC_NUMERICS, pi, e).
    """
    def _rec(node):
        if node == sp.pi:
            return 'pi'
        if node == sp.E:
            return 'e'

        if node.is_Number:
            if node in SPECIFIC_NUMERICS:
                return SPECIFIC_NUMERICS[node]
            return 'C'

        if node.is_Symbol:
            name = str(node)
            if name in var_map:
                return var_map[name]
            # Unknown symbol treated as fittable constant
            return 'C'

        if node.is_Add:
            args = [_rec(a) for a in node.args]
            out = args[0]
            for a in args[1:]:
                out = f'add {out} {a}'
            return out

        if node.is_Mul:
            args = [_rec(a) for a in node.args]
            out = args[0]
            for a in args[1:]:
                out = f'mul {out} {a}'
            return out

        if node.is_Pow:
            b, ex = node.args
            return f'pow {_rec(b)} {_rec(ex)}'

        if node.func in FUNC_MAP:
            return f'{FUNC_MAP[node.func]} {_rec(node.args[0])}'

        # Fallback: try to represent as string (will likely produce <unk>)
        return str(node)

    try:
        return _rec(expr)
    except Exception:
        return None


# ============================================================================
# Prefix -> SymPy
# ============================================================================

def prefix_to_sympy(tokens, strict: bool = True):
    """Parse a prefix string (or token list) into a SymPy expression.

    Args:
        tokens: prefix string or list of tokens.
        strict: reject input whose tokens are not fully consumed by one
            expression.  Prefix notation is self-delimiting, so
            ``add x1 C mul x1 C`` contains a complete expression
            (``add x1 C``) followed by garbage.  Parsing only the prefix
            makes a malformed prediction look algebraically correct while
            exact-match — which sees the whole string — fails, producing the
            contradictory "equiv=True, exact=False" pattern.  Pass
            ``strict=False`` only if partial parsing is genuinely wanted.

    Returns:
        (expr, constants) where constants is a list of sp.Symbol objects
        named c_0, c_1, ... representing fittable constants.

    Raises:
        ValueError: on unknown/truncated tokens, or (when strict) on tokens
            trailing a complete expression.
    """
    if isinstance(tokens, str):
        tokens = tokens.split()
    constants = []

    def _parse(i):
        if i >= len(tokens):
            raise ValueError(f'Truncated at position {i}')
        tok = tokens[i]

        if tok == 'C':
            c = sp.Symbol(f'c_{len(constants)}')
            constants.append(c)
            return c, i + 1
        if tok == 'pi':
            return sp.pi, i + 1
        if tok == 'e':
            return sp.E, i + 1
        if tok in NUMERIC_VALUES:
            return NUMERIC_VALUES[tok], i + 1
        if tok.startswith('x') and tok[1:].isdigit():
            return sp.Symbol(tok), i + 1
        if tok in BINARY_OPS:
            left, i = _parse(i + 1)
            right, i = _parse(i)
            return BINARY_OPS[tok](left, right), i
        if tok in UNARY_OPS:
            arg, i = _parse(i + 1)
            return UNARY_OPS[tok](arg), i

        raise ValueError(f'Unknown token: {tok}')

    expr, end_idx = _parse(0)
    if strict and end_idx != len(tokens):
        raise ValueError(
            f'Trailing tokens after complete prefix expression: '
            f'{tokens[end_idx:]}'
        )
    return expr, constants


# ============================================================================
# PrefixTokenizer
# ============================================================================

class PrefixTokenizer:
    """Tokenizer for prefix-notation symbolic expressions.

    Base vocabulary:
      - Special: <pad>, <sos>, <eos>, <unk>
      - Variables: x1..x9
      - Structural numerics: neg1, neg2, neg3, neg4, half, neghalf,
                             two, three, four
      - Constants: C (fittable), pi, e
      - Operators: add, mul, pow, neg, sin, cos, tan, asin, acos, atan,
                   exp, log, tanh, sinh, cosh

    Extendable via extend() for concept library tokens.
    """

    def __init__(self, max_vars: int = 9):
        special   = ['<pad>', '<sos>', '<eos>', '<unk>']
        variables = [f'x{i+1}' for i in range(max_vars)]
        numerics  = ['neg1', 'neg2', 'neg3', 'neg4',
                     'half', 'neghalf',
                     'two', 'three', 'four']
        constants = ['C', 'pi', 'e']
        operators = [
            'add', 'mul', 'pow', 'neg',
            'sin', 'cos', 'tan',
            'asin', 'acos', 'atan',
            'exp', 'log',
            'tanh', 'sinh', 'cosh',
        ]

        self.vocab = special + variables + numerics + constants + operators
        self.token2id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id2token = {i: tok for i, tok in enumerate(self.vocab)}
        self.pad_id = self.token2id['<pad>']
        self.sos_id = self.token2id['<sos>']
        self.eos_id = self.token2id['<eos>']
        self.unk_id = self.token2id['<unk>']

    def __len__(self):
        return len(self.vocab)

    def encode(self, prefix_str: str, add_special: bool = True) -> list[int]:
        """Encode a space-separated prefix string to token IDs."""
        tokens = prefix_str.split()
        ids = [self.token2id.get(t, self.unk_id) for t in tokens]
        if add_special:
            ids = [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special: bool = True) -> str:
        """Decode token IDs back to a space-separated string.

        Reading STOPS at the first ``<eos>``: it terminates the sequence, so
        anything after it is not part of the prediction.  Filtering ``<eos>``
        out of the list instead (the old behaviour) silently spliced post-EOS
        padding text onto an otherwise correct prediction, e.g.
        ``<sos> add x1 C <eos> mul x1 C`` -> ``add x1 C mul x1 C``.
        """
        tokens = []
        for i in ids:
            tok = self.id2token.get(int(i), '<unk>')
            if tok == '<eos>':
                break
            if skip_special and tok in ('<pad>', '<sos>'):
                continue
            tokens.append(tok)
        return ' '.join(tokens)

    def extend(self, new_tokens: list[str]):
        """Add new tokens (e.g. concept library entries)."""
        for tok in new_tokens:
            if tok not in self.token2id:
                idx = len(self.vocab)
                self.vocab.append(tok)
                self.token2id[tok] = idx
                self.id2token[idx] = tok
