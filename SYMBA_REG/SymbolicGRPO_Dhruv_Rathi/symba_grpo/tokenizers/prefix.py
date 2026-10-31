"""SymPy expression <-> prefix token list conversion.

Ported from Aryamaan's sympy_prefix.py, which follows the convention
of Lample & Charton (2019) and Charton (2021).
"""

import sys

import sympy as sp
import numpy as np

sys.setrecursionlimit(3000)

# ── Operator tables ──────────────────────────────────────────────────────────

OPERATORS = {
    sp.Add: "add", sp.Mul: "mul", sp.Pow: "pow",
    sp.exp: "exp", sp.log: "ln", sp.Abs: "abs", sp.sign: "sign",
    sp.sin: "sin", sp.cos: "cos", sp.tan: "tan",
    sp.cot: "cot", sp.sec: "sec", sp.csc: "csc",
    sp.asin: "asin", sp.acos: "acos", sp.atan: "atan",
    sp.acot: "acot", sp.asec: "asec", sp.acsc: "acsc",
    sp.sinh: "sinh", sp.cosh: "cosh", sp.tanh: "tanh",
    sp.coth: "coth", sp.sech: "sech", sp.csch: "csch",
    sp.asinh: "asinh", sp.acosh: "acosh", sp.atanh: "atanh",
    sp.acoth: "acoth", sp.asech: "asech", sp.acsch: "acsch",
    sp.Derivative: "derivative",
}

OPERATORS_INV = {v: k for k, v in OPERATORS.items()}

NARGS = {
    "add": 2, "sub": 2, "mul": 2, "div": 2, "pow": 2, "rac": 2,
    "inv": 1, "pow2": 1, "pow3": 1, "pow4": 1, "pow5": 1,
    "sqrt": 1, "exp": 1, "ln": 1, "abs": 1, "sign": 1,
    "sin": 1, "cos": 1, "tan": 1, "cot": 1, "sec": 1, "csc": 1,
    "asin": 1, "acos": 1, "atan": 1, "acot": 1, "asec": 1, "acsc": 1,
    "sinh": 1, "cosh": 1, "tanh": 1, "coth": 1, "sech": 1, "csch": 1,
    "asinh": 1, "acosh": 1, "atanh": 1, "acoth": 1, "asech": 1, "acsch": 1,
    "derivative": 2, "f": 1, "g": 2, "h": 3,
}

INTEGERS_TYPES = [
    sp.core.numbers.Integer,
    sp.core.numbers.One,
    sp.core.numbers.NegativeOne,
    sp.core.numbers.Zero,
]

NUMBERS_TYPES = INTEGERS_TYPES + [
    sp.core.numbers.Rational, sp.core.numbers.Half,
    sp.core.numbers.Exp1, sp.core.numbers.Pi,
    sp.core.numbers.ImaginaryUnit,
]

ATOMS = [str, sp.core.symbol.Symbol, sp.core.numbers.Exp1, sp.core.numbers.Pi] + NUMBERS_TYPES

# ── Helpers ──────────────────────────────────────────────────────────────────

def _flatten(lst, ltypes=(list, tuple)):
    ltype = type(lst)
    lst = list(lst)
    i = 0
    while i < len(lst):
        while isinstance(lst[i], ltypes):
            if not lst[i]:
                lst.pop(i)
                i -= 1
                break
            else:
                lst[i:i + 1] = lst[i]
        i += 1
    return ltype(lst)


def _repeat_binary(op, args, ret=None):
    """Expand a variadic binary op into nested binary calls.

    Convention: 1 + 2 + 3 -> add 1 add 2 3  (same as Charton 2021).
    """
    if ret is None:
        ret = []
    if len(args) == 0:
        return ret
    if len(ret) == 0:
        ret = [op] + args[-2:]
        args = args[:-2]
    else:
        ret = [op] + args[-1:] + ret
        args = args[:-1]
    return _repeat_binary(op, args, ret)


# ── Number formatting (Charton style) ───────────────────────────────────────

def format_integer(integer):
    """Integer -> ['s+', '1', '2', '3'] or ['s-', '4', '2']."""
    abs_num = abs(integer.p)
    is_neg = integer.could_extract_minus_sign()
    digits = list(str(abs_num))
    sign = "s-" if is_neg else "s+"
    return [sign] + digits


def format_number(number):
    t = type(number)
    if t in INTEGERS_TYPES:
        return format_integer(number)
    if t == sp.core.numbers.Rational:
        p = sp.sympify(number.p)
        q = sp.sympify(number.q)
        return ["mul"] + format_integer(p) + ["pow"] + format_integer(q) + ["s-", "1"]
    if t == sp.core.numbers.Half:
        return ["mul"] + ["s+", "1"] + ["pow"] + ["s+", "2"] + ["s-", "1"]
    if t == sp.core.numbers.Exp1:
        return ["E"]
    if t == sp.core.numbers.Pi:
        return ["pi"]
    if t == sp.core.numbers.ImaginaryUnit:
        raise ValueError("the real-valued decoder vocabulary does not support I")
    raise NotImplementedError(f"Cannot format number type {t}")


# ── sympy -> prefix ─────────────────────────────────────────────────────────

def sympy_to_prefix(expression) -> list:
    """Convert a SymPy expression to a flat prefix token list."""
    tokens = _flatten(_sympy_to_prefix_rec(expression, []))
    from symba_grpo.tokenizers.vocab import DEFAULT_DECODER_TOKENS

    allowed = set(DEFAULT_DECODER_TOKENS) - {"<pad>", "<bos>", "<eos>"}
    unsupported = [token for token in tokens if token not in allowed]
    if unsupported:
        raise ValueError(f"tokens outside the frozen decoder vocabulary: {unsupported}")
    return tokens


def _sympy_to_prefix_rec(expression, ret):
    if expression in [sp.core.numbers.Pi, sp.core.numbers.ImaginaryUnit]:
        f = expression
    else:
        f = expression.func

    if f in ATOMS:
        if type(expression) in NUMBERS_TYPES:
            return ret + format_number(expression)
        return ret + [str(expression)]

    f_str = OPERATORS[f]
    args = expression.args

    if len(args) == 1 and NARGS.get(f_str, 0) == 1:
        return _sympy_to_prefix_rec(args[0], ret + [f_str])
    if len(args) == 2:
        return ret + [f_str, _sympy_to_prefix_rec(args[0], []), _sympy_to_prefix_rec(args[1], [])]
    if len(args) > 2:
        expanded = [_sympy_to_prefix_rec(a, []) for a in args]
        return ret + _repeat_binary(f_str, expanded)
    return ret


# ── prefix -> sympy ─────────────────────────────────────────────────────────

def prefix_to_sympy(expr_arr, *, evaluate: bool = True):
    """Parse exactly one complete prefix expression, failing closed.

    ``evaluate=False`` preserves the bounded input tree instead of asking
    SymPy to canonicalize intermediate Add/Mul/Pow/function nodes.  It is used
    at the GP/model boundary after the token budget has been checked.
    """
    tokens = list(expr_arr)
    if not tokens:
        raise ValueError("empty prefix expression")
    def parse_at(pos):
        if pos >= len(tokens): raise ValueError("incomplete prefix expression")
        token = tokens[pos]
        from symba_grpo.tokenizers.vocab import DEFAULT_DECODER_TOKENS
        allowed = set(DEFAULT_DECODER_TOKENS) - {"<pad>", "<bos>", "<eos>"}
        if token not in allowed:
            raise ValueError(f"invalid prefix token {token!r}")
        if token in OPERATORS_INV:
            args, nxt = [], pos + 1
            for _ in range(NARGS[token]):
                arg, nxt = parse_at(nxt); args.append(arg)
            return OPERATORS_INV[token](*args, evaluate=evaluate), nxt
        if token in ("s+", "s-"):
            nxt, digits = pos + 1, []
            while nxt < len(tokens) and str(tokens[nxt]) in set("0123456789"):
                digits.append(str(tokens[nxt])); nxt += 1
            if not digits: raise ValueError("signed integer without digits")
            return sp.Integer(("-" if token == "s-" else "") + "".join(digits)), nxt
        if token == "E": return sp.E, pos + 1
        if token == "pi": return sp.pi, pos + 1
        if token.startswith("s_") and token[2:].isdigit(): return sp.Symbol(str(token)), pos + 1
        raise ValueError(f"invalid prefix token {token!r}")
    expression, consumed = parse_at(0)
    if consumed != len(tokens): raise ValueError("trailing prefix tokens")
    return expression
