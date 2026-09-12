"""Canonicalize commutative mass / Mandelstam factor order in amplitude token sequences."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

_MASS_RE = re.compile(r"^M(i?[A-Za-z0-9]+)?$")
_MANDEL_RE = re.compile(r"^s_\d+$")
_COEFF_RE = re.compile(
    r"^(?:\(-?\d+\)|[-+]?\d+/\d+(?:\*i)?|[-+]?\d+(?:/\d+)?(?:\*i)?|i)$"
)
_EXP_RE = re.compile(r"^\^(\(-?\d+\)|-?\d+)$")


def is_mass(tok: str) -> bool:
    return bool(_MASS_RE.match(tok))


def is_mandel(tok: str) -> bool:
    return bool(_MANDEL_RE.match(tok))


def is_coeff(tok: str) -> bool:
    return bool(_COEFF_RE.match(tok))


def _paren_end(tokens: Sequence[str], i: int) -> int:
    if tokens[i] != "(":
        raise ValueError(f"expected '(' at {i}, got {tokens[i]!r}")
    depth = 0
    for j in range(i, len(tokens)):
        if tokens[j] == "(":
            depth += 1
        elif tokens[j] == ")":
            depth -= 1
            if depth == 0:
                return j
    raise ValueError("unbalanced parentheses")


def _depth_at(tokens: Sequence[str], lo: int, idx: int) -> int:
    depth = 0
    for k in range(lo, idx):
        if tokens[k] == "(":
            depth += 1
        elif tokens[k] == ")":
            depth -= 1
    return depth


def _split_top_level(tokens: Sequence[str], lo: int, hi: int, sep: str) -> List[Tuple[int, int]]:
    parts: List[Tuple[int, int]] = []
    start = lo
    for i in range(lo, hi):
        if tokens[i] == sep and _depth_at(tokens, lo, i) == 0:
            if i > start:
                parts.append((start, i))
            start = i + 1
    if start < hi:
        parts.append((start, hi))
    return parts


def _read_exp(tokens: Sequence[str], i: int) -> Tuple[int, int]:
    if i < len(tokens) and _EXP_RE.match(tokens[i]):
        raw = tokens[i][1:]
        if raw.startswith("(") and raw.endswith(")"):
            raw = raw[1:-1]
        try:
            return int(raw), i + 1
        except ValueError:
            return 1, i + 1
    return 1, i


@dataclass(frozen=True)
class _Factor:
    kind: str
    base: str
    exp: int
    tokens: Tuple[str, ...]

    @property
    def sort_key(self) -> Tuple[str, str, int]:
        return (self.kind, self.base, self.exp)


def _read_factor(tokens: Sequence[str], i: int) -> Tuple[_Factor, int]:
    tok = tokens[i]
    if tok == "(":
        j = _paren_end(tokens, i)
        inner = canonicalize_tokens(list(tokens[i + 1 : j]))
        body = tuple(inner)
        # absorb a trailing exponent so '( ... ) ^2' stays one unit under reordering
        exp, nxt = _read_exp(tokens, j + 1)
        exp_toks = tuple(tokens[j + 1 : nxt])
        return _Factor("paren", "|".join(inner), exp, ("(", *body, ")", *exp_toks)), nxt

    if is_mass(tok):
        exp, nxt = _read_exp(tokens, i + 1)
        unit = (tok,) + tuple(tokens[i + 1 : nxt])
        return _Factor("mass", tok, exp, unit), nxt

    if is_mandel(tok):
        exp, nxt = _read_exp(tokens, i + 1)
        unit = (tok,) + tuple(tokens[i + 1 : nxt])
        return _Factor("mandel", tok, exp, unit), nxt

    if tok in ("g", "e"):
        exp, nxt = _read_exp(tokens, i + 1)
        unit = (tok,) + tuple(tokens[i + 1 : nxt])
        return _Factor("coupling", tok, exp, unit), nxt

    if tok == "i":
        return _Factor("i", "i", 1, ("i",)), i + 1

    if tok == "reg_prop":
        exp, nxt = _read_exp(tokens, i + 1)
        unit = (tok,) + tuple(tokens[i + 1 : nxt])
        return _Factor("reg_prop", tok, exp, unit), nxt

    if is_coeff(tok):
        exp, nxt = _read_exp(tokens, i + 1)
        unit = (tok,) + tuple(tokens[i + 1 : nxt])
        return _Factor("coeff", tok, exp, unit), nxt

    if tok == "-":
        # unary minus attached to next factor
        fac, nxt = _read_factor(tokens, i + 1)
        return _Factor(fac.kind, fac.base, fac.exp, ("-", *fac.tokens)), nxt

    exp, nxt = _read_exp(tokens, i + 1)
    unit = (tok,) + tuple(tokens[i + 1 : nxt])
    return _Factor("other", tok, exp, unit), nxt


def _parse_product(tokens: Sequence[str], lo: int, hi: int) -> Tuple[List[_Factor], List[List[_Factor]]]:
    """Return numerator factors and list-of-factor-lists for each denominator chunk."""
    chunks = _split_top_level(tokens, lo, hi, "/")
    num_lo, num_hi = chunks[0]
    num_factors: List[_Factor] = []
    i = num_lo
    while i < num_hi:
        fac, i = _read_factor(tokens, i)
        num_factors.append(fac)

    den_chunks: List[List[_Factor]] = []
    for den_lo, den_hi in chunks[1:]:
        den_factors: List[_Factor] = []
        j = den_lo
        while j < den_hi:
            fac, j = _read_factor(tokens, j)
            den_factors.append(fac)
        den_chunks.append(den_factors)
    return num_factors, den_chunks


def _canonicalize_product(tokens: Sequence[str], lo: int, hi: int) -> List[str]:
    num_factors, den_chunks = _parse_product(tokens, lo, hi)

    prefix: List[_Factor] = []
    commutative: List[_Factor] = []
    suffix: List[_Factor] = []
    phase = "prefix"

    for fac in num_factors:
        if phase == "prefix" and fac.kind in ("coeff", "i", "coupling"):
            prefix.append(fac)
            continue
        if fac.kind in ("mass", "mandel"):
            phase = "commutative"
            commutative.append(fac)
            continue
        phase = "suffix"
        suffix.append(fac)

    commutative.sort(key=lambda f: f.sort_key)
    ordered = prefix + commutative + suffix

    out: List[str] = []
    for fac in ordered:
        out.extend(fac.tokens)

    for den in den_chunks:
        out.append("/")
        if not den:
            continue
        # Only the first factor after '/' is the actual denominator
        # (a / x y == (a/x)*y), so it must keep its position. Factors after it
        # are numerator multipliers and may be commutatively reordered.
        den_out: List[str] = list(den[0].tokens)
        rest = den[1:]
        rest_comm = sorted(
            (f for f in rest if f.kind in ("mass", "mandel")),
            key=lambda f: f.sort_key,
        )
        rest_other = [f for f in rest if f.kind not in ("mass", "mandel")]
        for fac in rest_comm:
            den_out.extend(fac.tokens)
        for fac in rest_other:
            den_out.extend(fac.tokens)
        out.extend(den_out)
    return out


def _split_signed_terms(tokens: Sequence[str], lo: int, hi: int) -> List[Tuple[int, int, int]]:
    """Split a top-level sum into (sign, start, end) terms.

    Splits on '+' and on *binary* '-' (a minus that follows a completed factor).
    A leading or post-operator '-' is unary and folds into the term's sign.
    """
    terms: List[Tuple[int, int, int]] = []
    depth = 0
    term_start = lo
    sign = 1
    i = lo
    while i < hi:
        t = tokens[i]
        if t == "(":
            depth += 1
        elif t == ")":
            depth -= 1
        elif depth == 0 and t in ("+", "-"):
            prev = tokens[i - 1] if i > term_start else None
            is_binary = prev is not None and prev not in ("+", "-", "*", "/", ",")
            if is_binary:
                terms.append((sign, term_start, i))
                sign = 1 if t == "+" else -1
                term_start = i + 1
            elif t == "-" and i == term_start:
                sign = -sign
                term_start = i + 1
        i += 1
    if term_start < hi:
        terms.append((sign, term_start, hi))
    return terms


def _canonicalize_sum(tokens: Sequence[str], lo: int, hi: int) -> List[str]:
    raw_terms = _split_signed_terms(tokens, lo, hi)
    terms: List[Tuple[str, List[str]]] = []
    for sign, t_lo, t_hi in raw_terms:
        body = _canonicalize_product(tokens, t_lo, t_hi)
        if sign < 0:
            body = ["-", *body]
        terms.append(("+".join(body), body))

    terms.sort(key=lambda x: x[0])
    out: List[str] = []
    for idx, (_, body) in enumerate(terms):
        if idx > 0:
            if body and body[0] == "-":
                out.append("-")
                out.extend(body[1:])
            else:
                out.append("+")
                out.extend(body)
        else:
            out.extend(body)
    return out


def canonicalize_tokens(tokens: Sequence[str]) -> List[str]:
    """Return a canonical token sequence with commutative mass/Mandelstam order fixed."""
    if not tokens:
        return []
    toks = list(tokens)
    return _canonicalize_sum(toks, 0, len(toks))


def safe_canonicalize_tokens(tokens: Sequence[str]) -> List[str]:
    try:
        return canonicalize_tokens(tokens)
    except (ValueError, IndexError):
        return list(tokens)


def tokens_equal_mod_commutative(a: Sequence[str], b: Sequence[str]) -> bool:
    return safe_canonicalize_tokens(list(a)) == safe_canonicalize_tokens(list(b))
