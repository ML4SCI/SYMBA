"""SymPy-based simplification helpers for amplitude token sequences."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sympy as sp

from .diagram_parser import BaseDiagram, parse_diagram
from .expr_canonical import canonicalize_tokens, tokens_equal_mod_commutative
from .mass_rewrite import canonicalize_masses
from .tokenizer import tokenize_expr

LineItem = Tuple[Tuple[str, Optional[str]], str, Dict[str, object]]

_BIN_OPS = {"+", "-", "*", "/", "(", ")", ","}
_IMAG_RE = re.compile(r"^([-+]?\d+(?:/\d+)?)\*i$")


def coupling_for_model(model: str) -> str:
    return "e" if model.upper() == "QED" else "g"


def _needs_implicit_mul(prev: Optional[str], curr: str) -> bool:
    if prev is None:
        return False
    if curr.startswith("**"):
        return False
    if prev in ("(", "+", "-", "*", "/", "**"):
        return False
    if curr in (")", "+", "-", "*", "/", "**"):
        return False
    return True


def _token_to_sympy_piece(tok: str) -> str:
    if tok == "i":
        return "I"
    m = _IMAG_RE.match(tok)
    if m:
        return f"({m.group(1)})*I"
    return tok


def _tokens_to_sympy_string(tokens: Sequence[str]) -> str:
    pieces: List[str] = []
    prev: Optional[str] = None
    for tok in tokens:
        if tok.startswith("^"):
            piece = "**" + tok[1:]
        else:
            piece = _token_to_sympy_piece(tok)
        if _needs_implicit_mul(prev, piece):
            pieces.append("*")
        pieces.append(piece)
        prev = piece
    return "".join(pieces)


def _symbol_locals(expr_text: str) -> Dict[str, sp.Symbol]:
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr_text))
    names.discard("I")
    loc: Dict[str, sp.Symbol] = {name: sp.Symbol(name) for name in names}
    loc["I"] = sp.I
    return loc


def parse_tokens_to_sympy(tokens: Sequence[str]) -> sp.Expr:
    text = _tokens_to_sympy_string(tokens)
    return sp.sympify(text, locals=_symbol_locals(text))


def _expr_to_tokens(expr: sp.Expr) -> List[str]:
    if expr is None or expr == sp.Integer(1):
        return []
    text = sp.sstr(expr)
    text = text.replace("**", " ^")
    return tokenize_expr(text)


def extract_factor_g_num_den(
    expr: sp.Expr, g_symbol: str = "g"
) -> Tuple[sp.Expr, int, sp.Expr, sp.Expr]:
    g = sp.Symbol(g_symbol)
    expr = sp.factor(sp.together(expr))
    if expr == 0:
        return sp.Integer(1), 0, sp.Integer(0), sp.Integer(1)

    g_power = 0
    power = expr.as_powers_dict().get(g, sp.Integer(0))
    if bool(power.is_integer):
        candidate_power = int(power)
        candidate_rest = sp.cancel(expr / (g ** candidate_power))
        if not candidate_rest.has(g):
            g_power = candidate_power
            expr = candidate_rest

    numerator, denominator = sp.fraction(sp.cancel(sp.together(expr)))
    numerator = sp.factor(numerator)
    denominator = sp.factor(denominator)

    num_content, numerator = numerator.as_content_primitive()
    den_content, denominator = denominator.as_content_primitive()
    factor = sp.cancel(num_content / den_content)

    if numerator.could_extract_minus_sign():
        factor = -factor
        numerator = -numerator
    if denominator.could_extract_minus_sign():
        factor = -factor
        denominator = -denominator

    return sp.factor(factor), g_power, sp.factor(numerator), sp.factor(denominator)


def format_factorized_tokens(
    factor: sp.Expr,
    g_power: int,
    numerator: sp.Expr,
    denominator: sp.Expr,
    g_symbol: str,
) -> List[str]:
    out: List[str] = []
    if factor not in (1, sp.Integer(1)):
        out.extend(_expr_to_tokens(factor))
    if g_power:
        out.append(g_symbol)
        if g_power != 1:
            out.append(f"^{g_power}")
    num_toks = _expr_to_tokens(numerator) or ["1"]
    den_toks = _expr_to_tokens(denominator) or ["1"]
    out.extend(["("] + num_toks + [")", "/", "("] + den_toks + [")"])
    return out


def decompose_token_list(
    tokens: Sequence[str], g_symbol: str = "g"
) -> Tuple[List[str], int, List[str], List[str]]:
    fac: List[str] = []
    gp = 0
    num: List[str] = []
    den: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == g_symbol:
            i += 1
            if i < len(tokens) and tokens[i].startswith("^"):
                gp = int(tokens[i][1:].strip("()"))
                i += 1
            else:
                gp = 1
            continue
        if tok == "/":
            i += 1
            if i < len(tokens) and tokens[i] == "(":
                _, den, i = _read_paren_group(tokens, i)
            continue
        if tok == "(":
            _, num, i = _read_paren_group(tokens, i)
            continue
        fac.append(tok)
        i += 1
    return fac, gp, num, den


def _read_paren_group(tokens: Sequence[str], i: int) -> Tuple[str, List[str], int]:
    if tokens[i] != "(":
        raise ValueError("expected '('")
    depth = 0
    start = i
    for j in range(i, len(tokens)):
        if tokens[j] == "(":
            depth += 1
        elif tokens[j] == ")":
            depth -= 1
            if depth == 0:
                return "", list(tokens[start + 1 : j]), j + 1
    raise ValueError("unbalanced parentheses")


_SIMPLIFY_CACHE_PATH = os.environ.get(
    "AMPGNN_SIMPLIFY_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "ampgnn", "simplify_cache.sqlite"),
)
_SIMPLIFY_CACHE_VERSION = 2
_simplify_cache_conn: Optional[sqlite3.Connection] = None


def _get_simplify_cache() -> Optional[sqlite3.Connection]:
    global _simplify_cache_conn
    if _SIMPLIFY_CACHE_PATH in ("", "off", "none"):
        return None
    if _simplify_cache_conn is None:
        os.makedirs(os.path.dirname(_SIMPLIFY_CACHE_PATH), exist_ok=True)
        conn = sqlite3.connect(_SIMPLIFY_CACHE_PATH, timeout=30.0)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS simplify_cache (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.commit()
        _simplify_cache_conn = conn
    return _simplify_cache_conn


def simplify_token_list(
    tokens: Sequence[str],
    g_symbol: str = "g",
    *,
    canonicalize: bool = False,
) -> List[str]:
    cache = _get_simplify_cache()
    key = None
    if cache is not None:
        blob = json.dumps(
            [_SIMPLIFY_CACHE_VERSION, list(tokens), g_symbol]
        ).encode("utf-8")
        key = hashlib.sha256(blob).hexdigest()
        row = cache.execute(
            "SELECT value FROM simplify_cache WHERE key = ?", (key,)
        ).fetchone()
        if row is not None:
            out = json.loads(row[0])
            if canonicalize:
                out = canonicalize_tokens(out)
            return out

    expr = parse_tokens_to_sympy(tokens)
    simplified = sp.factor(sp.together(expr))
    factor, g_power, numerator, denominator = extract_factor_g_num_den(simplified, g_symbol)
    out = format_factorized_tokens(factor, g_power, numerator, denominator, g_symbol)

    if cache is not None and key is not None:
        cache.execute(
            "INSERT OR REPLACE INTO simplify_cache (key, value) VALUES (?, ?)",
            (key, json.dumps(out)),
        )
        cache.commit()

    if canonicalize:
        out = canonicalize_tokens(out)
    return out


def _bd_for_masses(bd_i: BaseDiagram, bd_j: Optional[BaseDiagram]) -> BaseDiagram:
    if bd_j is None:
        return bd_i
    merged_off = dict(bd_i.offshell_by_type)
    for sp, vids in bd_j.offshell_by_type.items():
        merged_off.setdefault(sp, vids)
    return BaseDiagram(
        vertices=bd_i.vertices,
        externals=bd_i.externals,
        offshell_by_type=merged_off,
        internal_edges=bd_i.internal_edges,
    )


def build_target_tokens(
    diagrams: Tuple[str, Optional[str]],
    expr: str,
    meta: Dict[str, object],
    *,
    mass_rewrite_mode: str = "leg_sets",
    mass_rewrite: bool = True,
    simplify_targets: bool = False,
    canonicalize_targets: bool = False,
) -> Tuple[List[str], Dict[str, str]]:
    """Mass-rewrite, tokenize, optional sympy simplify, optional canonicalization."""
    diag_i, diag_j = diagrams
    bd_i = parse_diagram(diag_i)
    bd_j = (
        parse_diagram(diag_j)
        if (diag_j is not None and diag_j != diag_i)
        else None
    )
    if mass_rewrite:
        rewritten, slot_map = canonicalize_masses(
            expr, _bd_for_masses(bd_i, bd_j), mode=mass_rewrite_mode
        )
    else:
        rewritten, slot_map = expr, {}
    toks = tokenize_expr(rewritten)
    if simplify_targets:
        g_symbol = coupling_for_model(str(meta.get("model", "QCD")))
        toks = simplify_token_list(toks, g_symbol=g_symbol, canonicalize=False)
    if canonicalize_targets:
        toks = canonicalize_tokens(toks)
    return toks, slot_map


def prepare_line_tokens(
    diagrams: Tuple[str, Optional[str]],
    expr: str,
    meta: Dict[str, object],
    *,
    mass_rewrite_mode: str = "leg_sets",
    canonicalize_targets: bool = False,
) -> Tuple[List[str], str]:
    toks, _ = build_target_tokens(
        diagrams,
        expr,
        meta,
        mass_rewrite_mode=mass_rewrite_mode,
        simplify_targets=False,
        canonicalize_targets=canonicalize_targets,
    )
    model = str(meta.get("model", "QCD"))
    return toks, coupling_for_model(model)


def _stats(values: Sequence[int]) -> Dict[str, float]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
    }


def analyze_lines(
    lines: Sequence[LineItem],
    *,
    mass_rewrite_mode: str = "leg_sets",
    canonicalize_targets: bool = False,
    simplify_targets: bool = True,
) -> Dict[str, Any]:
    raw_lens: List[int] = []
    simp_lens: List[int] = []
    unique_raw: set[str] = set()
    unique_simp: set[str] = set()
    failures: List[Dict[str, object]] = []

    for idx, (diagrams, expr, meta) in enumerate(lines):
        try:
            raw_tokens, _ = build_target_tokens(
                diagrams,
                expr,
                meta,
                mass_rewrite_mode=mass_rewrite_mode,
                simplify_targets=False,
                canonicalize_targets=False,
            )
            target_tokens, _ = build_target_tokens(
                diagrams,
                expr,
                meta,
                mass_rewrite_mode=mass_rewrite_mode,
                simplify_targets=simplify_targets,
                canonicalize_targets=canonicalize_targets,
            )
            raw_lens.append(len(raw_tokens))
            simp_lens.append(len(target_tokens))
            unique_raw.add(" ".join(raw_tokens))
            unique_simp.add(" ".join(target_tokens))
        except Exception as exc:
            failures.append({"idx": idx, "error": str(exc)})

    n_ok = len(raw_lens)
    n_fail = len(failures)
    reduction = (
        float(statistics.mean(simp_lens) / statistics.mean(raw_lens))
        if raw_lens and statistics.mean(raw_lens)
        else 0.0
    )
    return {
        "n_lines": len(lines),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "raw_len": _stats(raw_lens),
        "simp_len": _stats(simp_lens),
        "reduction_ratio": reduction,
        "unique_raw_forms": len(unique_raw),
        "unique_simp_forms": len(unique_simp),
        "failures": failures[:20],
    }


def _sympy_equal(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> bool:
    try:
        ea = parse_tokens_to_sympy(tokens_a)
        eb = parse_tokens_to_sympy(tokens_b)
        return sp.simplify(ea - eb) == 0
    except Exception:
        return False


def compare_lines(
    lines: Sequence[LineItem],
    *,
    mass_rewrite_mode: str = "leg_sets",
    canonicalize_targets: bool = False,
    simplify_targets: bool = True,
    max_lines: Optional[int] = None,
    progress_every: int = 100,
) -> Tuple[List[Dict[str, object]], Dict[str, Any]]:
    rows: List[Dict[str, object]] = []
    n_match = 0
    n_fail = 0
    total = len(lines) if max_lines is None else min(len(lines), max_lines)

    for idx, (diagrams, expr, meta) in enumerate(lines):
        if max_lines is not None and idx >= max_lines:
            break
        if progress_every > 0 and idx > 0 and idx % progress_every == 0:
            print(f"[compare] {idx}/{total} rows...", flush=True)
        try:
            raw_tokens, _ = build_target_tokens(
                diagrams,
                expr,
                meta,
                mass_rewrite_mode=mass_rewrite_mode,
                simplify_targets=False,
                canonicalize_targets=False,
            )
            export_tokens, _ = build_target_tokens(
                diagrams,
                expr,
                meta,
                mass_rewrite_mode=mass_rewrite_mode,
                simplify_targets=False,
                canonicalize_targets=canonicalize_targets,
            )
            target_tokens, _ = build_target_tokens(
                diagrams,
                expr,
                meta,
                mass_rewrite_mode=mass_rewrite_mode,
                simplify_targets=simplify_targets,
                canonicalize_targets=canonicalize_targets,
            )
            match_tokens = tokens_equal_mod_commutative(target_tokens, export_tokens)
            match_sympy = _sympy_equal(target_tokens, export_tokens)
            if match_tokens or match_sympy:
                n_match += 1
            rows.append(
                {
                    "idx": idx,
                    "model": meta.get("model"),
                    "rank": meta.get("rank"),
                    "pair": meta.get("pair"),
                    "raw_len": len(raw_tokens),
                    "target_len": len(target_tokens),
                    "export_len": len(export_tokens),
                    "match": match_tokens,
                    "match_sympy": match_sympy,
                    "raw_tokens": raw_tokens,
                    "target_tokens": target_tokens,
                    "export_tokens": export_tokens,
                }
            )
        except Exception as exc:
            n_fail += 1
            rows.append(
                {
                    "idx": idx,
                    "model": meta.get("model"),
                    "rank": meta.get("rank"),
                    "error": str(exc),
                }
            )

    summary = {
        "n_lines": len(lines),
        "n_match": n_match,
        "n_mismatch": len(rows) - n_match - n_fail,
        "n_fail": n_fail,
        "match_rate": (n_match / len(lines)) if lines else 0.0,
        "note": (
            "target_tokens are sympy-simplified training targets when simplify_targets=True; "
            "export_tokens are mass-rewritten dataset fields without sympy"
        ),
    }
    return rows, summary
