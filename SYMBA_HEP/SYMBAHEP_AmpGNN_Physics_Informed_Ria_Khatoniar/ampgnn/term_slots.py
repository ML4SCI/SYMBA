"""Structured term-slot targets for non-autoregressive CE decoding."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sympy as sp

from .expr_simplify import decompose_token_list, parse_tokens_to_sympy


TERM_SLOT_CACHE_VERSION = "term-slots-v1"
_term_slot_cache_conn: Optional[sqlite3.Connection] = None
_term_slot_cache_path: Optional[str] = None
_term_slot_cache_readonly = False
_term_slot_cache_hits = 0
_term_slot_cache_misses = 0


class TermSlotCacheMissError(KeyError):
    pass


def term_slot_cache_key(tokens: Sequence[str], coupling_symbol: str) -> str:
    payload = json.dumps(
        {
            "version": TERM_SLOT_CACHE_VERSION,
            "tokens": list(tokens),
            "coupling": str(coupling_symbol),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_requested_readonly() -> bool:
    return os.environ.get("AMPGNN_TERM_SLOT_CACHE_READONLY", "").lower() in {
        "1", "true", "yes", "on",
    }


def _get_term_slot_cache() -> Optional[sqlite3.Connection]:
    global _term_slot_cache_conn, _term_slot_cache_path, _term_slot_cache_readonly
    path = os.environ.get("AMPGNN_TERM_SLOT_CACHE")
    readonly = _cache_requested_readonly()
    if not path:
        return None
    path = os.path.abspath(os.path.expanduser(path))
    if (
        _term_slot_cache_conn is not None
        and (_term_slot_cache_path != path or _term_slot_cache_readonly != readonly)
    ):
        _term_slot_cache_conn.close()
        _term_slot_cache_conn = None
    if _term_slot_cache_conn is None:
        if readonly:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30.0)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            conn = sqlite3.connect(path, timeout=30.0)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS term_slot_cache "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.commit()
        # Fail immediately when a wrong SQLite file is supplied.
        conn.execute("SELECT key, value FROM term_slot_cache LIMIT 0")
        _term_slot_cache_conn = conn
        _term_slot_cache_path = path
        _term_slot_cache_readonly = readonly
    return _term_slot_cache_conn


def term_slot_cache_stats() -> Dict[str, object]:
    return {
        "path": _term_slot_cache_path or os.environ.get("AMPGNN_TERM_SLOT_CACHE"),
        "readonly": _term_slot_cache_readonly,
        "hits": _term_slot_cache_hits,
        "misses": _term_slot_cache_misses,
    }


def _monomial(expr: sp.Expr) -> Dict[str, object]:
    coefficient, factors_expr = expr.as_coeff_Mul(rational=True)
    factors: List[Dict[str, object]] = []
    if factors_expr != 1:
        for base, exponent in sorted(
            factors_expr.as_powers_dict().items(), key=lambda item: sp.default_sort_key(item[0])
        ):
            if not isinstance(base, sp.Symbol) or not bool(exponent.is_integer):
                raise ValueError(f"term is not a symbol-power monomial: {expr}")
            exponent_int = int(exponent)
            if exponent_int <= 0:
                raise ValueError(f"term contains a non-positive exponent: {expr}")
            factors.append({"symbol": str(base), "exponent": exponent_int})
    return {"coefficient": str(coefficient), "factors": factors}


def _build_term_slot_target_uncached(
    tokens: Sequence[str], coupling_symbol: str
) -> Dict[str, object]:
    factor_tokens, coupling_power, numerator_tokens, denominator_tokens = (
        decompose_token_list(tokens, coupling_symbol)
    )
    scalar = parse_tokens_to_sympy(factor_tokens or ["1"])
    if not bool(scalar.is_Rational):
        raise ValueError(f"global scalar must be rational, got {scalar}")

    numerator = sp.expand(parse_tokens_to_sympy(numerator_tokens or ["1"]))
    numerator_terms = [
        _monomial(term)
        for term in sorted(sp.Add.make_args(numerator), key=sp.default_sort_key)
    ]

    denominator = parse_tokens_to_sympy(denominator_tokens or ["1"])
    den_coefficient, den_factor_exprs = sp.factor_list(denominator)
    denominator_factors: List[Dict[str, object]] = []
    if den_coefficient != 1:
        denominator_factors.append({
            "power": 1,
            "terms": [_monomial(sp.sympify(den_coefficient))],
        })
    for base, power in sorted(
        den_factor_exprs, key=lambda item: sp.default_sort_key(item[0])
    ):
        polynomial_terms = [
            _monomial(term)
            for term in sorted(
                sp.Add.make_args(sp.expand(base)), key=sp.default_sort_key
            )
        ]
        denominator_factors.append({"power": int(power), "terms": polynomial_terms})

    return {
        "global": {
            "coefficient": str(scalar),
            "coupling": str(coupling_symbol),
            "coupling_power": int(coupling_power),
        },
        "numerator_terms": numerator_terms,
        "denominator_factors": denominator_factors,
    }


def build_term_slot_target(
    tokens: Sequence[str], coupling_symbol: str, *, use_cache: bool = True
) -> Dict[str, object]:
    """Decompose one compact target, optionally using the versioned SQLite cache."""
    global _term_slot_cache_hits, _term_slot_cache_misses
    cache = _get_term_slot_cache() if use_cache else None
    key = term_slot_cache_key(tokens, coupling_symbol)
    if cache is not None:
        row = cache.execute(
            "SELECT value FROM term_slot_cache WHERE key = ?", (key,)
        ).fetchone()
        if row is not None:
            _term_slot_cache_hits += 1
            return json.loads(row[0])
        _term_slot_cache_misses += 1
        if _term_slot_cache_readonly:
            raise TermSlotCacheMissError(
                "structured target is absent from the read-only term-slot cache: "
                f"{key}"
            )

    target = _build_term_slot_target_uncached(tokens, coupling_symbol)
    if cache is not None:
        cache.execute(
            "INSERT OR REPLACE INTO term_slot_cache (key, value) VALUES (?, ?)",
            (key, json.dumps(target, sort_keys=True, separators=(",", ":"))),
        )
        cache.commit()
    return target


def _all_monomials(target: Dict[str, object]) -> Iterable[Dict[str, object]]:
    yield from target["numerator_terms"]
    for denominator_factor in target["denominator_factors"]:
        yield from denominator_factor["terms"]


@dataclass(frozen=True)
class TermSlotVocabulary:
    coefficient_to_id: Dict[str, int]
    symbol_to_id: Dict[str, int]
    coupling_to_id: Dict[str, int]

    UNK = "[UNK]"

    @classmethod
    def build(cls, targets: Sequence[Dict[str, object]]) -> "TermSlotVocabulary":
        coefficients = {cls.UNK}
        symbols = {cls.UNK}
        couplings = {cls.UNK}
        for target in targets:
            coefficients.add(target["global"]["coefficient"])
            couplings.add(target["global"]["coupling"])
            for monomial in _all_monomials(target):
                coefficients.add(monomial["coefficient"])
                symbols.update(factor["symbol"] for factor in monomial["factors"])
        return cls(
            coefficient_to_id={value: i for i, value in enumerate(sorted(coefficients))},
            symbol_to_id={value: i for i, value in enumerate(sorted(symbols))},
            coupling_to_id={value: i for i, value in enumerate(sorted(couplings))},
        )

    @property
    def coefficient_size(self) -> int:
        return len(self.coefficient_to_id)

    @property
    def symbol_size(self) -> int:
        return len(self.symbol_to_id)

    @property
    def coupling_size(self) -> int:
        return len(self.coupling_to_id)

    def as_dict(self) -> Dict[str, Dict[str, int]]:
        return {
            "coefficient_to_id": dict(self.coefficient_to_id),
            "symbol_to_id": dict(self.symbol_to_id),
            "coupling_to_id": dict(self.coupling_to_id),
        }

    def encode(self, target: Dict[str, object]) -> Tuple[Dict[str, object], bool]:
        unknown = False

        def lookup(mapping: Dict[str, int], value: str) -> int:
            nonlocal unknown
            if value not in mapping:
                unknown = True
            return mapping.get(value, mapping[self.UNK])

        def encode_monomial(monomial: Dict[str, object]) -> Dict[str, object]:
            return {
                "coefficient": lookup(
                    self.coefficient_to_id, str(monomial["coefficient"])
                ),
                "factors": [
                    {
                        "symbol": lookup(self.symbol_to_id, str(factor["symbol"])),
                        "exponent": int(factor["exponent"]),
                    }
                    for factor in monomial["factors"]
                ],
            }

        encoded = {
            "global": {
                "coefficient": lookup(
                    self.coefficient_to_id, str(target["global"]["coefficient"])
                ),
                "coupling": lookup(
                    self.coupling_to_id, str(target["global"]["coupling"])
                ),
                "coupling_power": int(target["global"]["coupling_power"]),
            },
            "numerator_terms": [
                encode_monomial(term) for term in target["numerator_terms"]
            ],
            "denominator_factors": [
                {
                    "power": int(factor["power"]),
                    "terms": [encode_monomial(term) for term in factor["terms"]],
                }
                for factor in target["denominator_factors"]
            ],
            "topology_degree": dict(target["topology_degree"]),
        }
        return encoded, unknown

    def symbol_degree2(self) -> List[int]:
        """Twice the mass dimension for each symbol class."""
        values = [0] * self.symbol_size
        for symbol, symbol_id in self.symbol_to_id.items():
            if symbol == self.UNK:
                degree2 = 0
            elif symbol.startswith("M") or symbol.startswith("m_"):
                degree2 = 1
            else:
                # Mandelstam invariants and the regulator carry mass dimension two.
                degree2 = 2
            values[int(symbol_id)] = degree2
        return values


def validate_term_slot_target(
    target: Dict[str, object],
    *,
    max_num_terms: int,
    max_num_factors: int,
    max_den_factors: int,
    max_den_terms: int,
    max_den_term_factors: int,
    max_exponent: int,
) -> None:
    if len(target["numerator_terms"]) > max_num_terms:
        raise ValueError("numerator exceeds configured term-slot capacity")
    if len(target["denominator_factors"]) > max_den_factors:
        raise ValueError("denominator exceeds configured factor-slot capacity")
    if int(target["global"]["coupling_power"]) > max_exponent:
        raise ValueError("coupling power exceeds configured exponent classes")
    for monomial in target["numerator_terms"]:
        if len(monomial["factors"]) > max_num_factors:
            raise ValueError("numerator monomial exceeds configured factor capacity")
    for denominator_factor in target["denominator_factors"]:
        if int(denominator_factor["power"]) > max_exponent:
            raise ValueError("denominator power exceeds configured exponent classes")
        if len(denominator_factor["terms"]) > max_den_terms:
            raise ValueError("denominator polynomial exceeds configured term capacity")
        for monomial in denominator_factor["terms"]:
            if len(monomial["factors"]) > max_den_term_factors:
                raise ValueError("denominator monomial exceeds configured factor capacity")
    for monomial in _all_monomials(target):
        if any(int(factor["exponent"]) > max_exponent for factor in monomial["factors"]):
            raise ValueError("monomial exponent exceeds configured exponent classes")


def slot_vocabulary_hash(vocabulary: TermSlotVocabulary) -> str:
    import hashlib
    import json

    payload = json.dumps(vocabulary.as_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
