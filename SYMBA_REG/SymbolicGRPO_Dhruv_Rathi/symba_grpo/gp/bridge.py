"""Bounded conversions between SymPy, DEAP trees, and model prefix tokens.

Ported from Aryamaan's sympy_deap.py.
"""

from dataclasses import dataclass

import sympy as sp
from deap import gp


class TreePrefixError(ValueError):
    """A GP tree cannot be represented by the frozen decoder vocabulary."""


@dataclass(frozen=True)
class GPExpression:
    """A GP result kept structural so the parent never simplifies it."""

    tree: gp.PrimitiveTree
    num_vars: int

    @classmethod
    def from_tree(cls, tree: gp.PrimitiveTree, num_vars: int) -> "GPExpression":
        # Copy the compact node sequence; never construct a symbolic expression.
        return cls(gp.PrimitiveTree(tree), int(num_vars))

    def __str__(self) -> str:
        return str(self.tree)

    def exact_worker_record(self) -> dict:
        nodes = []
        for node in self.tree:
            if isinstance(node, gp.Primitive):
                nodes.append(["p", node.name, node.arity])
            elif isinstance(node, gp.Terminal):
                nodes.append(["t", node.value, 0])
            else:
                raise TreePrefixError(f"unsupported GP node type: {type(node).__name__}")
        return {
            "version": 1,
            "source": "gp",
            "nodes": nodes,
            "num_vars": self.num_vars,
        }


# The GP tree is already stored in prefix order.  These are the only
# primitives in ``make_primitive_set``; mappings that need more than one model
# token are expanded without constructing a SymPy object.
GP_PRIMITIVE_ARITIES = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "protected_div": 2,
    "protected_pow": 2,
    "protected_exp": 1,
    "protected_log": 1,
    "protected_sqrt": 1,
    "sin": 1,
    "cos": 1,
    "tan": 1,
    "tanh": 1,
    "abs": 1,
}


def deap_tree_fingerprint(tree: gp.PrimitiveTree) -> tuple:
    """Return a compact structural key without rendering a functional string."""
    fingerprint = []
    for node in tree:
        if isinstance(node, gp.Primitive):
            fingerprint.append(("p", node.name, node.arity))
        elif isinstance(node, gp.Terminal):
            fingerprint.append(("t", node.value, 0))
        else:
            raise TreePrefixError(f"unsupported GP node type: {type(node).__name__}")
    return tuple(fingerprint)


def deap_tree_to_prefix(
    tree: gp.PrimitiveTree,
    *,
    num_vars: int,
    max_body_tokens: int,
) -> list[str]:
    """Compile a DEAP prefix tree directly to bounded decoder-prefix tokens.

    No expression parsing or symbolic simplification occurs here.  Traversal
    stops as soon as the emitted representation cannot fit in the model's
    sequence body.  This is an interface limit imposed by the frozen decoder,
    not a limit on the GP search tree itself.
    """
    if not 1 <= int(num_vars) <= 11:
        raise TreePrefixError(f"invalid declared variable count: {num_vars}")
    if max_body_tokens <= 0:
        raise TreePrefixError("prefix-token budget must be positive")
    if not isinstance(tree, gp.PrimitiveTree) or len(tree) == 0:
        raise TreePrefixError("expected a nonempty DEAP PrimitiveTree")

    emitted = 0

    def bounded(parts: list[str]) -> list[str]:
        nonlocal emitted
        emitted += len(parts)
        if emitted > max_body_tokens:
            raise TreePrefixError(
                f"GP tree exceeds the {max_body_tokens}-token decoder body limit"
            )
        return parts

    def compile_at(index: int) -> tuple[list[str], int]:
        if index >= len(tree):
            raise TreePrefixError("incomplete DEAP prefix tree")
        node = tree[index]
        if isinstance(node, gp.Terminal):
            if node.arity != 0:
                raise TreePrefixError("terminal with nonzero arity")
            value = node.value
            if isinstance(value, str) and value.startswith("s_") and value[2:].isdigit():
                variable_index = int(value[2:])
                if not 1 <= variable_index <= num_vars:
                    raise TreePrefixError(f"undeclared variable terminal: {value}")
                return bounded([value]), index + 1
            if value in ("pi", "E"):
                return bounded([str(value)]), index + 1
            # bool is an int subclass but is not a declared GP constant.
            if type(value) is int and 1 <= value <= 4:
                return bounded(["s+", str(value)]), index + 1
            raise TreePrefixError(f"unsupported GP terminal: {value!r}")

        if not isinstance(node, gp.Primitive):
            raise TreePrefixError(f"unsupported GP node type: {type(node).__name__}")
        name = node.name
        expected_arity = GP_PRIMITIVE_ARITIES.get(name)
        if expected_arity is None or node.arity != expected_arity:
            raise TreePrefixError(f"unsupported GP primitive/arity: {name}/{node.arity}")

        # Emit the model operator before descending so a long tree is rejected
        # after visiting at most O(max_body_tokens) representable structure.
        direct = {
            "add": "add", "mul": "mul", "protected_pow": "pow",
            "protected_exp": "exp", "protected_log": "ln",
            "sin": "sin", "cos": "cos", "tan": "tan",
            "tanh": "tanh", "abs": "abs",
        }
        if name in direct:
            result = bounded([direct[name]])
            next_index = index + 1
            for _ in range(expected_arity):
                child, next_index = compile_at(next_index)
                result.extend(child)
            return result, next_index

        if name == "sub":
            result = bounded(["add"])
            left, next_index = compile_at(index + 1)
            result.extend(left)
            result.extend(bounded(["mul", "s-", "1"]))
            right, next_index = compile_at(next_index)
            result.extend(right)
            return result, next_index
        if name == "protected_div":
            result = bounded(["mul"])
            numerator, next_index = compile_at(index + 1)
            result.extend(numerator)
            result.extend(bounded(["pow"]))
            denominator, next_index = compile_at(next_index)
            result.extend(denominator)
            result.extend(bounded(["s-", "1"]))
            return result, next_index
        if name == "protected_sqrt":
            result = bounded(["pow"])
            argument, next_index = compile_at(index + 1)
            result.extend(argument)
            # Charton-style prefix for the exact rational exponent 1/2.
            result.extend(bounded(["mul", "s+", "1", "pow", "s+", "2", "s-", "1"]))
            return result, next_index
        raise TreePrefixError(f"unmapped GP primitive: {name}")

    tokens, consumed = compile_at(0)
    if consumed != len(tree):
        raise TreePrefixError("trailing nodes in DEAP prefix tree")
    return tokens

SYMPY_TO_DEAP_OPS = {
    sp.Add: "add",
    sp.Mul: "mul",
    sp.exp: "exp",
    sp.log: "ln",
    sp.Abs: "abs",
    sp.sin: "sin",
    sp.cos: "cos",
    sp.tan: "tan",
    sp.sinh: "sinh",
    sp.cosh: "cosh",
    sp.tanh: "tanh",
    sp.asin: "asin",
    sp.acos: "acos",
    sp.atan: "atan",
    sp.sign: "sign",
}


BINARY_OPS = {sp.Add, sp.Mul}

_MAX_EXACT_INTEGER = 2**31 - 1
_MAX_EXPANDED_INTEGER_POWER = 16


def _integer_to_deap(value: int) -> str:
    """Encode an integer using only the GP set's declared terminals 1--4."""
    value = int(value)
    if abs(value) > _MAX_EXACT_INTEGER:
        raise ValueError("integer is too large for bounded GP seed lowering")
    if 1 <= value <= 4:
        return str(value)
    if value == 0:
        return "sub(1, 1)"
    if value < 0:
        return f"sub(sub(1, 1), {_integer_to_deap(-value)})"

    quotient, remainder = divmod(value, 4)
    result = f"mul(4, {_integer_to_deap(quotient)})"
    if remainder:
        result = f"add({result}, {_integer_to_deap(remainder)})"
    return result


def _positive_integer_power(base: str, exponent: int) -> str:
    """Lower a small positive integer power to multiplication.

    This preserves the sign of negative bases.  The GP's protected power uses
    ``abs(base)`` and therefore is not a faithful lowering for odd powers.
    """
    if not 1 <= exponent <= _MAX_EXPANDED_INTEGER_POWER:
        raise ValueError("integer power is too large for semantics-preserving GP lowering")
    result = base
    for _ in range(1, exponent):
        result = f"mul({result}, {base})"
    return result


def _power_to_deap(expr: sp.Pow) -> str:
    base_expr, exponent_expr = expr.args
    base = sympy_to_deap(base_expr)
    base_is_nonnegative = (
        base_expr.is_nonnegative is True or base_expr.func is sp.Abs
    )

    if isinstance(exponent_expr, sp.Integer):
        exponent = int(exponent_expr)
        if exponent == 0:
            return "1"
        if abs(exponent) <= _MAX_EXPANDED_INTEGER_POWER:
            powered = _positive_integer_power(base, abs(exponent))
            return powered if exponent > 0 else f"protected_div(1, {powered})"
        if not base_is_nonnegative:
            raise ValueError(
                "cannot faithfully lower a large power of a sign-unknown GP seed"
            )
        return f"protected_pow({base}, {_integer_to_deap(exponent)})"

    # For a noninteger exponent, protected_pow(abs(base), exponent) agrees
    # with the source expression only when nonnegativity is established from
    # the expression itself.  Otherwise reject the seed instead of silently
    # changing its function on negative inputs.
    if not base_is_nonnegative:
        raise ValueError(
            "cannot faithfully lower a noninteger power of a sign-unknown GP seed"
        )
    return f"protected_pow({base}, {sympy_to_deap(exponent_expr)})"


def sympy_to_deap(expr) -> str:
    """Convert a SymPy expression to a DEAP-compatible functional string.

    SymPy Add/Mul are n-ary but DEAP primitives are binary, so we fold
    n-ary operations into left-nested binary calls:
    Mul(a, b, c) -> mul(mul(a, b), c)
    """
    if isinstance(expr, sp.Pow):
        return _power_to_deap(expr)

    for key, value in SYMPY_TO_DEAP_OPS.items():
        if isinstance(expr, key):
            args = expr.args
            converted = [sympy_to_deap(a) for a in args]
            if key in BINARY_OPS and len(converted) > 2:
                result = converted[0]
                for c in converted[1:]:
                    result = f"{value}({result}, {c})"
                return result
            return f"{value}({', '.join(converted)})"

    if expr is sp.pi:
        return "pi"
    if expr is sp.E:
        return "E"

    if isinstance(expr, sp.Integer):
        return _integer_to_deap(int(expr))

    if isinstance(expr, sp.Rational) or isinstance(expr, sp.Float):
        r = sp.Rational(expr)
        return f"protected_div({_integer_to_deap(r.p)}, {_integer_to_deap(r.q)})"

    if isinstance(expr, sp.Number):
        return str(expr)

    if isinstance(expr, sp.Symbol):
        return str(expr)

    raise ValueError(f"Unsupported expression type: {type(expr)}")


def convert_to_protected_form(expr_str: str) -> str:
    """Replace standard ops with protected versions for GP compatibility.

    Uses word-boundary-aware replacement to avoid double-protecting
    (e.g. 'protected_exp(' would NOT re-match 'exp(').
    """
    import re
    # Match function names NOT preceded by 'protected_'
    replacements = {
        "exp": "protected_exp",
        "log": "protected_log",
        "ln": "protected_log",
        "sqrt": "protected_sqrt",
        "pow": "protected_pow",
        "div": "protected_div",
    }
    for old, new in replacements.items():
        expr_str = re.sub(rf'(?<!protected_)(?<!\w){re.escape(old)}\(', f'{new}(', expr_str)
    return expr_str
