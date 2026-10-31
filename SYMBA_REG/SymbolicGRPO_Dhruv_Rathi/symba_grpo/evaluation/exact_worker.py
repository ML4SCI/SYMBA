"""Resource-limited exact-equivalence worker; intentionally project-import free."""

from __future__ import annotations

import json
import resource
import sys

MEMORY_LIMIT_BYTES = 1_073_741_824
CPU_LIMIT_SECONDS = 6

# Limit the fresh interpreter before importing or invoking the symbolic stack.
resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT_SECONDS, CPU_LIMIT_SECONDS))

import sympy as sp  # noqa: E402


LOCALS = {
    "E": sp.E, "pi": sp.pi, "exp": sp.exp, "ln": sp.log, "log": sp.log,
    "sqrt": sp.sqrt, "abs": sp.Abs, "Abs": sp.Abs,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "tanh": sp.tanh,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
}

PREFIX_ARITY = {
    "add": 2, "mul": 2, "pow": 2,
    "exp": 1, "ln": 1, "abs": 1, "sign": 1,
    "sin": 1, "cos": 1, "tan": 1, "cot": 1, "sec": 1, "csc": 1,
    "asin": 1, "acos": 1, "atan": 1, "acot": 1, "asec": 1, "acsc": 1,
    "sinh": 1, "cosh": 1, "tanh": 1, "coth": 1, "sech": 1, "csch": 1,
    "asinh": 1, "acosh": 1, "atanh": 1, "acoth": 1, "asech": 1, "acsch": 1,
}
PREFIX_FUNCTION = {
    "add": sp.Add, "mul": sp.Mul, "pow": sp.Pow,
    "exp": sp.exp, "ln": sp.log, "abs": sp.Abs, "sign": sp.sign,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot,
    "sec": sp.sec, "csc": sp.csc, "asin": sp.asin, "acos": sp.acos,
    "atan": sp.atan, "acot": sp.acot, "asec": sp.asec, "acsc": sp.acsc,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh, "coth": sp.coth,
    "sech": sp.sech, "csch": sp.csch, "asinh": sp.asinh, "acosh": sp.acosh,
    "atanh": sp.atanh, "acoth": sp.acoth, "asech": sp.asech, "acsch": sp.acsch,
}


def _prefix_expression(tokens: list) -> sp.Expr:
    def at(index: int):
        if index >= len(tokens):
            raise ValueError("incomplete prefix record")
        token = tokens[index]
        if token in PREFIX_ARITY:
            args, nxt = [], index + 1
            for _ in range(PREFIX_ARITY[token]):
                argument, nxt = at(nxt)
                args.append(argument)
            return PREFIX_FUNCTION[token](*args, evaluate=False), nxt
        if token in ("s+", "s-"):
            nxt, digits = index + 1, []
            while nxt < len(tokens) and str(tokens[nxt]) in set("0123456789"):
                digits.append(str(tokens[nxt])); nxt += 1
            if not digits:
                raise ValueError("integer without digits")
            return sp.Integer(("-" if token == "s-" else "") + "".join(digits)), nxt
        if token == "E":
            return sp.E, index + 1
        if token == "pi":
            return sp.pi, index + 1
        if isinstance(token, str) and token.startswith("s_") and token[2:].isdigit():
            return sp.Symbol(token), index + 1
        raise ValueError("unsupported prefix token")

    expression, consumed = at(0)
    if consumed != len(tokens):
        raise ValueError("trailing prefix tokens")
    return expression


def _gp_expression(nodes: list, num_vars: int) -> sp.Expr:
    """Build a GP expression from a typed structural record without evaluation."""
    def at(index: int):
        if index >= len(nodes):
            raise ValueError("incomplete GP record")
        node = nodes[index]
        if not isinstance(node, list) or len(node) != 3:
            raise ValueError("malformed GP record")
        kind, value, arity = node
        if kind == "t":
            if arity != 0:
                raise ValueError("terminal arity")
            if isinstance(value, str) and value.startswith("s_") and value[2:].isdigit():
                if not 1 <= int(value[2:]) <= num_vars:
                    raise ValueError("undeclared variable")
                return sp.Symbol(value), index + 1
            if value == "pi":
                return sp.pi, index + 1
            if value == "E":
                return sp.E, index + 1
            if type(value) is int and 1 <= value <= 4:
                return sp.Integer(value), index + 1
            raise ValueError("unsupported terminal")
        expected = {
            "add": 2, "sub": 2, "mul": 2, "protected_div": 2,
            "protected_pow": 2, "protected_exp": 1, "protected_log": 1,
            "protected_sqrt": 1, "sin": 1, "cos": 1, "tan": 1,
            "tanh": 1, "abs": 1,
        }.get(value)
        if kind != "p" or expected is None or arity != expected:
            raise ValueError("unsupported primitive")
        args, nxt = [], index + 1
        for _ in range(arity):
            argument, nxt = at(nxt)
            args.append(argument)
        if value == "add":
            expression = sp.Add(*args, evaluate=False)
        elif value == "sub":
            expression = sp.Add(
                args[0], sp.Mul(sp.Integer(-1), args[1], evaluate=False), evaluate=False,
            )
        elif value == "mul":
            expression = sp.Mul(*args, evaluate=False)
        elif value == "protected_div":
            expression = sp.Mul(
                args[0], sp.Pow(args[1], sp.Integer(-1), evaluate=False), evaluate=False,
            )
        elif value == "protected_pow":
            expression = sp.Pow(*args, evaluate=False)
        elif value == "protected_sqrt":
            expression = sp.Pow(args[0], sp.Rational(1, 2), evaluate=False)
        else:
            function = {
                "protected_exp": sp.exp, "protected_log": sp.log,
                "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                "tanh": sp.tanh, "abs": sp.Abs,
            }[value]
            expression = function(args[0], evaluate=False)
        return expression, nxt

    expression, consumed = at(0)
    if consumed != len(nodes):
        raise ValueError("trailing GP nodes")
    return expression


def _is_exactly_equal(pred_expr: sp.Expr, true_expr: sp.Expr) -> bool:
    """Return symbolic equality without discarding an already-proven zero.

    ``Expr.is_zero`` can be definitively true even when SymPy also reports the
    unevaluated difference as numeric.  A structural ``diff == 0`` fallback
    must therefore only run after the definitive result has been checked.
    """
    diff = pred_expr - true_expr
    exact = diff.is_zero is True
    if not exact:
        exact = bool(sp.simplify(diff) == 0)
    return exact


def main() -> None:
    def emit(status: str) -> None:
        print(json.dumps({"exact": status == "exact", "status": status}))

    try:
        payload = json.load(sys.stdin)
        prediction = payload["prediction"]
        if prediction.get("version") != 1:
            raise ValueError("unknown prediction IR version")
        if prediction["source"] == "gp":
            pred_expr = _gp_expression(prediction["nodes"], int(prediction["num_vars"]))
        elif prediction["source"] == "beam":
            pred_expr = _prefix_expression(prediction["tokens"])
        else:
            raise ValueError("unknown prediction kind")
        true_expr = sp.sympify(payload["truth"], locals=LOCALS, evaluate=False)
    except MemoryError:
        emit("resource_limit")
        return
    except Exception:
        emit("parse_failure")
        return

    try:
        exact = _is_exactly_equal(pred_expr, true_expr)
        emit("exact" if exact else "nonmatch")
    except MemoryError:
        emit("resource_limit")
    except BaseException:
        emit("worker_failure")


if __name__ == "__main__":
    main()
