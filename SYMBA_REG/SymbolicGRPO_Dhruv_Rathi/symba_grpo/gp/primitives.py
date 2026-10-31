"""DEAP primitive set and protected operations for GP.

Ported from Aryamaan's custom_gp.py and Samyak's gp.py.
"""

import math
import operator

from deap import gp

OVERFLOW_LIMIT = 1e7


def protected_div(x1, x2):
    try:
        return x1 / x2
    except (ZeroDivisionError, FloatingPointError):
        return OVERFLOW_LIMIT


def protected_exp(x1):
    try:
        if x1 > 100:
            return OVERFLOW_LIMIT
        if x1 < -100:
            return 0.0
        return math.exp(x1)
    except (OverflowError, ValueError):
        return OVERFLOW_LIMIT


def protected_log(x1):
    try:
        return math.log(abs(x1)) if x1 != 0 else -OVERFLOW_LIMIT
    except (ValueError, OverflowError):
        return -OVERFLOW_LIMIT


def protected_sqrt(x1):
    try:
        return math.sqrt(abs(x1))
    except (ValueError, OverflowError):
        return 0.0


def protected_pow(x1, x2):
    try:
        result = math.pow(abs(x1), x2)
        return result if abs(result) < OVERFLOW_LIMIT else OVERFLOW_LIMIT
    except (OverflowError, ValueError):
        return OVERFLOW_LIMIT


def make_primitive_set(num_vars: int) -> gp.PrimitiveSet:
    """Create the frozen DEAP primitive set."""
    pset = gp.PrimitiveSet("MAIN", num_vars)

    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(protected_pow, 2)
    pset.addPrimitive(protected_exp, 1)
    pset.addPrimitive(protected_log, 1)
    pset.addPrimitive(protected_sqrt, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.tan, 1)
    pset.addPrimitive(math.tanh, 1)
    pset.addPrimitive(abs, 1)

    for i in range(1, 5):
        pset.addTerminal(int(i))
    pset.addTerminal(math.pi, name="pi")
    pset.addTerminal(math.e, name="E")

    rename = {f"ARG{i}": f"s_{i+1}" for i in range(num_vars)}
    pset.renameArguments(**rename)

    return pset
