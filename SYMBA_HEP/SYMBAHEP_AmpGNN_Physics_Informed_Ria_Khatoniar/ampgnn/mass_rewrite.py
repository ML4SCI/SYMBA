from __future__ import annotations

import re
from typing import Dict, List, Literal, Tuple

from .diagram_parser import BaseDiagram

MassRewriteMode = Literal["species", "leg_sets"]

_MASS_RE = re.compile(r"(?<![A-Za-z0-9_])m_([A-Za-z][A-Za-z0-9_]*)")


def _slot_map_from_diagram(bd: BaseDiagram) -> Dict[str, str]:
    slot: Dict[str, str] = {}

    first_xid: Dict[str, int] = {}
    for xid, ext in bd.externals.items():
        sp = ext.particle
        if sp not in first_xid or xid < first_xid[sp]:
            first_xid[sp] = xid

    externals_sorted = sorted(first_xid.items(), key=lambda kv: (kv[1], kv[0]))
    for i, (sp, _) in enumerate(externals_sorted, start=1):
        slot[sp] = f"M{i}"

    internals_sorted = sorted(bd.offshell_by_type.keys())
    next_int = 1
    for sp in internals_sorted:
        if sp in slot:
            continue
        slot[sp] = f"Mi{next_int}"
        next_int += 1

    return slot


def _letter_encode_xid(xid: int) -> str:
    if xid < 1:
        return "?"
    s = ""
    x = xid - 1
    while True:
        s = chr(ord("A") + (x % 26)) + s
        x = x // 26 - 1
        if x < 0:
            break
    return s


def _leg_set_slot_map_from_diagram(bd: BaseDiagram) -> Dict[str, str]:
    slot: Dict[str, str] = {}

    species_to_legs: Dict[str, List[int]] = {}
    for xid, ext in bd.externals.items():
        species_to_legs.setdefault(ext.particle, []).append(int(xid))

    for sp in sorted(species_to_legs.keys()):
        xids_sorted = sorted(species_to_legs[sp])
        letters = "".join(_letter_encode_xid(x) for x in xids_sorted)
        slot[sp] = f"M{letters}"

    next_int = 1
    for sp in sorted(bd.offshell_by_type.keys()):
        if sp in slot:
            continue
        slot[sp] = f"Mi{_letter_encode_xid(next_int)}"
        next_int += 1

    return slot


def canonicalize_masses(
    expr: str, bd: BaseDiagram, mode: MassRewriteMode = "species"
) -> Tuple[str, Dict[str, str]]:
    if mode == "species":
        diagram_slots = _slot_map_from_diagram(bd)
        fallback_is_letter = False
    elif mode == "leg_sets":
        diagram_slots = _leg_set_slot_map_from_diagram(bd)
        fallback_is_letter = True
    else:
        raise ValueError(f"Unknown mass rewrite mode: {mode!r}")

    full_slot_map: Dict[str, str] = dict(diagram_slots)

    fallback_order: List[str] = []
    for match in _MASS_RE.finditer(expr):
        sp = match.group(1)
        if sp in full_slot_map:
            continue
        if sp in fallback_order:
            continue
        fallback_order.append(sp)
    for i, sp in enumerate(fallback_order, start=1):
        if fallback_is_letter:
            full_slot_map[sp] = f"Mx{_letter_encode_xid(i)}"
        else:
            full_slot_map[sp] = f"Mx{i}"

    def _repl(m: re.Match) -> str:
        sp = m.group(1)
        return full_slot_map[sp]

    return _MASS_RE.sub(_repl, expr), full_slot_map


def decanonicalize_masses(expr: str, slot_map: Dict[str, str]) -> str:
    inv = {slot: sp for sp, slot in slot_map.items()}


    slots_sorted = sorted(inv.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(s) for s in slots_sorted) + r")(?![A-Za-z0-9_])"
    )

    def _repl(m: re.Match) -> str:
        return f"m_{inv[m.group(1)]}"

    return pattern.sub(_repl, expr)
