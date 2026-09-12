import itertools
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class External:
    x_id: int
    particle: str
    antiparticle: bool
    at_vertex: int

@dataclass
class Vertex:
    v_id: int
    attachments: List[Tuple[str, bool, str, int]] = field(default_factory=list)

@dataclass
class BaseDiagram:
    vertices: Dict[int, Vertex]
    externals: Dict[int, External]
    offshell_by_type: Dict[str, List[int]]
    internal_edges: List[Tuple[int,int,str]]

VERTEX_RE = re.compile(r"Vertex\s+V_(\d+):")


def _explicit_internal_edges(stubs: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    edges: List[Tuple[int, int, str]] = []
    cnt = Counter((v, t, sp) for v, t, sp in stubs)
    for (v, t, sp) in sorted(cnt):
        k = cnt[(v, t, sp)]
        if v < t:
            k2 = cnt.get((t, v, sp), 0)
            if k2 != k:
                raise ValueError(
                    f"unreciprocated propagator stubs for {sp}: "
                    f"V_{v}->V_{t} x{k} vs V_{t}->V_{v} x{k2}"
                )
            edges.extend([(v, t, sp)] * k)
        elif v == t:
            if k % 2:
                raise ValueError(f"odd self-loop stub count for {sp} at V_{v}")
            edges.extend([(v, v, sp)] * (k // 2))
    return edges


def _perfect_matchings(items: List[int]):
    if len(items) < 2:
        yield []
        return
    a = items[0]
    seen = set()
    for k in range(1, len(items)):
        b = items[k]
        if b in seen:
            continue
        seen.add(b)
        rest = items[1:k] + items[k + 1:]
        for m in _perfect_matchings(rest):
            yield [(a, b)] + m


def _legacy_internal_edges(
    stubs_by_type: Dict[str, List[int]], vertex_ids: List[int]
) -> List[Tuple[int, int, str]]:
    per_species: List[Tuple[str, List[int]]] = []
    for sp in sorted(stubs_by_type.keys()):
        vids = sorted(stubs_by_type[sp])
        if len(vids) % 2:
            vids = vids[:-1]
        per_species.append((sp, vids))

    def connected(edges):
        if not vertex_ids:
            return True
        adj = {v: set() for v in vertex_ids}
        for a, b, _ in edges:
            adj[a].add(b)
            adj[b].add(a)
        seed = min(vertex_ids)
        seen, stack = {seed}, [seed]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return len(seen) == len(vertex_ids)

    matchings = [(sp, list(_perfect_matchings(vids))) for sp, vids in per_species]
    best_no_self = None
    budget = 4000
    for combo in itertools.product(*[m for _, m in matchings]) if matchings else iter([()]):
        budget -= 1
        if budget < 0:
            break
        edges = []
        for (sp, _), pairs in zip(matchings, combo):
            edges.extend((a, b, sp) for a, b in pairs)
        if any(a == b for a, b, _ in edges):
            continue
        if best_no_self is None:
            best_no_self = edges
        if connected(edges):
            return edges
    if best_no_self is not None:
        return best_no_self

    edges = []
    for sp, vids in per_species:
        for k in range(0, len(vids), 2):
            edges.append((vids[k], vids[k + 1], sp))
    return edges


def parse_diagram(text: str) -> BaseDiagram:
    vertices: Dict[int, Vertex] = {}
    externals: Dict[int, External] = {}
    offshell_by_type: Dict[str, List[int]] = {}
    for m in VERTEX_RE.finditer(text):
        v_id = int(m.group(1))
        start = m.end()
        next_m = VERTEX_RE.search(text, start)
        end = next_m.start() if next_m else len(text)
        section = text[start:end].strip().strip(',')
        vertices[v_id] = Vertex(v_id=v_id)
        if not section:
            continue
        parts = [p.strip() for p in section.split(',') if p.strip()]
        for p in parts:
            if '(' not in p or ')' not in p:
                continue
            head, arg = p.split('(', 1)
            arg = arg.rstrip(')')
            head = head.strip()
            tokens = [t for t in head.split() if t]
            if not tokens:
                continue
            base = tokens[-1]
            wrappers = set(tokens[:-1])
            antip = "AntiPart" in wrappers
            if arg.startswith('X_'):
                x_id = int(arg.split('_')[1])
                vertices[v_id].attachments.append((base, antip, "external", x_id))
                externals[x_id] = External(x_id=x_id, particle=base, antiparticle=antip, at_vertex=v_id)
            elif arg.startswith('V_'):
                v2 = int(arg.split('_')[1])
                vertices[v_id].attachments.append((base, antip, "offshell", v2))
                offshell_by_type.setdefault(base, []).append(v_id)
            else:
                pass

    stubs: List[Tuple[int, int, str]] = []
    for v_id in vertices:
        for base, _antip, role, tgt in vertices[v_id].attachments:
            if role == "offshell":
                stubs.append((v_id, tgt, base))

    all_self_referential = all(v == t for v, t, _ in stubs)
    if stubs and not all_self_referential:
        internal_edges = _explicit_internal_edges(stubs)
    else:
        internal_edges = _legacy_internal_edges(offshell_by_type, list(vertices.keys()))

    return BaseDiagram(vertices=vertices, externals=externals, offshell_by_type=offshell_by_type, internal_edges=internal_edges)
