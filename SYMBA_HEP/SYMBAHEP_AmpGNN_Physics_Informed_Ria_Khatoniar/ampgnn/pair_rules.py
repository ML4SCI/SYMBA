from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

OutgoingKey = Tuple[str, bool]

SELF_CONJ_DEFAULT = {
    "A", "g", "Z", "h", "G", "gamma", "photon", "gluon", "H", "Z0"
}

@dataclass(frozen=True)
class PairRules:
    self_conjugate: Iterable[str] = tuple(SELF_CONJ_DEFAULT)

    enforce_particle_match: bool = True
    enforce_bijection: bool = True
    group_by_particle: bool = True

    def normalize(self, particle: str, anti: bool) -> OutgoingKey:
        if particle in self.self_conjugate:
            return (particle, False)
        return (particle, bool(anti))

    def allowed(self, left: OutgoingKey, right: OutgoingKey) -> bool:
        if not self.enforce_particle_match:
            return True
        return left == right


def is_valid_pair_graph(built, mapping: Dict[int, int], rules: PairRules | None = None) -> bool:
    rules = rules or PairRules()


    rng = [mapping.get(x, None) for x in built.outgoing_xids]
    if any(r is None for r in rng):
        return False


    if rules.enforce_bijection and (len(set(rng)) != len(rng)):
        return False


    for xi in built.incoming_xids:
        if mapping.get(xi, xi) != xi:
            return False


    for src in built.outgoing_xids:
        dst = mapping[src]
        lp, la = built.xinfo[src]
        rp, ra = built.xinfo[dst]
        if not rules.allowed(rules.normalize(lp, la), rules.normalize(rp, ra)):
            return False

    return True
