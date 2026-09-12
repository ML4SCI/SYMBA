"""Physics-blind graph construction.

Same diagram topology as `ampgnn.graph_builder` (vertices, propagators, external
attachments), but every physics prior is removed from the features:

* particle properties (spin, charge, T3, hypercharge, colour, generation,
  fermion number, self-conjugacy, chirality) are replaced by a one-hot over the
  raw particle symbol, with separate slots for particle vs. antiparticle;
* the amplitude<->conjugate cross edges, which encode the interference leg
  pairing, are dropped;
* only the identity leg permutation is emitted, so there is no augmentation over
  the exchange symmetry of identical outgoing particles.

The baseline therefore reads the graph structure and opaque labels, and has to
learn any particle relationships from the data.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch_geometric.data import Data

from ..graph_builder import BuiltBase, PairGraph, build_edge_type_map, build_particle_maps
from ..pair_rules import PairRules, is_valid_pair_graph

EDGE_ROLES = ("prop", "attach")


class BlindFeaturizer:
    """Symbol-identity node/edge features, with no physics table lookups."""

    def __init__(self, particles: List[str]):
        symbols = sorted(set(particles))
        self._sym_id = {p: i for i, p in enumerate(symbols)}
        self._n_sym = len(symbols) + 1  # + unknown symbol
        # [external flag][particle one-hots][antiparticle one-hots][antiparticle count]
        self.node_dim = 1 + 2 * self._n_sym + 1
        self.edge_dim = len(EDGE_ROLES) + self._n_sym

    def _slot(self, particle: Optional[str]) -> int:
        return self._sym_id.get(particle, self._n_sym - 1)

    def vertex(self, attachments) -> torch.Tensor:
        v = torch.zeros(self.node_dim, dtype=torch.float32)
        antip_count = 0.0
        for base, antip, _role, _target in attachments:
            block = 1 + self._n_sym if antip else 1
            v[block + self._slot(base)] += 1.0
            if antip:
                antip_count += 1.0
        v[-1] = antip_count
        return v

    def external(self, particle: str, antiparticle: bool) -> torch.Tensor:
        v = torch.zeros(self.node_dim, dtype=torch.float32)
        v[0] = 1.0
        block = 1 + self._n_sym if antiparticle else 1
        v[block + self._slot(particle)] = 1.0
        v[-1] = 1.0 if antiparticle else 0.0
        return v

    def edge(self, role: str, particle: Optional[str]) -> torch.Tensor:
        v = torch.zeros(self.edge_dim, dtype=torch.float32)
        v[EDGE_ROLES.index(role)] = 1.0
        if particle is not None:
            v[len(EDGE_ROLES) + self._slot(particle)] = 1.0
        return v


def build_blind_base_graph(bd, particles: List[str], n_initial: int = 2) -> BuiltBase:
    feat = BlindFeaturizer(particles)

    vids = sorted(bd.vertices.keys())
    xids = sorted(bd.externals.keys())
    vid_to_nid = {v: i for i, v in enumerate(vids)}
    xid_to_nid = {x: len(vids) + i for i, x in enumerate(xids)}

    x_list = [feat.vertex(bd.vertices[v].attachments) for v in vids]
    for xid in xids:
        ext = bd.externals[xid]
        x_list.append(feat.external(ext.particle, ext.antiparticle))
    x_feat = torch.stack(x_list, dim=0)

    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_vecs: List[torch.Tensor] = []

    for u, v, ptype in bd.internal_edges:
        iu, iv = vid_to_nid[u], vid_to_nid[v]
        vec = feat.edge("prop", ptype)
        edge_src += [iu, iv]
        edge_dst += [iv, iu]
        edge_vecs += [vec, vec]

    for xid in xids:
        ext = bd.externals[xid]
        iv = vid_to_nid[ext.at_vertex]
        ix = xid_to_nid[xid]
        vec = feat.edge("attach", ext.particle)
        edge_src += [iv, ix]
        edge_dst += [ix, iv]
        edge_vecs += [vec, vec]

    edge_index = (
        torch.tensor([edge_src, edge_dst], dtype=torch.long)
        if edge_src else torch.empty((2, 0), dtype=torch.long)
    )
    edge_attr = (
        torch.stack(edge_vecs, dim=0) if edge_vecs
        else torch.zeros((0, feat.edge_dim), dtype=torch.float32)
    )

    data = Data(x=x_feat, edge_index=edge_index, edge_attr=edge_attr)
    data.num_vertices = len(vids)
    data.num_externals = len(xids)
    data.vids = torch.tensor(vids, dtype=torch.long)
    data.xids = torch.tensor(xids, dtype=torch.long)

    xinfo = {x: (bd.externals[x].particle, bd.externals[x].antiparticle) for x in xids}

    return BuiltBase(
        data=data,
        vid_to_nid=vid_to_nid,
        xid_to_nid=xid_to_nid,
        x_to_vid={xid: bd.externals[xid].at_vertex for xid in xids},
        outgoing_xids=xids[n_initial:],
        incoming_xids=xids[:n_initial],
        particle_to_id=build_particle_maps(particles),
        edge_type_to_id=build_edge_type_map(particles),
        xinfo=xinfo,
    )


def build_blind_pair_graphs(built: BuiltBase, built_conj: Optional[BuiltBase] = None,
                            rules: Optional[PairRules] = None) -> List[PairGraph]:
    """Concatenate amplitude and conjugate diagrams, identity pairing only.

    Returns at most one graph: no cross edges and no permutation augmentation,
    so the two sides are separate components distinguished by a side flag.
    """
    bi = built
    bj = built_conj if built_conj is not None else built
    rules = rules or PairRules()

    base_i, base_j = bi.data, bj.data
    N_i = base_i.num_vertices + base_i.num_externals

    if bj is not bi:
        if sorted(bi.xid_to_nid.keys()) != sorted(bj.xid_to_nid.keys()):
            return []
        if any(bj.xinfo.get(x_id) != info for x_id, info in bi.xinfo.items()):
            return []

    identity: Dict[int, int] = {x: x for x in bi.incoming_xids + bi.outgoing_xids}
    if not is_valid_pair_graph(bi, identity, rules):
        return []

    x = torch.cat([base_i.x, base_j.x], dim=0)
    side = torch.zeros((x.size(0), 1), dtype=torch.float32)
    side[N_i:, 0] = 1.0
    x = torch.cat([x, side], dim=1)

    e1 = base_i.edge_index.clone()
    e2 = base_j.edge_index.clone()
    if e2.numel() > 0:
        e2 = e2 + N_i
    edge_index = torch.cat([e1, e2], dim=1) if e1.numel() > 0 else e2

    ea1 = base_i.edge_attr.clone()
    ea2 = base_j.edge_attr.clone()
    edge_attr = torch.cat([ea1, ea2], dim=0) if ea1.numel() > 0 else ea2

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.N_base = N_i
    pairs = torch.tensor([[src, src] for src in bi.incoming_xids + bi.outgoing_xids],
                         dtype=torch.long)
    data.mapping_pairs = pairs
    return [PairGraph(data=data, mapping=pairs)]
