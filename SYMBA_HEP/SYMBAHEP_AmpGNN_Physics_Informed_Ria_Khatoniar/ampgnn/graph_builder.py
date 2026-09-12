from dataclasses import dataclass
from collections import Counter, deque
from typing import Dict, List, Optional, Sequence, Tuple
import torch
from torch_geometric.data import Data
from .pair_rules import PairRules, is_valid_pair_graph
from .particle_properties import PHYS_VEC_DIM, phys_props, phys_vec


_NODE_STRUCT_FEATURES = (
    "incoming_external",
    "outgoing_external",
    "vertex_valence",
    "external_attachments",
    "internal_attachments",
    "interaction_ffv",
    "interaction_vvv",
    "interaction_vvvv",
    "has_fermion",
    "has_vector",
)
_NODE_STRUCT_OFFSET = 1 + PHYS_VEC_DIM + 1
NODE_FEAT_DIM = _NODE_STRUCT_OFFSET + len(_NODE_STRUCT_FEATURES)
_NODE_STRUCT_INDEX = {
    name: _NODE_STRUCT_OFFSET + i for i, name in enumerate(_NODE_STRUCT_FEATURES)
}


_EDGE_ROLE = ("prop", "attach", "cross")
MAX_CHANNEL_LEGS = 16
_EDGE_CHANNEL_OFFSET = len(_EDGE_ROLE) + PHYS_VEC_DIM
EDGE_FEAT_DIM = _EDGE_CHANNEL_OFFSET + MAX_CHANNEL_LEGS + 3


GRAPH_FEATURE_NAMES = (
    "external_multiplicity",
    "propagators_amplitude",
    "propagators_conjugate",
    "shared_channels",
    "diagonal_pair",
    "quartic_vertices_amplitude",
    "quartic_vertices_conjugate",
)
GRAPH_FEAT_DIM = len(GRAPH_FEATURE_NAMES)


def build_particle_maps(particles: List[str]):
    return {p: i for i, p in enumerate(particles)}


def _vertex_bag(local_tokens, model: Optional[str] = None):
    vec = torch.zeros(NODE_FEAT_DIM, dtype=torch.float32)
    antip_count = 0.0
    fermions = 0
    vectors = 0
    for base, antip, _role, _tgt in local_tokens:
        vec[1:1 + PHYS_VEC_DIM] += phys_vec(base, antiparticle=antip, model=model)
        if antip:
            antip_count += 1.0
        props = phys_props(base, model=model)
        if props is not None and abs(float(props.spin) - 0.5) < 1e-6:
            fermions += 1
        if props is not None and abs(float(props.spin) - 1.0) < 1e-6:
            vectors += 1
    vec[_NODE_STRUCT_OFFSET - 1] = antip_count
    valence = len(local_tokens)
    external_count = sum(role == "external" for _, _, role, _ in local_tokens)
    internal_count = sum(role == "offshell" for _, _, role, _ in local_tokens)
    vec[_NODE_STRUCT_INDEX["vertex_valence"]] = float(valence)
    vec[_NODE_STRUCT_INDEX["external_attachments"]] = float(external_count)
    vec[_NODE_STRUCT_INDEX["internal_attachments"]] = float(internal_count)
    vec[_NODE_STRUCT_INDEX["interaction_ffv"]] = float(
        valence == 3 and fermions == 2 and vectors == 1
    )
    vec[_NODE_STRUCT_INDEX["interaction_vvv"]] = float(
        valence == 3 and vectors == 3
    )
    vec[_NODE_STRUCT_INDEX["interaction_vvvv"]] = float(
        valence == 4 and vectors == 4
    )
    vec[_NODE_STRUCT_INDEX["has_fermion"]] = float(fermions > 0)
    vec[_NODE_STRUCT_INDEX["has_vector"]] = float(vectors > 0)
    return vec


def _external_feat(
    particle: str,
    antiparticle: bool,
    *,
    incoming: bool,
    model: Optional[str] = None,
):
    v = torch.zeros(NODE_FEAT_DIM, dtype=torch.float32)
    v[0] = 1.0
    v[1:1 + PHYS_VEC_DIM] = phys_vec(particle, antiparticle=antiparticle, model=model)
    v[_NODE_STRUCT_OFFSET - 1] = 1.0 if antiparticle else 0.0
    role = "incoming_external" if incoming else "outgoing_external"
    v[_NODE_STRUCT_INDEX[role]] = 1.0
    return v


def _edge_feat(
    role: str,
    particle: Optional[str],
    model: Optional[str] = None,
    channel_cut: Optional[Sequence[int]] = None,
    n_initial: int = 2,
) -> torch.Tensor:
    v = torch.zeros(EDGE_FEAT_DIM, dtype=torch.float32)
    v[_EDGE_ROLE.index(role)] = 1.0
    if role != "cross" and particle is not None:
        phys_start = len(_EDGE_ROLE)
        v[phys_start:phys_start + PHYS_VEC_DIM] = phys_vec(
            particle, antiparticle=False, model=model
        )
    if role == "prop" and channel_cut is not None:
        if len(channel_cut) > MAX_CHANNEL_LEGS:
            raise ValueError(
                f"channel cut has {len(channel_cut)} external legs; "
                f"MAX_CHANNEL_LEGS={MAX_CHANNEL_LEGS}"
            )
        cut = torch.as_tensor(channel_cut, dtype=torch.float32)
        v[_EDGE_CHANNEL_OFFSET:_EDGE_CHANNEL_OFFSET + len(channel_cut)] = cut
        count_offset = _EDGE_CHANNEL_OFFSET + MAX_CHANNEL_LEGS
        v[count_offset] = cut.sum()
        v[count_offset + 1] = cut[:n_initial].sum()
        v[count_offset + 2] = cut[n_initial:].sum()
    return v


def _canonical_cut(bits: Sequence[int]) -> Tuple[int, ...]:
    direct = tuple(int(v) for v in bits)
    complement = tuple(1 - v for v in direct)
    return min(direct, complement)


def _propagator_channel_cuts(bd, xids: Sequence[int]) -> List[Tuple[int, ...]]:
    """Return the canonical external-leg partition induced by each bridge propagator."""
    if len(xids) > MAX_CHANNEL_LEGS:
        raise ValueError(
            f"diagram has {len(xids)} external legs; MAX_CHANNEL_LEGS={MAX_CHANNEL_LEGS}"
        )
    cuts: List[Tuple[int, ...]] = []
    vertices = list(bd.vertices.keys())
    for removed, (left, right, _particle) in enumerate(bd.internal_edges):
        adjacency = {v: [] for v in vertices}
        for edge_index, (u, v, _ptype) in enumerate(bd.internal_edges):
            if edge_index == removed:
                continue
            adjacency[u].append(v)
            adjacency[v].append(u)

        reachable = {left}
        queue = deque([left])
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if v not in reachable:
                    reachable.add(v)
                    queue.append(v)

        if right in reachable:
            cuts.append(tuple(0 for _ in xids))
            continue
        bits = [int(bd.externals[xid].at_vertex in reachable) for xid in xids]
        cuts.append(_canonical_cut(bits))
    return cuts


def build_edge_type_map(particles: List[str]):
    return {"prop": 0, "attach": 1, "cross": 2}

@dataclass
class BuiltBase:
    data: Data
    vid_to_nid: Dict[int,int]
    xid_to_nid: Dict[int,int]
    x_to_vid: Dict[int,int]
    outgoing_xids: List[int]
    incoming_xids: List[int]
    particle_to_id: Dict[str,int]
    edge_type_to_id: Dict[str,int]
    xinfo: Dict[int, tuple]
    internal_edges: List[Tuple[int, int, str]]
    channel_cuts: List[Tuple[int, ...]]
    quartic_vertices: int

def build_base_graph(bd, particles: List[str], n_initial: int = 2,
                     model: Optional[str] = None) -> BuiltBase:
    particle_to_id = build_particle_maps(particles)
    edge_type_to_id = build_edge_type_map(particles)

    vids = sorted(bd.vertices.keys())
    xids = sorted(bd.externals.keys())
    vid_to_nid = {v: i for i, v in enumerate(vids)}
    xid_to_nid = {x: len(vids) + i for i, x in enumerate(xids)}
    incoming_xids = xids[:n_initial]
    outgoing_xids = xids[n_initial:]

    x_list = []
    for v in vids:
        local = bd.vertices[v].attachments
        x_list.append(_vertex_bag(local, model=model))
    for xid in xids:
        ext = bd.externals[xid]
        x_list.append(
            _external_feat(
                ext.particle,
                ext.antiparticle,
                incoming=xid in incoming_xids,
                model=model,
            )
        )
    x_feat = torch.stack(x_list, dim=0)

    edge_src, edge_dst, edge_vecs = [], [], []
    channel_cuts = _propagator_channel_cuts(bd, xids)

    for edge_number, (u, v, ptype) in enumerate(bd.internal_edges):
        iu, iv = vid_to_nid[u], vid_to_nid[v]
        vec = _edge_feat(
            "prop",
            ptype,
            model=model,
            channel_cut=channel_cuts[edge_number],
            n_initial=n_initial,
        )
        edge_src += [iu, iv]
        edge_dst += [iv, iu]
        edge_vecs += [vec, vec]

    for xid in xids:
        ext = bd.externals[xid]
        iv = vid_to_nid[ext.at_vertex]
        ix = xid_to_nid[xid]
        vec = _edge_feat("attach", ext.particle, model=model)
        edge_src += [iv, ix]
        edge_dst += [ix, iv]
        edge_vecs += [vec, vec]

    edge_index = (
        torch.tensor([edge_src, edge_dst], dtype=torch.long)
        if edge_src else torch.empty((2, 0), dtype=torch.long)
    )
    if edge_vecs:
        edge_attr = torch.stack(edge_vecs, dim=0)
    else:
        edge_attr = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float32)

    x_to_vid = {xid: bd.externals[xid].at_vertex for xid in xids}

    data = Data(x=x_feat, edge_index=edge_index, edge_attr=edge_attr)
    data.num_vertices = len(vids)
    data.num_externals = len(xids)
    data.vids = torch.tensor(vids, dtype=torch.long)
    data.xids = torch.tensor(xids, dtype=torch.long)
    xinfo = {}
    for x in xids:
        ext = bd.externals[x]
        xinfo[x] = (ext.particle, ext.antiparticle)
    quartic_vertices = sum(len(bd.vertices[v].attachments) == 4 for v in vids)
    return BuiltBase(data=data, vid_to_nid=vid_to_nid, xid_to_nid=xid_to_nid,
                     x_to_vid=x_to_vid, outgoing_xids=outgoing_xids,
                     incoming_xids=incoming_xids, particle_to_id=particle_to_id,
                     edge_type_to_id=edge_type_to_id, xinfo=xinfo,
                     internal_edges=list(bd.internal_edges),
                     channel_cuts=channel_cuts, quartic_vertices=quartic_vertices)

@dataclass
class PairGraph:
    data: Data
    mapping: Dict[int,int]

def _group_outgoing(built: BuiltBase, rules: PairRules):
    if not rules.group_by_particle:
        return {("__ALL__", False): list(built.outgoing_xids)}


    group = {}
    for x_id in built.outgoing_xids:
        particle, antip = built.xinfo.get(x_id, ("UNK", False))
        key = rules.normalize(particle, antip)
        group.setdefault(key, []).append(x_id)
    return group

def _permutations_for_groups(grouped, limit_per_group: int = 999):
    per_group_maps = []
    for key, xs in grouped.items():
        xs_sorted = list(sorted(xs))
        n = len(xs_sorted)
        local = []
        if n <= 1:
            local.append({i:i for i in xs_sorted})
        else:
            if limit_per_group <= 1:
                local.append({i:i for i in xs_sorted})
            elif limit_per_group == 2:
                identity = {i:i for i in xs_sorted}
                local.append(identity)
                swapped = dict(identity)
                a, b = xs_sorted[-2], xs_sorted[-1]
                swapped[a], swapped[b] = b, a
                local.append(swapped)
            else:
                import itertools
                for perm in itertools.permutations(xs_sorted, n):
                    local.append({xs_sorted[i]: perm[i] for i in range(n)})
        per_group_maps.append(local)
    perms = []
    import itertools
    for prod in itertools.product(*per_group_maps):
        merged = {}
        for d in prod:
            merged.update(d)
        perms.append(merged)
    if not perms:
        perms = [dict()]
    return perms


def _aligned_channel_signatures(
    built: BuiltBase,
    source_xids: Sequence[int],
    mapping: Dict[int, int],
) -> Counter:
    target_positions = {xid: i for i, xid in enumerate(built.xid_to_nid.keys())}
    signatures = []
    for (_, _, particle), bits in zip(built.internal_edges, built.channel_cuts):
        if not any(bits):
            continue
        aligned = [bits[target_positions[mapping.get(xid, xid)]] for xid in source_xids]
        signatures.append((particle, _canonical_cut(aligned)))
    return Counter(signatures)


def _graph_features(bi: BuiltBase, bj: BuiltBase, mapping: Dict[int, int]) -> torch.Tensor:
    source_xids = list(bi.xid_to_nid.keys())
    sig_i = Counter(
        (particle, bits)
        for (_, _, particle), bits in zip(bi.internal_edges, bi.channel_cuts)
        if any(bits)
    )
    sig_j = _aligned_channel_signatures(bj, source_xids, mapping)
    shared_channels = sum((sig_i & sig_j).values())
    values = [
        len(source_xids),
        len(bi.internal_edges),
        len(bj.internal_edges),
        shared_channels,
        int(bj is bi),
        bi.quartic_vertices,
        bj.quartic_vertices,
    ]
    return torch.tensor([values], dtype=torch.float32)

def build_pair_graphs(built: BuiltBase, built_conj: Optional[BuiltBase] = None,
                      limit_per_group: int = 999, rules: PairRules | None = None) -> List[PairGraph]:
    bi = built
    bj = built_conj if built_conj is not None else built

    base_i, base_j = bi.data, bj.data
    N_i = base_i.num_vertices + base_i.num_externals

    if bj is not bi:
        if sorted(bi.xid_to_nid.keys()) != sorted(bj.xid_to_nid.keys()):
            return []
        if any(bj.xinfo.get(x_id) != info for x_id, info in bi.xinfo.items()):
            return []

    rules = rules or PairRules()
    group = _group_outgoing(bi, rules)
    perms = _permutations_for_groups(group, limit_per_group=limit_per_group)
    pair_graphs: List[PairGraph] = []

    for P in perms:
        ok = is_valid_pair_graph(bi, P, rules)
        if not ok:
            continue
        x = torch.cat([base_i.x, base_j.x], dim=0)
        side = torch.zeros((x.size(0), 1), dtype=torch.float32)
        side[N_i:, 0] = 1.0
        x = torch.cat([x, side], dim=1)

        e1 = base_i.edge_index.clone()
        ea1 = base_i.edge_attr.clone()
        e2 = base_j.edge_index.clone() + N_i if base_j.edge_index.numel() > 0 else base_j.edge_index.clone()
        ea2 = base_j.edge_attr.clone()
        edge_index = torch.cat([e1, e2], dim=1) if e1.numel() > 0 else e2
        edge_attr = torch.cat([ea1, ea2], dim=0) if ea1.numel() > 0 else ea2


        cross_src, cross_dst = [], []
        for x_id in bi.incoming_xids + bi.outgoing_xids:
            xi = bi.xid_to_nid[x_id]
            target_xid = x_id if x_id in bi.incoming_xids else P.get(x_id, x_id)
            xj = bj.xid_to_nid[target_xid] + N_i
            cross_src += [xi, xj]
            cross_dst += [xj, xi]
        if cross_src:
            cross_edge_index = torch.tensor([cross_src, cross_dst], dtype=torch.long)
            cross_vec = _edge_feat("cross", None)
            cross_edge_attr = cross_vec.unsqueeze(0).expand(len(cross_src), -1).clone()
            edge_index = torch.cat([edge_index, cross_edge_index], dim=1) if edge_index.numel() > 0 else cross_edge_index
            edge_attr = torch.cat([edge_attr, cross_edge_attr], dim=0) if edge_attr.numel() > 0 else cross_edge_attr

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.N_base = N_i
        data.graph_features = _graph_features(bi, bj, P)
        pairs = torch.tensor(
            [[src, P.get(src, src)] for src in bi.incoming_xids + bi.outgoing_xids],
            dtype=torch.long
        )
        data.mapping_pairs = pairs
        pair_graphs.append(PairGraph(data=data, mapping=pairs))
    return pair_graphs
