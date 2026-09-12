import torch

from ampgnn.diagram_parser import parse_diagram
from ampgnn.graph_builder import (
    EDGE_FEAT_DIM,
    GRAPH_FEATURE_NAMES,
    MAX_CHANNEL_LEGS,
    NODE_FEAT_DIM,
    _EDGE_CHANNEL_OFFSET,
    _NODE_STRUCT_INDEX,
    build_base_graph,
    build_pair_graphs,
)


def _two_to_two_quark_exchange():
    return parse_diagram(
        "Vertex V_0: u(X_1), u(X_3), OffShell G(V_0), "
        "Vertex V_1: u(X_2), u(X_4), OffShell G(V_1)"
    )


def test_node_roles_interactions_and_channel_cut_features():
    built = build_base_graph(
        _two_to_two_quark_exchange(), ["u", "G"], n_initial=2, model="QCD"
    )

    assert built.data.x.shape == (6, NODE_FEAT_DIM)
    assert built.data.edge_attr.shape == (10, EDGE_FEAT_DIM)
    assert built.data.x[:2, _NODE_STRUCT_INDEX["interaction_ffv"]].tolist() == [1, 1]
    assert built.data.x[2:4, _NODE_STRUCT_INDEX["incoming_external"]].tolist() == [1, 1]
    assert built.data.x[4:6, _NODE_STRUCT_INDEX["outgoing_external"]].tolist() == [1, 1]

    channel = built.data.edge_attr[0]
    assert channel[
        _EDGE_CHANNEL_OFFSET:_EDGE_CHANNEL_OFFSET + 4
    ].tolist() == [0, 1, 0, 1]
    count_offset = _EDGE_CHANNEL_OFFSET + MAX_CHANNEL_LEGS
    assert channel[count_offset:count_offset + 3].tolist() == [2, 1, 1]


def test_graph_features_follow_the_external_permutation():
    built = build_base_graph(
        _two_to_two_quark_exchange(), ["u", "G"], n_initial=2, model="QCD"
    )
    variants = build_pair_graphs(built, limit_per_group=2)
    assert len(variants) == 2

    features = [
        dict(zip(GRAPH_FEATURE_NAMES, pair.data.graph_features[0].tolist()))
        for pair in variants
    ]
    assert features[0] == {
        "external_multiplicity": 4,
        "propagators_amplitude": 1,
        "propagators_conjugate": 1,
        "shared_channels": 1,
        "diagonal_pair": 1,
        "quartic_vertices_amplitude": 0,
        "quartic_vertices_conjugate": 0,
    }
    assert features[1]["shared_channels"] == 0
    assert features[1]["diagonal_pair"] == 1
