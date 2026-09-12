import sqlite3

import torch
from torch_geometric.data import Batch, Data

from ampgnn.model import LitAmpGNN
from ampgnn.term_slot_objective import (
    collate_term_slot_targets,
    structured_slot_objective,
)
from ampgnn.term_slots import (
    TermSlotVocabulary,
    build_term_slot_target,
    term_slot_cache_key,
    validate_term_slot_target,
)


TOKENS = [
    "2", "g", "^4", "(", "s_12", "+", "M1", "^2", ")", "/", "(",
    "(", "-", "2", "s_13", "+", "reg_prop", ")", "^2", ")",
]


def _target_and_vocab():
    target = build_term_slot_target(TOKENS, "g")
    target["topology_degree"] = {
        "numerator_degree2": 2,
        "denominator_degree": 2,
    }
    vocab = TermSlotVocabulary.build([target])
    encoded, unknown = vocab.encode(target)
    assert not unknown
    return target, encoded, vocab


def test_structured_target_decomposition_and_zero():
    target, _, _ = _target_and_vocab()
    assert target["global"] == {
        "coefficient": "2",
        "coupling": "g",
        "coupling_power": 4,
    }
    assert len(target["numerator_terms"]) == 2
    assert target["denominator_factors"][0]["power"] == 2
    assert len(target["denominator_factors"][0]["terms"]) == 2

    zero = build_term_slot_target(["(", "0", ")", "/", "(", "1", ")"], "g")
    assert zero["numerator_terms"] == [{"coefficient": "0", "factors": []}]
    assert zero["denominator_factors"] == []


def test_slot_capacities_and_symbol_dimensions():
    target, _, vocab = _target_and_vocab()
    validate_term_slot_target(
        target,
        max_num_terms=2,
        max_num_factors=1,
        max_den_factors=1,
        max_den_terms=2,
        max_den_term_factors=1,
        max_exponent=4,
    )
    dimensions = vocab.symbol_degree2()
    assert dimensions[vocab.symbol_to_id["M1"]] == 1
    assert dimensions[vocab.symbol_to_id["s_12"]] == 2
    assert dimensions[vocab.symbol_to_id["reg_prop"]] == 2


def _perfect_logits(encoded, vocab, *, num_rows=(0, 1), den_row=0, term_rows=(0, 1)):
    batch, num_slots, num_factors = 1, 4, 2
    den_slots, den_terms, den_factors = 2, 3, 2
    coefficient_size = vocab.coefficient_size
    symbol_size = vocab.symbol_size
    coupling_size = vocab.coupling_size
    exponent_size = 9

    def logits(*shape):
        return torch.full(shape, -12.0)

    outputs = {
        "global_coefficient_logits": logits(batch, coefficient_size),
        "global_coupling_logits": logits(batch, coupling_size),
        "global_power_logits": logits(batch, exponent_size),
        "num_active_logits": logits(batch, num_slots, 2),
        "num_coefficient_logits": logits(batch, num_slots, coefficient_size),
        "num_factor_active_logits": logits(batch, num_slots, num_factors, 2),
        "num_symbol_logits": logits(batch, num_slots, num_factors, symbol_size),
        "num_exponent_logits": logits(batch, num_slots, num_factors, exponent_size),
        "den_active_logits": logits(batch, den_slots, 2),
        "den_power_logits": logits(batch, den_slots, exponent_size),
        "den_term_active_logits": logits(batch, den_slots, den_terms, 2),
        "den_term_coefficient_logits": logits(
            batch, den_slots, den_terms, coefficient_size
        ),
        "den_factor_active_logits": logits(
            batch, den_slots, den_terms, den_factors, 2
        ),
        "den_symbol_logits": logits(
            batch, den_slots, den_terms, den_factors, symbol_size
        ),
        "den_exponent_logits": logits(
            batch, den_slots, den_terms, den_factors, exponent_size
        ),
    }

    def select(tensor, index, value):
        tensor[index + (int(value),)] = 12.0

    select(outputs["global_coefficient_logits"], (0,), encoded["global"]["coefficient"])
    select(outputs["global_coupling_logits"], (0,), encoded["global"]["coupling"])
    select(outputs["global_power_logits"], (0,), encoded["global"]["coupling_power"])

    outputs["num_active_logits"][..., 0] = 12.0
    outputs["num_factor_active_logits"][..., 0] = 12.0
    for row, term in zip(num_rows, encoded["numerator_terms"]):
        outputs["num_active_logits"][0, row] = torch.tensor([-12.0, 12.0])
        select(outputs["num_coefficient_logits"], (0, row), term["coefficient"])
        for factor_index, factor in enumerate(term["factors"]):
            outputs["num_factor_active_logits"][0, row, factor_index] = torch.tensor(
                [-12.0, 12.0]
            )
            select(
                outputs["num_symbol_logits"],
                (0, row, factor_index),
                factor["symbol"],
            )
            select(
                outputs["num_exponent_logits"],
                (0, row, factor_index),
                factor["exponent"],
            )

    outputs["den_active_logits"][..., 0] = 12.0
    outputs["den_term_active_logits"][..., 0] = 12.0
    outputs["den_factor_active_logits"][..., 0] = 12.0
    if encoded["denominator_factors"]:
        den_target = encoded["denominator_factors"][0]
        outputs["den_active_logits"][0, den_row] = torch.tensor([-12.0, 12.0])
        select(outputs["den_power_logits"], (0, den_row), den_target["power"])
        for term_row, term in zip(term_rows, den_target["terms"]):
            outputs["den_term_active_logits"][0, den_row, term_row] = torch.tensor(
                [-12.0, 12.0]
            )
            select(
                outputs["den_term_coefficient_logits"],
                (0, den_row, term_row),
                term["coefficient"],
            )
            for factor_index, factor in enumerate(term["factors"]):
                outputs["den_factor_active_logits"][
                    0, den_row, term_row, factor_index
                ] = torch.tensor([-12.0, 12.0])
                select(
                    outputs["den_symbol_logits"],
                    (0, den_row, term_row, factor_index),
                    factor["symbol"],
                )
                select(
                    outputs["den_exponent_logits"],
                    (0, den_row, term_row, factor_index),
                    factor["exponent"],
                )
    return {key: value.requires_grad_() for key, value in outputs.items()}


def test_canonical_slot_ce_is_exact_and_degree_valid():
    _, encoded, vocab = _target_and_vocab()
    outputs = _perfect_logits(encoded, vocab)
    result = structured_slot_objective(
        outputs,
        [encoded],
        keep_loss=torch.tensor([True]),
        symbol_degree2=torch.tensor(vocab.symbol_degree2(), dtype=torch.float32),
        zero_coefficient_id=vocab.coefficient_to_id.get("0", -1),
    )
    loss = result["main_loss"] + 0.1 * result["degree_loss"]
    loss.backward()
    assert all(value.grad is not None for value in outputs.values())
    assert result["seq_acc"].item() == 1.0
    assert result["token_acc"].item() == 1.0
    assert result["degree_valid"].item() == 1.0


def test_vectorized_target_batch_matches_list_objective():
    _, encoded, vocab = _target_and_vocab()
    outputs = _perfect_logits(encoded, vocab)
    kwargs = {
        "keep_loss": torch.tensor([True]),
        "symbol_degree2": torch.tensor(vocab.symbol_degree2(), dtype=torch.float32),
        "zero_coefficient_id": vocab.coefficient_to_id.get("0", -1),
        "label_smoothing": 0.05,
    }
    list_result = structured_slot_objective(outputs, [encoded], **kwargs)
    dense = collate_term_slot_targets(
        [encoded],
        num_slots=4,
        num_factors=2,
        den_slots=2,
        den_terms=3,
        den_term_factors=2,
        keep_loss=[True],
    )
    dense_result = structured_slot_objective(outputs, dense, **kwargs)
    for key in ("main_loss", "degree_loss", "token_acc", "seq_acc", "degree_valid"):
        assert torch.equal(list_result[key], dense_result[key])
    assert list_result["token_weight"] == dense_result["token_weight"]


def test_term_slot_sql_cache_round_trip(tmp_path, monkeypatch):
    cache = tmp_path / "term_slots.sqlite"
    monkeypatch.setenv("AMPGNN_TERM_SLOT_CACHE", str(cache))
    monkeypatch.delenv("AMPGNN_TERM_SLOT_CACHE_READONLY", raising=False)
    expected = build_term_slot_target(TOKENS, "g")
    key = term_slot_cache_key(TOKENS, "g")
    with sqlite3.connect(cache) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM term_slot_cache WHERE key = ?", (key,)
        ).fetchone()[0] == 1

    monkeypatch.setenv("AMPGNN_TERM_SLOT_CACHE_READONLY", "1")
    assert build_term_slot_target(TOKENS, "g") == expected


def test_canonical_slot_ce_rejects_permuted_slots():
    _, encoded, vocab = _target_and_vocab()
    outputs = _perfect_logits(
        encoded,
        vocab,
        num_rows=(1, 0),
        den_row=1,
        term_rows=(1, 0),
    )
    result = structured_slot_objective(
        outputs,
        [encoded],
        keep_loss=torch.tensor([True]),
        symbol_degree2=torch.tensor(vocab.symbol_degree2(), dtype=torch.float32),
        zero_coefficient_id=vocab.coefficient_to_id.get("0", -1),
    )
    assert result["seq_acc"].item() == 0.0


def test_zero_target_does_not_receive_conflicting_degree_supervision():
    target = build_term_slot_target(["(", "0", ")", "/", "(", "1", ")"], "g")
    target["topology_degree"] = {"numerator_degree2": 6, "denominator_degree": 4}
    vocab = TermSlotVocabulary.build([target])
    encoded, unknown = vocab.encode(target)
    assert not unknown
    outputs = _perfect_logits(encoded, vocab, num_rows=(0,))
    result = structured_slot_objective(
        outputs,
        [encoded],
        keep_loss=torch.tensor([True]),
        symbol_degree2=torch.tensor(vocab.symbol_degree2(), dtype=torch.float32),
        zero_coefficient_id=vocab.coefficient_to_id["0"],
    )
    assert result["seq_acc"].item() == 1.0
    assert result["degree_loss"].item() == 0.0
    assert result["degree_valid"].item() == 1.0


def test_lightning_term_slot_step_is_parallel_and_differentiable():
    _, encoded, vocab = _target_and_vocab()
    graph = Data(
        x=torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
            ]
        ),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
        edge_attr=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        anchor_dists=torch.tensor(
            [[0.0, 1.0], [0.5, 1.0], [0.0, 1.0], [0.5, 1.0]]
        ),
    )
    graph_batch = Batch.from_data_list([graph])
    model = LitAmpGNN(
        in_dim=4,
        edge_dim=3,
        vocab_size=5,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        lr=1e-3,
        weight_decay=0.0,
        scheduler="none",
        warmup_steps=0,
        loss_mode="ce",
        enc_hid=16,
        enc_layers=1,
        enc_heads=4,
        d_model=16,
        dec_nhead=4,
        dec_layers=1,
        output_mode="term_slots",
        slot_coefficient_size=vocab.coefficient_size,
        slot_symbol_size=vocab.symbol_size,
        slot_coupling_size=vocab.coupling_size,
        slot_max_exponent=8,
        slot_num_slots=4,
        slot_num_factors=2,
        slot_den_slots=2,
        slot_den_terms=3,
        slot_den_term_factors=2,
        slot_symbol_degree2=vocab.symbol_degree2(),
        slot_zero_coefficient_id=vocab.coefficient_to_id.get("0", -1),
        num_anchors=2,
    )
    batch = (
        [graph_batch],
        [torch.tensor([0])],
        torch.tensor([[True]]),
        torch.tensor([[1]]),
        torch.tensor([[2]]),
        torch.tensor([True]),
        [{"model": "QCD", "rank": "2_to_2", "term_slot_target": encoded}],
    )
    loss, _, _, _, _, _, _, degree_loss, _ = model._step_common(batch)
    assert torch.isfinite(loss)
    assert torch.isfinite(degree_loss)
    loss.backward()
    assert model.model.decoder.num_symbol.weight.grad is not None


def test_sequence_ce_path_remains_compatible():
    graph = Data(
        x=torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
            ]
        ),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
        edge_attr=torch.ones(4, 3),
        anchor_dists=torch.tensor(
            [[0.0, 1.0], [0.5, 1.0], [0.0, 1.0], [0.5, 1.0]]
        ),
    )
    graph_batch = Batch.from_data_list([graph])
    model = LitAmpGNN(
        in_dim=4,
        edge_dim=3,
        vocab_size=6,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        lr=1e-3,
        weight_decay=0.0,
        scheduler="none",
        warmup_steps=0,
        loss_mode="ce",
        enc_hid=16,
        enc_layers=1,
        enc_heads=4,
        d_model=16,
        dec_nhead=4,
        dec_layers=1,
        dec_max_len=8,
        num_anchors=2,
    )
    batch = (
        [graph_batch],
        [torch.tensor([0])],
        torch.tensor([[True]]),
        torch.tensor([[1, 3, 4]]),
        torch.tensor([[3, 4, 2]]),
        torch.tensor([True]),
        [{"model": "QCD", "rank": "2_to_2"}],
    )
    loss, _, _, _, _, _, _, degree_loss, degree_valid = model._step_common(batch)
    assert torch.isfinite(loss)
    assert degree_loss.item() == 0.0
    assert degree_valid.item() == 0.0
    loss.backward()
    assert model.model.decoder.to_logits.weight.grad is not None
