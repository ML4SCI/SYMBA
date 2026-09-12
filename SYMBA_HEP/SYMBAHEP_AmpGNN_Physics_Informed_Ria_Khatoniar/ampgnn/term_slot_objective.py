"""Vectorized canonical-slot CE objective for structured algebraic outputs."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


EncodedMonomial = Dict[str, object]
EncodedTarget = Dict[str, object]
DenseTargets = Dict[str, object]


def collate_term_slot_targets(
    targets: Sequence[EncodedTarget],
    *,
    num_slots: int,
    num_factors: int,
    den_slots: int,
    den_terms: int,
    den_term_factors: int,
    keep_loss: Optional[Sequence[bool]] = None,
) -> DenseTargets:
    """Pack variable-size encoded targets into fixed-shape CPU tensors."""
    batch_size = len(targets)
    shape_specs = {
        "global_coefficient": (batch_size,),
        "global_coupling": (batch_size,),
        "global_power": (batch_size,),
        "num_active": (batch_size, num_slots),
        "num_coefficient": (batch_size, num_slots),
        "num_factor_active": (batch_size, num_slots, num_factors),
        "num_symbol": (batch_size, num_slots, num_factors),
        "num_exponent": (batch_size, num_slots, num_factors),
        "den_active": (batch_size, den_slots),
        "den_power": (batch_size, den_slots),
        "den_term_active": (batch_size, den_slots, den_terms),
        "den_term_coefficient": (batch_size, den_slots, den_terms),
        "den_factor_active": (
            batch_size, den_slots, den_terms, den_term_factors
        ),
        "den_symbol": (batch_size, den_slots, den_terms, den_term_factors),
        "den_exponent": (batch_size, den_slots, den_terms, den_term_factors),
        "topology_num_degree2": (batch_size,),
        "topology_den_degree": (batch_size,),
        "field_total_per_sample": (batch_size,),
    }
    dense: DenseTargets = {
        key: torch.zeros(shape, dtype=torch.long) for key, shape in shape_specs.items()
    }

    for batch_index, target in enumerate(targets):
        global_target = target["global"]
        dense["global_coefficient"][batch_index] = int(global_target["coefficient"])
        dense["global_coupling"][batch_index] = int(global_target["coupling"])
        dense["global_power"][batch_index] = int(global_target["coupling_power"])

        numerator = list(target["numerator_terms"])
        if len(numerator) > num_slots:
            raise ValueError("numerator exceeds configured term-slot capacity")
        field_total = 5  # Three global fields plus the two activity layouts.
        for row, monomial in enumerate(numerator):
            factors = list(monomial["factors"])
            if len(factors) > num_factors:
                raise ValueError("numerator monomial exceeds factor capacity")
            dense["num_active"][batch_index, row] = 1
            dense["num_coefficient"][batch_index, row] = int(
                monomial["coefficient"]
            )
            field_total += 2 + 2 * len(factors)
            for factor_index, factor in enumerate(factors):
                dense["num_factor_active"][batch_index, row, factor_index] = 1
                dense["num_symbol"][batch_index, row, factor_index] = int(
                    factor["symbol"]
                )
                dense["num_exponent"][batch_index, row, factor_index] = int(
                    factor["exponent"]
                )

        denominator = list(target["denominator_factors"])
        if len(denominator) > den_slots:
            raise ValueError("denominator exceeds configured factor-slot capacity")
        for den_row, factor in enumerate(denominator):
            terms = list(factor["terms"])
            if len(terms) > den_terms:
                raise ValueError("denominator polynomial exceeds term capacity")
            dense["den_active"][batch_index, den_row] = 1
            dense["den_power"][batch_index, den_row] = int(factor["power"])
            field_total += 2
            for term_row, monomial in enumerate(terms):
                factors = list(monomial["factors"])
                if len(factors) > den_term_factors:
                    raise ValueError("denominator monomial exceeds factor capacity")
                dense["den_term_active"][batch_index, den_row, term_row] = 1
                dense["den_term_coefficient"][
                    batch_index, den_row, term_row
                ] = int(monomial["coefficient"])
                field_total += 2 + 2 * len(factors)
                for factor_index, monomial_factor in enumerate(factors):
                    dense["den_factor_active"][
                        batch_index, den_row, term_row, factor_index
                    ] = 1
                    dense["den_symbol"][
                        batch_index, den_row, term_row, factor_index
                    ] = int(monomial_factor["symbol"])
                    dense["den_exponent"][
                        batch_index, den_row, term_row, factor_index
                    ] = int(monomial_factor["exponent"])

        topology = target["topology_degree"]
        dense["topology_num_degree2"][batch_index] = int(
            topology["numerator_degree2"]
        )
        dense["topology_den_degree"][batch_index] = int(
            topology["denominator_degree"]
        )
        dense["field_total_per_sample"][batch_index] = field_total

    keep = [True] * batch_size if keep_loss is None else [bool(x) for x in keep_loss]
    if len(keep) != batch_size:
        raise ValueError("keep_loss and structured targets have different batch sizes")
    dense["kept_examples"] = sum(keep)
    dense["kept_field_total"] = sum(
        int(dense["field_total_per_sample"][i])
        for i, use_example in enumerate(keep)
        if use_example
    )
    return dense


def _layout_from_outputs(outputs: Mapping[str, torch.Tensor]) -> Dict[str, int]:
    return {
        "num_slots": outputs["num_active_logits"].size(1),
        "num_factors": outputs["num_factor_active_logits"].size(2),
        "den_slots": outputs["den_active_logits"].size(1),
        "den_terms": outputs["den_term_active_logits"].size(2),
        "den_term_factors": outputs["den_factor_active_logits"].size(3),
    }


def _dense_on_device(
    outputs: Mapping[str, torch.Tensor],
    targets: Union[Mapping[str, object], Sequence[EncodedTarget]],
    keep_loss: Optional[torch.Tensor],
) -> DenseTargets:
    if isinstance(targets, Mapping):
        dense = dict(targets)
    else:
        keep_values = None
        if keep_loss is not None:
            keep_values = keep_loss.detach().cpu().tolist()
        dense = collate_term_slot_targets(
            targets, **_layout_from_outputs(outputs), keep_loss=keep_values
        )
    device = outputs["global_coefficient_logits"].device
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in dense.items()
    }


def _cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float
) -> torch.Tensor:
    shape = targets.shape
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        label_smoothing=float(label_smoothing),
    ).reshape(shape)


def _weighted_active_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    no_object_weight: float,
    label_smoothing: float,
) -> torch.Tensor:
    """Match PyTorch's weighted-mean CE independently for each example."""
    batch_size = logits.size(0)
    weights = logits.new_tensor([float(no_object_weight), 1.0])
    losses = F.cross_entropy(
        logits.reshape(-1, 2),
        targets.reshape(-1),
        weight=weights,
        reduction="none",
        label_smoothing=float(label_smoothing),
    ).reshape(batch_size, -1)
    denominator = (
        weights.to(losses.dtype)[targets].reshape(batch_size, -1).sum(dim=1)
    )
    return losses.sum(dim=1) / denominator.clamp_min(torch.finfo(losses.dtype).eps)


def _masked_mean(
    values: torch.Tensor, mask: torch.Tensor, dims: Union[int, Tuple[int, ...]]
) -> torch.Tensor:
    mask = mask.to(values.dtype)
    denominator = mask.sum(dim=dims)
    numerator = (values * mask).sum(dim=dims)
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1.0),
        torch.zeros_like(numerator),
    )


def _monomial_losses(
    coefficient_logits: torch.Tensor,
    factor_active_logits: torch.Tensor,
    symbol_logits: torch.Tensor,
    exponent_logits: torch.Tensor,
    coefficient_target: torch.Tensor,
    factor_active_target: torch.Tensor,
    symbol_target: torch.Tensor,
    exponent_target: torch.Tensor,
    label_smoothing: float,
) -> torch.Tensor:
    coefficient = _cross_entropy(
        coefficient_logits, coefficient_target, label_smoothing
    )
    factor_layout = _cross_entropy(
        factor_active_logits, factor_active_target, label_smoothing
    ).mean(dim=-1)
    symbol = _cross_entropy(symbol_logits, symbol_target, label_smoothing)
    exponent = _cross_entropy(exponent_logits, exponent_target, label_smoothing)
    factor_semantics = _masked_mean(
        symbol + exponent, factor_active_target.bool(), dims=-1
    ) / 2.0
    return coefficient + factor_layout + factor_semantics


def _expected_monomial_degree2(
    factor_active_logits: torch.Tensor,
    symbol_logits: torch.Tensor,
    exponent_logits: torch.Tensor,
    symbol_degree2: torch.Tensor,
) -> torch.Tensor:
    active_probability = factor_active_logits.softmax(-1)[..., 1]
    symbol_degree = (
        symbol_logits.softmax(-1)
        * symbol_degree2.to(device=symbol_logits.device, dtype=symbol_logits.dtype)
    ).sum(dim=-1)
    exponent_values = torch.arange(
        exponent_logits.size(-1),
        device=exponent_logits.device,
        dtype=exponent_logits.dtype,
    )
    exponent = (exponent_logits.softmax(-1) * exponent_values).sum(dim=-1)
    return (active_probability * symbol_degree * exponent).sum(dim=-1)


def _decoded_monomial(
    coefficient_logits: torch.Tensor,
    factor_active_logits: torch.Tensor,
    symbol_logits: torch.Tensor,
    exponent_logits: torch.Tensor,
) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
    coefficient = int(coefficient_logits.argmax(-1).item())
    active = factor_active_logits.argmax(-1).detach().cpu().tolist()
    symbols = symbol_logits.argmax(-1).detach().cpu().tolist()
    exponents = exponent_logits.argmax(-1).detach().cpu().tolist()
    factors = tuple(
        (int(symbols[index]), int(exponents[index]))
        for index, is_active in enumerate(active)
        if int(is_active) == 1
    )
    return coefficient, factors


@torch.no_grad()
def decode_structured_slots(
    outputs: Mapping[str, torch.Tensor]
) -> List[Dict[str, object]]:
    """Argmax-decode a batch while preserving learned slot order."""
    decoded: List[Dict[str, object]] = []
    batch_size = outputs["global_coefficient_logits"].size(0)
    num_active = outputs["num_active_logits"].argmax(-1).detach().cpu().tolist()
    den_active = outputs["den_active_logits"].argmax(-1).detach().cpu().tolist()
    den_term_active = (
        outputs["den_term_active_logits"].argmax(-1).detach().cpu().tolist()
    )
    global_coefficient = (
        outputs["global_coefficient_logits"].argmax(-1).detach().cpu().tolist()
    )
    global_coupling = (
        outputs["global_coupling_logits"].argmax(-1).detach().cpu().tolist()
    )
    global_power = outputs["global_power_logits"].argmax(-1).detach().cpu().tolist()
    den_power = outputs["den_power_logits"].argmax(-1).detach().cpu().tolist()

    for batch_index in range(batch_size):
        numerator = tuple(
            _decoded_monomial(
                outputs["num_coefficient_logits"][batch_index, row],
                outputs["num_factor_active_logits"][batch_index, row],
                outputs["num_symbol_logits"][batch_index, row],
                outputs["num_exponent_logits"][batch_index, row],
            )
            for row, active in enumerate(num_active[batch_index])
            if int(active) == 1
        )
        denominator = []
        for den_row, active in enumerate(den_active[batch_index]):
            if int(active) != 1:
                continue
            terms = tuple(
                _decoded_monomial(
                    outputs["den_term_coefficient_logits"][
                        batch_index, den_row, term_row
                    ],
                    outputs["den_factor_active_logits"][
                        batch_index, den_row, term_row
                    ],
                    outputs["den_symbol_logits"][batch_index, den_row, term_row],
                    outputs["den_exponent_logits"][batch_index, den_row, term_row],
                )
                for term_row, is_active in enumerate(
                    den_term_active[batch_index][den_row]
                )
                if int(is_active) == 1
            )
            denominator.append((int(den_power[batch_index][den_row]), terms))
        decoded.append(
            {
                "global": {
                    "coefficient": int(global_coefficient[batch_index]),
                    "coupling": int(global_coupling[batch_index]),
                    "coupling_power": int(global_power[batch_index]),
                },
                "numerator_terms": numerator,
                "denominator_factors": tuple(denominator),
            }
        )
    return decoded


def structured_slot_objective(
    outputs: Dict[str, torch.Tensor],
    targets: Union[Mapping[str, object], Sequence[EncodedTarget]],
    *,
    keep_loss: Optional[torch.Tensor],
    symbol_degree2: torch.Tensor,
    zero_coefficient_id: int,
    label_smoothing: float = 0.0,
    no_object_weight: float = 0.1,
    return_predictions: bool = False,
) -> Dict[str, object]:
    """Compute canonical-slot CE, exact metrics, and topology degree loss."""
    if no_object_weight <= 0:
        raise ValueError("no_object_weight must be positive")
    dense = _dense_on_device(outputs, targets, keep_loss)
    device = outputs["global_coefficient_logits"].device
    batch_size = outputs["global_coefficient_logits"].size(0)
    keep = (
        torch.ones(batch_size, dtype=torch.bool, device=device)
        if keep_loss is None
        else keep_loss.to(device=device, dtype=torch.bool)
    )
    keep_float = keep.to(outputs["global_coefficient_logits"].dtype)

    global_loss = torch.stack(
        (
            _cross_entropy(
                outputs["global_coefficient_logits"],
                dense["global_coefficient"],
                label_smoothing,
            ),
            _cross_entropy(
                outputs["global_coupling_logits"],
                dense["global_coupling"],
                label_smoothing,
            ),
            _cross_entropy(
                outputs["global_power_logits"],
                dense["global_power"],
                label_smoothing,
            ),
        ),
        dim=-1,
    ).mean(dim=-1)

    num_active = dense["num_active"]
    num_active_mask = num_active.bool()
    num_active_loss = _weighted_active_loss(
        outputs["num_active_logits"],
        num_active,
        no_object_weight=no_object_weight,
        label_smoothing=label_smoothing,
    )
    num_monomial_loss = _monomial_losses(
        outputs["num_coefficient_logits"],
        outputs["num_factor_active_logits"],
        outputs["num_symbol_logits"],
        outputs["num_exponent_logits"],
        dense["num_coefficient"],
        dense["num_factor_active"],
        dense["num_symbol"],
        dense["num_exponent"],
        label_smoothing,
    )
    num_semantic_loss = _masked_mean(
        num_monomial_loss, num_active_mask, dims=-1
    )

    den_active = dense["den_active"]
    den_active_mask = den_active.bool()
    den_term_active = dense["den_term_active"]
    den_term_mask = den_active_mask.unsqueeze(-1) & den_term_active.bool()
    den_active_loss = _weighted_active_loss(
        outputs["den_active_logits"],
        den_active,
        no_object_weight=no_object_weight,
        label_smoothing=label_smoothing,
    )
    den_term_active_loss = _weighted_active_loss(
        outputs["den_term_active_logits"],
        den_term_active,
        no_object_weight=no_object_weight,
        label_smoothing=label_smoothing,
    )
    den_power_loss = _cross_entropy(
        outputs["den_power_logits"], dense["den_power"], label_smoothing
    )
    den_term_monomial_loss = _monomial_losses(
        outputs["den_term_coefficient_logits"],
        outputs["den_factor_active_logits"],
        outputs["den_symbol_logits"],
        outputs["den_exponent_logits"],
        dense["den_term_coefficient"],
        dense["den_factor_active"],
        dense["den_symbol"],
        dense["den_exponent"],
        label_smoothing,
    )
    den_row_loss = (
        den_power_loss
        + (den_term_monomial_loss * den_term_active.to(den_power_loss.dtype)).sum(-1)
    ) / (1.0 + den_term_active.sum(-1).to(den_power_loss.dtype))
    den_semantic_loss = _masked_mean(den_row_loss, den_active_mask, dims=-1)

    sample_main_loss = (
        global_loss
        + num_active_loss
        + num_semantic_loss
        + den_active_loss
        + den_term_active_loss
        + den_semantic_loss
    )
    kept_count = keep_float.sum()
    zero = outputs["global_coefficient_logits"].sum() * 0.0
    main_loss = torch.where(
        kept_count > 0,
        (sample_main_loss * keep_float).sum() / kept_count.clamp_min(1.0),
        zero,
    )

    num_degree_target = dense["topology_num_degree2"].to(global_loss.dtype)
    den_degree_target = dense["topology_den_degree"].to(global_loss.dtype)
    target_is_zero = (
        (num_active.sum(-1) == 1)
        & (dense["num_coefficient"][:, 0] == int(zero_coefficient_id))
        & (dense["num_factor_active"][:, 0].sum(-1) == 0)
    )
    expected_num_degree = _expected_monomial_degree2(
        outputs["num_factor_active_logits"],
        outputs["num_symbol_logits"],
        outputs["num_exponent_logits"],
        symbol_degree2,
    )
    num_scale = num_degree_target.abs().clamp_min(1.0).unsqueeze(-1)
    num_degree_error = (
        (expected_num_degree - num_degree_target.unsqueeze(-1)) / num_scale
    ).square()

    den_active_probability = outputs["den_active_logits"].softmax(-1)[..., 1]
    power_values = torch.arange(
        outputs["den_power_logits"].size(-1),
        device=device,
        dtype=outputs["den_power_logits"].dtype,
    )
    expected_den_power = (
        outputs["den_power_logits"].softmax(-1) * power_values
    ).sum(-1)
    expected_den_degree = (den_active_probability * expected_den_power).sum(-1)
    den_scale = den_degree_target.abs().clamp_min(1.0)
    den_degree_error = ((expected_den_degree - den_degree_target) / den_scale).square()

    expected_den_term_degree = _expected_monomial_degree2(
        outputs["den_factor_active_logits"],
        outputs["den_symbol_logits"],
        outputs["den_exponent_logits"],
        symbol_degree2,
    )
    den_term_degree_error = ((expected_den_term_degree - 2.0) / 2.0).square()
    degree_error_sum = (
        (num_degree_error * num_active.to(num_degree_error.dtype)).sum(-1)
        + den_degree_error
        + (den_term_degree_error * den_term_mask.to(den_term_degree_error.dtype))
        .sum(dim=(-1, -2))
    )
    degree_error_count = (
        num_active.sum(-1).to(global_loss.dtype)
        + 1.0
        + den_term_mask.sum(dim=(-1, -2)).to(global_loss.dtype)
    )
    sample_degree_loss = degree_error_sum / degree_error_count.clamp_min(1.0)
    sample_degree_loss = torch.where(
        target_is_zero, torch.zeros_like(sample_degree_loss), sample_degree_loss
    )
    degree_loss = torch.where(
        kept_count > 0,
        (sample_degree_loss * keep_float).sum() / kept_count.clamp_min(1.0),
        zero,
    )

    predictions = {key: value.argmax(-1) for key, value in outputs.items()}
    global_correct = torch.stack(
        (
            predictions["global_coefficient_logits"] == dense["global_coefficient"],
            predictions["global_coupling_logits"] == dense["global_coupling"],
            predictions["global_power_logits"] == dense["global_power"],
        ),
        dim=-1,
    )
    num_layout_correct = (predictions["num_active_logits"] == num_active).all(-1)
    num_coefficient_correct = (
        predictions["num_coefficient_logits"] == dense["num_coefficient"]
    )
    num_factor_layout_correct = (
        predictions["num_factor_active_logits"] == dense["num_factor_active"]
    ).all(-1)
    num_factor_mask = num_active_mask.unsqueeze(-1) & dense[
        "num_factor_active"
    ].bool()
    num_symbol_correct = predictions["num_symbol_logits"] == dense["num_symbol"]
    num_exponent_correct = (
        predictions["num_exponent_logits"] == dense["num_exponent"]
    )

    den_layout_correct = (predictions["den_active_logits"] == den_active).all(-1)
    den_power_correct = predictions["den_power_logits"] == dense["den_power"]
    den_term_layout_correct = (
        predictions["den_term_active_logits"] == den_term_active
    ).all(-1)
    den_term_coefficient_correct = (
        predictions["den_term_coefficient_logits"]
        == dense["den_term_coefficient"]
    )
    den_factor_layout_correct = (
        predictions["den_factor_active_logits"] == dense["den_factor_active"]
    ).all(-1)
    den_factor_mask = den_term_mask.unsqueeze(-1) & dense[
        "den_factor_active"
    ].bool()
    den_symbol_correct = predictions["den_symbol_logits"] == dense["den_symbol"]
    den_exponent_correct = (
        predictions["den_exponent_logits"] == dense["den_exponent"]
    )

    field_correct_per_sample = (
        global_correct.sum(-1)
        + num_layout_correct.long()
        + (num_coefficient_correct & num_active_mask).sum(-1)
        + (num_factor_layout_correct & num_active_mask).sum(-1)
        + (num_symbol_correct & num_factor_mask).sum(dim=(-1, -2))
        + (num_exponent_correct & num_factor_mask).sum(dim=(-1, -2))
        + den_layout_correct.long()
        + (den_power_correct & den_active_mask).sum(-1)
        + (den_term_layout_correct & den_active_mask).sum(-1)
        + (den_term_coefficient_correct & den_term_mask).sum(dim=(-1, -2))
        + (den_factor_layout_correct & den_term_mask).sum(dim=(-1, -2))
        + (den_symbol_correct & den_factor_mask).sum(dim=(-1, -2, -3))
        + (den_exponent_correct & den_factor_mask).sum(dim=(-1, -2, -3))
    )
    field_total = dense["field_total_per_sample"].to(field_correct_per_sample.dtype)
    field_correct = (field_correct_per_sample * keep.long()).sum()
    kept_field_total = (field_total * keep.long()).sum()
    token_acc = field_correct.to(global_loss.dtype) / kept_field_total.clamp_min(1)

    num_semantics_exact = (
        (~num_active_mask | num_coefficient_correct)
        & (~num_active_mask | num_factor_layout_correct)
    ).all(-1) & (
        (~num_factor_mask | (num_symbol_correct & num_exponent_correct))
        .flatten(1)
        .all(-1)
    )
    den_semantics_exact = (
        (~den_active_mask | den_power_correct).all(-1)
        & (~den_term_mask | den_term_coefficient_correct).flatten(1).all(-1)
        & (~den_term_mask | den_factor_layout_correct).flatten(1).all(-1)
        & (
            ~den_factor_mask | (den_symbol_correct & den_exponent_correct)
        ).flatten(1).all(-1)
    )
    exact_per_sample = (
        global_correct.all(-1)
        & num_layout_correct
        & den_layout_correct
        & den_term_layout_correct.flatten(1).all(-1)
        & num_semantics_exact
        & den_semantics_exact
    )
    seq_acc = (
        (exact_per_sample & keep).sum().to(global_loss.dtype)
        / kept_count.clamp_min(1.0)
    )

    predicted_num_active = predictions["num_active_logits"].bool()
    predicted_num_factor_active = predictions["num_factor_active_logits"].bool()
    predicted_num_degree = (
        predicted_num_factor_active.long()
        * symbol_degree2.to(device=device, dtype=torch.long)[
            predictions["num_symbol_logits"]
        ]
        * predictions["num_exponent_logits"]
    ).sum(-1)
    predicted_num_valid = (
        ~predicted_num_active
        | (predictions["num_coefficient_logits"] == int(zero_coefficient_id))
        | (predicted_num_degree == dense["topology_num_degree2"].unsqueeze(-1))
    ).all(-1)
    predicted_den_active = predictions["den_active_logits"].bool()
    predicted_den_degree = (
        predicted_den_active.long() * predictions["den_power_logits"]
    ).sum(-1)
    predicted_den_term_active = (
        predicted_den_active.unsqueeze(-1)
        & predictions["den_term_active_logits"].bool()
    )
    predicted_den_factor_active = predictions[
        "den_factor_active_logits"
    ].bool()
    predicted_den_term_degree = (
        predicted_den_factor_active.long()
        * symbol_degree2.to(device=device, dtype=torch.long)[
            predictions["den_symbol_logits"]
        ]
        * predictions["den_exponent_logits"]
    ).sum(-1)
    predicted_den_terms_valid = (
        ~predicted_den_term_active | (predicted_den_term_degree == 2)
    ).flatten(1).all(-1)
    zero_degree_valid = (
        predicted_num_active.any(-1)
        & (
            ~predicted_num_active
            | (predictions["num_coefficient_logits"] == int(zero_coefficient_id))
        ).all(-1)
        & ~predicted_den_active.any(-1)
    )
    nonzero_degree_valid = (
        predicted_num_active.any(-1)
        & predicted_num_valid
        & (predicted_den_degree == dense["topology_den_degree"])
        & predicted_den_terms_valid
    )
    degree_valid_per_sample = torch.where(
        target_is_zero, zero_degree_valid, nonzero_degree_valid
    )
    degree_valid = (
        (degree_valid_per_sample & keep).sum().to(global_loss.dtype)
        / kept_count.clamp_min(1.0)
    )

    decoded_predictions: List[Dict[str, object]] = []
    exact_flags: List[Optional[bool]] = []
    if return_predictions:
        decoded_predictions = decode_structured_slots(outputs)
        exact_cpu = exact_per_sample.detach().cpu().tolist()
        keep_cpu = keep.detach().cpu().tolist()
        exact_flags = [
            bool(exact) if bool(use_example) else None
            for exact, use_example in zip(exact_cpu, keep_cpu)
        ]
        decoded_predictions = [
            prediction if bool(use_example) else {}
            for prediction, use_example in zip(decoded_predictions, keep_cpu)
        ]

    return {
        "main_loss": main_loss,
        "degree_loss": degree_loss,
        "token_acc": token_acc,
        "seq_acc": seq_acc,
        "degree_valid": degree_valid,
        "batch_size": int(dense.get("kept_examples", batch_size)),
        "token_weight": int(dense.get("kept_field_total", 0)),
        "predictions": decoded_predictions,
        "exact_flags": exact_flags,
    }
