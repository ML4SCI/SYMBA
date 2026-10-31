"""PIGP without refinement: the unrefined-prior baseline.

The arm reuses ``SymbolicGRPOTrainer``'s loop unchanged and neutralizes only the
policy step, by overriding :meth:`grpo_step` to take no sample and no gradient.
Because the override returns no candidate expression, the loop's admission
branch is skipped, so the decoder is never updated and no sampled trajectory
ever enters the candidate pool. Everything else: the number of cycles, both
beam banks per cycle, the GP budget and its RNG stream, and the pre- and
post-selection panels, is identical to the refinement arms, so the
candidate-selection budget matches them exactly.

What it does and does not control for. It answers whether the refinement
apparatus as a whole beats a frozen pretrained prior seeding GP. It does NOT
isolate the gradient step, because the block it neutralizes also admits the
group-argmax of each sampled group into the candidate pool: with the frozen
configuration that is grpo_group_size * grpo_steps_per_cycle * dpo_num_cycles =
192 extra proposal expressions per equation that this arm never draws. A
sampling-only arm, which would take those draws with the optimizer disabled, is
the control that separates the update from the extra draws, and is not part of
this study.
"""

from __future__ import annotations

from typing import Optional, Tuple

import sympy as sp

from symba_grpo.training.grpo import SymbolicGRPOTrainer


class PIGPBaselineTrainer(SymbolicGRPOTrainer):
    """SymbolicGRPO's loop with the policy step neutralized."""

    def __init__(self, model, vocab, config):
        super().__init__(model, vocab, config)
        # No parameter is ever updated in this arm; do not inherit GRPO's claim.
        self.accounting["trainable_parameter_scope"] = "none"

    def grpo_step(
        self, memory, X, y, lr: float, support_n: int = 500
    ) -> Tuple[float, float, Optional[sp.Expr], float]:
        """Take no sample and no gradient, and nominate no candidate.

        Returning ``None`` for the expression makes the caller's admission
        branch a no-op, so this arm's best expression comes only from the beam
        banks and the GP search.
        """
        return 0.0, 0.0, None, 0.0
