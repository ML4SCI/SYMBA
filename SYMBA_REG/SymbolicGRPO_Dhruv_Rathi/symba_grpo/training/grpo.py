"""Sequence-level group-relative policy updates for symbolic regression.

Each step samples a group from the current policy, evaluates every completion
on one shared numerical support, normalizes reward within the group, and takes
two clipped decoder passes against one behavior snapshot. The proposals seed the same GP
search used to track the returned expression.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import sympy as sp
import torch
import torch.nn.functional as F
from loguru import logger

from symba_grpo.gp.primitives import make_primitive_set
from symba_grpo.gp.evolution import setup_toolbox, run_gp
from symba_grpo.training.predictor import (
    allowed_next_token_ids,
    generate_square_subsequent_mask,
    grammar_sequence_logprobs,
)
from symba_grpo.tokenizers.prefix import prefix_to_sympy
from symba_grpo.tokenizers.vocab import Vocab
from symba_grpo.rng import isolated_global_rng, stable_seed


class SymbolicGRPOTrainer:
    """Group-relative policy optimization for the SR transformer."""

    def __init__(self, model, vocab: Vocab, config):
        self.model = model
        self.vocab = vocab
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        for module in (self.model.transformer.encoder, self.model.src_tok_emb):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.group_size = getattr(config, "grpo_group_size", 16)
        self.clip_eps = getattr(config, "grpo_clip_eps", 0.2)
        self.temperature = getattr(config, "grpo_temperature", 1.0)
        self.max_len = getattr(config, "grpo_max_len", 64)
        self.policy_epochs = getattr(config, "grpo_policy_epochs", 2)
        if self.policy_epochs < 2:
            raise ValueError("GRPO requires at least two passes over each sampled group")
        self.model.eval()
        self.accounting = {
            "trainable_parameter_scope": "decoder_target_embedding_generator",
            "beam_banks": 0,
            "gp_searches": 0,
            "group_trajectories": 0,
            "optimizer_passes": 0,
            "groups_without_reward_variation": 0,
            "sample_decoder_forwards": 0,
            "sampled_tokens": 0,
            "likelihood_model_forwards": 0,
            "gp_generations": 0,
            "gp_early_stops": 0,
        }

    def _decoder_parameters(self):
        modules = (self.model.transformer.decoder, self.model.tgt_tok_emb, self.model.generator)
        return [p for module in modules for p in module.parameters()]

    # ── encoding ──────────────────────────────────────────────────────
    @torch.no_grad()
    def _encode(self, src, num_array):
        src = src.to(self.device)
        num_array = num_array.to(self.device)
        src_mask = torch.zeros(src.shape[1], src.shape[1], device=self.device).float()
        src_pad = ((src == 0).all(dim=-1) if src.dim() == 3 else (src == 0)).float()
        src_pad = src_pad.masked_fill(src_pad == 1, float("-inf")).to(self.device)
        return self.model.encode(src, num_array, src_mask, src_pad)

    # ── sampling a group ──────────────────────────────────────────────
    @torch.no_grad()
    def _sample_group(self, memory, g: int) -> List[List[int]]:
        """Sample g completions by temperature sampling from the policy."""
        seqs = []
        for _ in range(g):
            ys = [self.vocab.bos_id]
            for _ in range(self.max_len - 1):
                t = torch.tensor([ys], dtype=torch.long, device=self.device)
                mask = generate_square_subsequent_mask(t.size(1), self.device)
                out = self.model.decode(t, memory, mask)
                self.accounting["sample_decoder_forwards"] += 1
                logits = self.model.generator(out[:, -1]).squeeze(0) / max(self.temperature, 1e-6)
                allowed = allowed_next_token_ids(
                    self.vocab, ys, getattr(self, "_active_num_vars", 1)
                )
                if not allowed:
                    break
                masked = torch.full_like(logits, -torch.inf)
                masked[allowed] = logits[allowed]
                probs = F.softmax(masked, dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())
                ys.append(nxt)
                self.accounting["sampled_tokens"] += 1
                if nxt == self.vocab.eos_id:
                    break
            seqs.append(ys)
        return seqs

    def _seq_logprob(self, memory, seq_ids: List[int]) -> torch.Tensor:
        """Differentiable sum log-prob of a token sequence under current policy."""
        t = torch.tensor([seq_ids], dtype=torch.long, device=self.device)
        tgt_in, tgt_out = t[:, :-1], t[:, 1:]
        mask = generate_square_subsequent_mask(tgt_in.size(1), self.device)
        out = self.model.decode(tgt_in, memory, mask)
        self.accounting["likelihood_model_forwards"] += 1
        logits = self.model.generator(out)
        return grammar_sequence_logprobs(
            logits,
            tgt_in,
            tgt_out,
            self.vocab,
            getattr(self, "_active_num_vars", 1),
            temperature=self.temperature,
            padded=False,
        ).squeeze(0)

    # ── reward ────────────────────────────────────────────────────────
    def _reward(self, ids: List[int], X, y) -> Tuple[float, Optional[sp.Expr]]:
        try:
            toks = self.vocab.decode_generated(ids)
        except ValueError:
            return 0.0, None
        try:
            expr = prefix_to_sympy(toks, evaluate=False)
        except Exception:
            return 0.0, None
        from symba_grpo.evaluation.safe_eval import safe_evaluate
        from symba_grpo.evaluation.metrics import r2_score
        try:
            yp = safe_evaluate(expr, X)
            if not np.isfinite(yp).all():
                return 0.0, expr
            r2 = r2_score(y, yp)
            return (max(0.0, min(1.0, r2)) if math.isfinite(r2) else 0.0), expr
        except Exception:
            return 0.0, expr

    def grpo_step(self, memory, X, y, lr: float, support_n: int = 500) -> Tuple[float, float, Optional[sp.Expr], float]:
        """Sample, score, normalize within group, and take clipped policy passes.

        Returns (loss, mean_reward, best_expr, best_reward).
        """
        seqs = self._sample_group(memory, self.group_size)
        self.accounting["group_trajectories"] += len(seqs)
        n = min(support_n, X.shape[0])
        rng = np.random.default_rng(getattr(self, "_group_support_seed", 0))
        idx = rng.choice(X.shape[0], n, replace=False)
        group_X, group_y = X[idx], y[idx]
        rewards, exprs = [], []
        for s in seqs:
            r, e = self._reward(s, group_X, group_y)
            rewards.append(r)
            exprs.append(e)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        if rewards_t.std() < 1e-8:
            # No learning signal this group (all-equal rewards); skip update.
            best_i = int(rewards_t.argmax())
            self.accounting["groups_without_reward_variation"] += 1
            return 0.0, float(rewards_t.mean()), exprs[best_i], float(rewards_t[best_i])
        adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

        self.model.eval()
        with torch.no_grad():
            old_logps = torch.stack([self._seq_logprob(memory, s) for s in seqs])

        parameters = self._decoder_parameters()
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
        loss = torch.zeros((), device=self.device)
        for _ in range(self.policy_epochs):
            optimizer.zero_grad()
            new_logps = torch.stack([self._seq_logprob(memory, s) for s in seqs])
            ratio = torch.exp(new_logps - old_logps)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            loss = -torch.min(unclipped, clipped).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            self.accounting["optimizer_passes"] += 1
        self._last_ratio_departure = float(torch.max(torch.abs(ratio.detach() - 1.0)))
        self._last_clip_fraction = float(
            ((ratio.detach() < 1 - self.clip_eps) | (ratio.detach() > 1 + self.clip_eps)).float().mean()
        )
        self.model.eval()

        best_i = int(rewards_t.argmax())
        return float(loss.item()), float(rewards_t.mean()), exprs[best_i], float(rewards_t[best_i])

    def training_loop(
        self,
        predictor,
        src: torch.Tensor,
        num_array: torch.Tensor,
        points: list,
        num_vars: int,
        num_cycles: Optional[int] = None,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        selection_supports: Optional[list[dict[str, np.ndarray]]] = None,
        experiment_key: str = "experiment",
    ) -> Tuple:
        """PIGP + GRPO loop. Returns (model, best_expr, best_r2).

        Each cycle: (1) PIGP-style transformer+GP search to track the best
        expression (never-regress), then (2) several GRPO steps that fine-tune
        the transformer from its own group rewards (no frozen reference).
        """
        num_cycles = num_cycles or self.config.dpo_num_cycles
        lr = getattr(self.config, "grpo_lr", self.config.dpo_lr)
        steps_per_cycle = getattr(self.config, "grpo_steps_per_cycle", self.config.dpo_epochs_per_cycle)
        pset = make_primitive_set(num_vars)
        self._active_num_vars = num_vars

        best_expr, best_r2 = None, -float("inf")

        for cycle in range(num_cycles):
            logger.info("GRPO Cycle {}/{}", cycle + 1, num_cycles)
            try:
                memory = self._encode(src, num_array)

                # (1) One beam bank is scored and reused as the GP seed bank.
                from symba_grpo.gp.pigp import _score_candidates, _safe_r2
                cands = predictor.predict_sympy(
                    src.to(self.device), num_array.to(self.device),
                    num_vars,
                    self.config.beam_size, self.config.num_equations,
                )
                self.accounting["beam_banks"] += 1
                if X is not None and y is not None and cands:
                    indices = selection_supports[cycle]["pre"] if selection_supports else None
                    scored = _score_candidates(cands, X, y, support_indices=indices)
                    if scored:
                        candidate = scored[0][0]
                        candidate_r2 = _safe_r2(candidate, X, y)
                        if candidate_r2 > best_r2:
                            best_r2, best_expr = candidate_r2, candidate

                from symba_grpo.gp.bridge import convert_to_protected_form, sympy_to_deap
                seed_exprs = []
                for expression in cands:
                    try:
                        seed_exprs.append(convert_to_protected_form(sympy_to_deap(expression)))
                    except Exception:
                        continue
                toolbox = setup_toolbox(pset, points)
                with isolated_global_rng(experiment_key, cycle, "gp"):
                    ind, gp_r2, population, hof, convergence = run_gp(
                        toolbox, points, pset, seed_exprs,
                        self.config.pop_size, self.config.num_generations,
                        self.config.cxpb, self.config.mutpb, self.config.gp_verbose,
                    )
                self.accounting["gp_searches"] += 1
                generations = max(generation for generation, _ in convergence)
                self.accounting["gp_generations"] += generations
                self.accounting["gp_early_stops"] += int(generations < self.config.num_generations)
                if X is not None and y is not None:
                    from symba_grpo.gp.fast_eval import select_ordinary_gp_candidate
                    selected_tree, gp_full, selection_diagnostics = select_ordinary_gp_candidate(
                        population, hof, X, y,
                    )
                    logger.info(
                        "  GP ordinary shortlist: size={} valid={} ties={}",
                        selection_diagnostics["shortlist_size"],
                        selection_diagnostics["ordinary_valid"],
                        selection_diagnostics["numerical_ties"],
                    )
                else:
                    selected_tree = ind
                    gp_full = gp_r2
                if selected_tree is not None and gp_full > best_r2:
                    from symba_grpo.gp.bridge import GPExpression
                    best_r2 = gp_full
                    best_expr = GPExpression.from_tree(selected_tree, num_vars)

                # (2) GRPO updates from the policy's own sampled group.
                if X is not None and y is not None:
                    for step in range(steps_per_cycle):
                        step_seed = stable_seed(experiment_key, cycle, "PIGP+GRPO", step)
                        self._group_support_seed = step_seed
                        with isolated_global_rng(experiment_key, cycle, "PIGP+GRPO", step):
                            loss, mean_r, bexpr, br = self.grpo_step(memory, X, y, lr)
                        # The reward support nominates one group-best sample;
                        # admission uses ordinary semantics on the full policy
                        # panel, never the 500-row reward subset.
                        if bexpr is not None:
                            sampled_full_r2 = _safe_r2(bexpr, X, y)
                            if sampled_full_r2 > best_r2:
                                best_expr, best_r2 = bexpr, sampled_full_r2
                        memory = self._encode(src, num_array)  # refresh after update
                    logger.info("  GRPO cycle {} done | last loss={:.4f} mean_reward={:.4f}",
                                cycle + 1, loss, mean_r)

                # Match DPO's second proposal call and direct post-update
                # selection.
                post_candidates = predictor.predict_sympy(
                    src.to(self.device), num_array.to(self.device),
                    num_vars,
                    self.config.beam_size, self.config.num_equations,
                )
                self.accounting["beam_banks"] += 1
                if X is not None and y is not None and post_candidates:
                    from symba_grpo.gp.pigp import _score_candidates
                    indices = selection_supports[cycle]["post"] if selection_supports else None
                    post_scored = _score_candidates(post_candidates, X, y, support_indices=indices)
                    if post_scored:
                        candidate = post_scored[0][0]
                        candidate_r2 = _safe_r2(candidate, X, y)
                        if candidate_r2 > best_r2:
                            best_expr, best_r2 = candidate, candidate_r2
            except Exception as e:
                raise RuntimeError(f"group-relative cycle {cycle + 1} failed") from e

        logger.info("  GRPO final best R²={:.6f}  expr={}", best_r2, str(best_expr)[:100] if best_expr else "None")
        return self.model, best_expr, best_r2
