"""Symbolic Direct Preference Optimisation (SymbolicDPO).

Iteratively refines a transformer using GP-generated preference pairs.
Ported from Samyak's SymbolicDPOTrainer with bug fixes and cleanup.

The loop:
  1. Transformer generates candidates (beam search)
  2. GP evolves and refines them
  3. Top GP solutions form preference pairs (better, worse)
  4. DPO fine-tunes the transformer
  5. Repeat with increasing beta
"""

import copy
import math
import signal
from typing import List, Tuple, Optional

import sympy as sp

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from loguru import logger

from symba_grpo.gp.primitives import make_primitive_set
from symba_grpo.gp.evolution import setup_toolbox, run_gp
from symba_grpo.gp.bridge import (
    GPExpression,
    deap_tree_fingerprint,
    deap_tree_to_prefix,
    convert_to_protected_form,
    sympy_to_deap,
)
from symba_grpo.models.seq2seq import create_masks
from symba_grpo.tokenizers.prefix import sympy_to_prefix, prefix_to_sympy
from symba_grpo.tokenizers.vocab import Vocab
from symba_grpo.rng import isolated_global_rng, stable_seed
from symba_grpo.training.predictor import full_vocabulary_sequence_logprobs

DPO_DEFAULT_PREF_MAX_LEN = 80
DPO_CHECK_BAD_MAX_LEN = DPO_DEFAULT_PREF_MAX_LEN - 2
class PreferenceDataset(Dataset):
    """Dataset of (better, worse) tokenized expression pairs."""

    def __init__(self, pairs: List[Tuple[list, list]], vocab: Vocab, max_len: int = 80):
        self.pairs = pairs
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        better_tokens, worse_tokens = self.pairs[idx]

        better_ids = self.vocab.encode_sequence(better_tokens)
        worse_ids = self.vocab.encode_sequence(worse_tokens)

        better_padded = self.vocab.pad_sequence(better_ids, self.max_len)
        worse_padded = self.vocab.pad_sequence(worse_ids, self.max_len)

        return (
            torch.from_numpy(better_padded),
            torch.from_numpy(worse_padded),
        )


def _brief_expr(expression, limit: int = 100) -> str:
    """Render an expression for a log line, with a hard time bound.

    SymPy's printer can escalate mpmath precision without bound on deeply
    nested trigonometric powers, so an untimed ``str()`` here can block the
    search for many minutes on a string that is only ever logged and then
    truncated. Term ordering is disabled because the ordering pass is what
    triggers the numeric evaluation.
    """
    if expression is None:
        return "None"

    def _expired(signum, frame):
        raise TimeoutError

    previous = None
    try:
        previous = signal.signal(signal.SIGALRM, _expired)
        signal.alarm(2)
    except (ValueError, AttributeError):
        previous = None  # not the main thread, or no SIGALRM on this platform
    try:
        return sp.sstr(expression, order="none")[:limit]
    except Exception:
        return f"<{type(expression).__name__}: unprintable>"
    finally:
        if previous is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)


def generate_preference_pairs(
    population: list,
    top_n: int = 10,
    compare_with_n: int = 100,
) -> List[Tuple]:
    """Generate (better, worse) DEAP individual pairs from a GP population.

    Selects top_n individuals as preferred, pairs each against a pool of
    worse individuals drawn from rank top_n onwards.
    """
    sorted_pop = sorted(population, key=lambda ind: ind.fitness.values[0])
    top_individuals = sorted_pop[:top_n]
    compare_pool = sorted_pop[top_n:top_n + compare_with_n]

    seen_better = set()
    pairs = []
    for ind1 in top_individuals:
        fit1 = ind1.fitness.values[0]
        if not math.isfinite(fit1):
            continue
        ind1_key = deap_tree_fingerprint(ind1)
        if ind1_key in seen_better:
            continue
        seen_better.add(ind1_key)
        for ind2 in compare_pool:
            fit2 = ind2.fitness.values[0]
            if not math.isfinite(fit2):
                continue
            if fit1 < fit2:
                pairs.append((ind1, ind2))

    return pairs


def _check_bad(tokens: list, max_len: int = DPO_CHECK_BAD_MAX_LEN) -> bool:
    """Reject only bodies that cannot fit with the required BOS/EOS pair."""
    return not tokens or len(tokens) > max_len


def tokenize_preference_pairs(
    deap_pairs: list,
    vocab: Vocab,
    num_vars: int,
) -> Tuple[List[Tuple[list, list]], dict[str, int]]:
    """Compile each unique GP tree once without constructing SymPy objects."""
    token_pairs = []
    cache: dict[tuple, Optional[list[str]]] = {}
    object_keys: dict[int, tuple] = {}
    stats = {
        "preference_unique_trees": 0,
        "preference_cache_hits": 0,
        "preference_serialization_rejections": 0,
    }

    def convert(individual) -> Optional[list[str]]:
        identity = id(individual)
        key = object_keys.get(identity)
        if key is None:
            key = deap_tree_fingerprint(individual)
            object_keys[identity] = key
        if key in cache:
            stats["preference_cache_hits"] += 1
            return cache[key]
        stats["preference_unique_trees"] += 1
        tokens = None
        try:
            tokens = deap_tree_to_prefix(
                individual,
                num_vars=num_vars,
                max_body_tokens=DPO_CHECK_BAD_MAX_LEN,
            )
            # Check the exact decoder vocabulary as well as the structural map.
            for token in tokens:
                vocab.encode_token(token)
            if _check_bad(tokens):
                stats["preference_serialization_rejections"] += 1
                tokens = None
        except Exception:
            stats["preference_serialization_rejections"] += 1
        cache[key] = tokens
        return tokens

    for better_ind, worse_ind in deap_pairs:
        better_tokens = convert(better_ind)
        worse_tokens = convert(worse_ind)
        if better_tokens is not None and worse_tokens is not None:
            token_pairs.append((better_tokens, worse_tokens))
    return token_pairs, stats


def dpo_loss(pi_logps: torch.Tensor, ref_logps: torch.Tensor, beta: float) -> torch.Tensor:
    """Compute DPO loss.

    Args:
        pi_logps: (B, 2) - log-probs for [better, worse] under policy
        ref_logps: (B, 2) - log-probs for [better, worse] under reference
        beta: sensitivity parameter

    Returns:
        scalar loss
    """
    pi_logratios = pi_logps[:, 0] - pi_logps[:, 1]
    ref_logratios = ref_logps[:, 0] - ref_logps[:, 1]
    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    return losses.mean()


def _get_sequence_logprobs(model, src, num_array, tgt, vocab, device, num_vars):
    """Compute sum of log-probs for a target sequence under a model."""
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

    src_mask, tgt_mask, src_pad, tgt_pad = create_masks(
        src, tgt_in, pad_idx=vocab.pad_id, device=device,
    )
    logits = model(
        src, num_array, tgt_in,
        src_mask, tgt_mask, src_pad, tgt_pad, src_pad,
    )
    return full_vocabulary_sequence_logprobs(
        logits, tgt_in, tgt_out, vocab, num_vars, temperature=1.0, padded=True
    )


class SymbolicDPOTrainer:
    """Iterative DPO training loop for symbolic regression."""

    def __init__(
        self,
        model,
        vocab: Vocab,
        config,
    ):
        self.model = model
        self.reference_model = copy.deepcopy(model)
        for p in self.reference_model.parameters():
            p.requires_grad = False
        self.reference_model.eval()

        self.vocab = vocab
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.reference_model.to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        self.accounting = {
            "trainable_parameter_scope": "all_model_parameters",
            "beam_banks": 0,
            "gp_searches": 0,
            "preference_pairs": 0,
            "optimizer_steps": 0,
            "likelihood_model_forwards": 0,
            "gp_generations": 0,
            "gp_early_stops": 0,
            "preference_unique_trees": 0,
            "preference_cache_hits": 0,
            "preference_serialization_rejections": 0,
            "preference_cycles_without_pairs": 0,
        }
    def _policy_parameters(self):
        """Train the complete encoder--decoder policy."""
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def train_dpo_epoch(
        self,
        preference_pairs: List[Tuple[list, list]],
        src_batch: torch.Tensor,
        num_array_batch: torch.Tensor,
        beta: float,
        lr: float = 1e-4,
        epochs: int = 10,
        all_src_contexts: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        rng_seed: Optional[int] = None,
        num_vars: int = 1,
    ) -> float:
        """One round of DPO training on a set of preference pairs.

        If all_src_contexts is provided (multi-context augmentation), each
        training step samples a random context from the list.
        """
        if not preference_pairs:
            return 0.0

        batch_size = getattr(self.config, "dpo_batch_size", 16)
        dataset = PreferenceDataset(
            preference_pairs, self.vocab, max_len=DPO_DEFAULT_PREF_MAX_LEN,
        )
        generator = torch.Generator()
        generator.manual_seed(int(rng_seed or 0))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

        parameters = self._policy_parameters()
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

        last_epoch_avg_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for better, worse in loader:
                better = better.to(self.device)
                worse = worse.to(self.device)
                cur_bs = better.size(0)

                if all_src_contexts and len(all_src_contexts) > 1:
                    ctx_idx = torch.randint(0, len(all_src_contexts), (1,)).item()
                    src, num_arr = all_src_contexts[ctx_idx]
                else:
                    src = src_batch.to(self.device)
                    num_arr = num_array_batch.to(self.device)

                src_exp = src.expand(cur_bs, -1, -1) if src.size(0) == 1 else src
                num_exp = num_arr.expand(cur_bs, *(-1,) * (num_arr.dim() - 1)) if num_arr.size(0) == 1 else num_arr

                with torch.no_grad():
                    ref_better = _get_sequence_logprobs(
                        self.reference_model, src_exp, num_exp, better,
                        self.vocab, self.device, num_vars,
                    )
                    ref_worse = _get_sequence_logprobs(
                        self.reference_model, src_exp, num_exp, worse,
                        self.vocab, self.device, num_vars,
                    )
                self.accounting["likelihood_model_forwards"] += 2

                # DPO compares policy and reference likelihoods under the same
                # deterministic dropout state. Eval mode preserves autograd.
                self.model.eval()
                optimizer.zero_grad()

                pi_better_train = _get_sequence_logprobs(
                    self.model, src_exp, num_exp, better,
                    self.vocab, self.device, num_vars,
                )
                pi_worse_train = _get_sequence_logprobs(
                    self.model, src_exp, num_exp, worse,
                    self.vocab, self.device, num_vars,
                )
                self.accounting["likelihood_model_forwards"] += 2

                pi_logps = torch.stack([pi_better_train, pi_worse_train], dim=1)
                ref_logps = torch.stack([ref_better, ref_worse], dim=1)

                loss = dpo_loss(pi_logps, ref_logps, beta)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                self.accounting["optimizer_steps"] += 1
                epoch_loss += loss.item()

            scheduler.step()
            last_epoch_avg_loss = epoch_loss / max(len(loader), 1)

        return last_epoch_avg_loss

    def training_loop(
        self,
        predictor,
        src: torch.Tensor,
        num_array: torch.Tensor,
        points: list,
        num_vars: int,
        num_cycles: Optional[int] = None,
        X: Optional["np.ndarray"] = None,
        y: Optional["np.ndarray"] = None,
        selection_supports: Optional[list[dict[str, "np.ndarray"]]] = None,
        experiment_key: str = "experiment",
    ) -> Tuple:
        """Full iterative DPO loop used by the paired DPO/GRPO experiment.

        Args:
            X, y: Full dataset arrays for scoring candidates.

        Returns (model, best_expr, best_r2)
        """
        import numpy as np
        num_cycles = num_cycles or self.config.dpo_num_cycles
        beta = self.config.dpo_beta
        pset = make_primitive_set(num_vars)

        all_src_contexts = None

        best_expr = None
        best_r2 = -float("inf")

        for cycle in range(num_cycles):
            logger.info("DPO Cycle {}/{} | beta={:.4f}", cycle + 1, num_cycles, beta)
            # Unexpected cycle failures propagate to the experiment boundary,
            # which emits a complete failure row with frozen provenance.
            try:
                sympy_candidates = predictor.predict_sympy(
                    src.to(self.device), num_array.to(self.device),
                    num_vars,
                    self.config.beam_size, self.config.num_equations,
                )
                self.accounting["beam_banks"] += 1
                logger.info("  Transformer produced {} candidates", len(sympy_candidates))

                if X is not None and y is not None and sympy_candidates:
                    from symba_grpo.gp.pigp import _score_candidates
                    indices = selection_supports[cycle]["pre"] if selection_supports else None
                    scored = _score_candidates(sympy_candidates, X, y, support_indices=indices)
                    if scored:
                        from symba_grpo.gp.pigp import _safe_r2
                        tf_best_expr = scored[0][0]
                        tf_best_r2 = _safe_r2(tf_best_expr, X, y)
                        logger.info("  Transformer best R²={:.6f}  expr={}", tf_best_r2, _brief_expr(tf_best_expr, 80))
                        if tf_best_r2 > best_r2:
                            best_r2 = tf_best_r2
                            best_expr = tf_best_expr

                # Reuse the beam candidates already decoded above for scoring as
                # GP seeds: avoids a second (expensive, GPU-bound) beam search.
                from symba_grpo.gp.bridge import sympy_to_deap, convert_to_protected_form
                seed_exprs = []
                for e in sympy_candidates:
                    try:
                        seed_exprs.append(convert_to_protected_form(sympy_to_deap(e)))
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
                logger.info("  GP best R²(internal)={:.6f}", gp_r2)

                if X is not None and y is not None:
                    from symba_grpo.gp.fast_eval import select_ordinary_gp_candidate
                    selected_tree, gp_full_r2, selection_diagnostics = select_ordinary_gp_candidate(
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
                    gp_full_r2 = gp_r2

                if selected_tree is not None and gp_full_r2 > best_r2:
                    best_r2 = gp_full_r2
                    best_expr = GPExpression.from_tree(selected_tree, num_vars)
                    logger.info("  GP improved global best to R²={:.6f}", best_r2)

                deap_pairs = generate_preference_pairs(
                    population,
                    self.config.dpo_top_n, self.config.dpo_compare_n,
                )
                token_pairs, conversion_stats = tokenize_preference_pairs(
                    deap_pairs,
                    self.vocab,
                    num_vars=num_vars,
                )
                for key, value in conversion_stats.items():
                    self.accounting[key] += value
                self.accounting["preference_pairs"] += len(token_pairs)
                logger.info(
                    "  Preference serialization: unique={} cache_hits={} rejected={}",
                    conversion_stats["preference_unique_trees"],
                    conversion_stats["preference_cache_hits"],
                    conversion_stats["preference_serialization_rejections"],
                )
                logger.info("  Generated {} preference pairs", len(token_pairs))
                if not token_pairs:
                    self.accounting["preference_cycles_without_pairs"] += 1
                    logger.info("  No valid preference pairs; skipping this DPO update")

                if token_pairs:
                    loss = self.train_dpo_epoch(
                        token_pairs, src, num_array,
                        beta, self.config.dpo_lr, self.config.dpo_epochs_per_cycle,
                        all_src_contexts=all_src_contexts,
                        rng_seed=stable_seed(experiment_key, cycle, "PIGP+DPO"),
                        num_vars=num_vars,
                    )
                    logger.info("  DPO loss: {:.4f}", loss)

                # Evaluate one post-update proposal bank so the last update in
                # every cycle can affect the returned expression.  This bank is
                # scored directly; the GP budget remains one search per cycle.
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
                        from symba_grpo.gp.pigp import _safe_r2
                        post_expr = post_scored[0][0]
                        post_r2 = _safe_r2(post_expr, X, y)
                        if post_r2 > best_r2:
                            best_expr, best_r2 = post_expr, post_r2
            except Exception as e:
                raise RuntimeError(f"DPO cycle {cycle + 1} failed") from e

            beta *= self.config.dpo_beta_multiplier

        logger.info("  DPO final best R²={:.6f}  expr={}", best_r2, _brief_expr(best_expr, 100))
        return self.model, best_expr, best_r2
