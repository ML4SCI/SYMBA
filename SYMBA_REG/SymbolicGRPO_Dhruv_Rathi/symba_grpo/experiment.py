"""Data panels and the two evaluation arms (DPO, GRPO)."""

from __future__ import annotations

import copy
import hashlib
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
import torch
from loguru import logger
from tqdm import tqdm

from symba_grpo.evaluation.metrics import evaluate_batch, evaluate_expression, prediction_record
from symba_grpo.training.dpo import SymbolicDPOTrainer
from symba_grpo.training.grpo import SymbolicGRPOTrainer
from symba_grpo.training.pigp_baseline import PIGPBaselineTrainer
from symba_grpo.training.predictor import BeamPredictor
from symba_grpo.rng import stable_seed


GP_ROWS = (0, 10_000)
POLICY_ROWS = (10_000, 20_000)
REPORTING_ROWS = (20_000, 120_000)
PAD_INDEX = 0
NUMBER_INDEX = 1

FEYNMAN_LOCALS = {
    "pi": sp.pi,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "exp": sp.exp,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "tanh": sp.tanh,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "arctan": sp.atan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
}


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _variable_names(row: pd.Series, n_variables: int) -> list[str]:
    names = []
    for index in range(1, 12):
        value = row.get(f"v{index}_name")
        if pd.notna(value) and str(value).strip():
            names.append(str(value).strip())
    return names if len(names) == n_variables else [f"x{i + 1}" for i in range(n_variables)]


def load_benchmark(data_dir: str | Path, equations: tuple[str, ...]) -> dict[str, dict]:
    """Load only the frozen equations and construct three non-overlapping panels."""
    data_dir = Path(data_dir)
    table = pd.read_csv(data_dir / "FeynmanEquations.csv", keep_default_na=False)
    table["Filename"] = table["Filename"].astype(str).str.strip()
    by_name = table.set_index("Filename", drop=False)
    loaded = {}
    for name in equations:
        if name not in by_name.index:
            raise ValueError(f"{name} is absent from FeynmanEquations.csv")
        raw_path = data_dir / f"{name}.npy"
        if not raw_path.is_file():
            raise FileNotFoundError(f"missing required array: {raw_path}")
        raw = np.load(raw_path, mmap_mode="r")
        if raw.ndim != 2 or raw.shape[0] < REPORTING_ROWS[1] or not np.isfinite(raw[: REPORTING_ROWS[1]]).all():
            raise ValueError(f"invalid support array for {name}: shape={raw.shape}")
        n_variables = raw.shape[1] - 1
        row = by_name.loc[name]
        names = _variable_names(row, n_variables)
        truth = sp.sympify(str(row.get("Formula", "")), locals=FEYNMAN_LOCALS)
        truth = truth.subs({sp.Symbol(variable): sp.Symbol(f"s_{i + 1}") for i, variable in enumerate(names)})
        gp = raw[slice(*GP_ROWS)]
        policy = raw[slice(*POLICY_ROWS)]
        reporting = raw[slice(*REPORTING_ROWS)]
        context_size = min(200, max(30, 50 * n_variables), len(policy))
        context = policy[:context_size]
        loaded[name] = {
            "filename": name,
            "n_vars": n_variables,
            "true_expr": truth,
            "cloud": context,
            "context_row_indices": list(range(POLICY_ROWS[0], POLICY_ROWS[0] + context_size)),
            "points": cloud_to_gp_points(gp),
            "X": policy[:, :-1],
            "y": policy[:, -1],
            "eval_X": reporting[:, :-1],
            "eval_y": reporting[:, -1],
        }
    return loaded


def add_target_noise(data: dict[str, dict], sigma_fraction: float, seed: int) -> dict[str, dict]:
    """Perturb context, GP, and policy targets with deterministic Gaussian noise.

    The symbolic truth and held-out reporting targets remain clean. Every
    proposal, fitting, reward, and selection path observes the noisy targets,
    while final numerical and exact reporting use the fixed clean endpoint.
    """
    if sigma_fraction < 0 or not np.isfinite(sigma_fraction):
        raise ValueError("sigma_fraction must be finite and non-negative")
    noisy = {}
    for equation, item in data.items():
        rng = np.random.default_rng(stable_seed(seed, equation, "target_noise", sigma_fraction))
        result = dict(item)
        context = np.asarray(item["cloud"], dtype=float).copy()
        points = [(tuple(x), float(y)) for x, y in item["points"]]
        policy_y = np.asarray(item["y"], dtype=float).copy()
        report_y = np.asarray(item["eval_y"], dtype=float).copy()
        gp_y = np.asarray([y for _, y in points], dtype=float)
        scale = float(np.std(np.concatenate([gp_y, policy_y])))
        if sigma_fraction:
            # The context is a view of the first policy rows.  Draw policy noise
            # once, then expose the same perturbed values to both transformer and
            # numerical search so the noisy dataset remains internally coherent.
            policy_noise = rng.normal(0.0, sigma_fraction * scale, size=policy_y.shape)
            policy_y += policy_noise
            context[:, -1] = policy_y[: len(context)]
            points = [(x, y + float(rng.normal(0.0, sigma_fraction * scale))) for x, y in points]
            # Reporting targets remain clean: robustness is measured on the
            # fixed held-out endpoint, while all fitted/search panels are noisy.
        result["cloud"] = context
        result["points"] = points
        result["y"] = policy_y
        result["eval_y"] = report_y
        noisy[equation] = result
    return noisy


def cloud_to_gp_points(cloud: np.ndarray) -> list:
    return [(tuple(row[:-1].tolist()), float(row[-1])) for row in cloud]


def cloud_to_model_input(cloud: np.ndarray, max_points: int, chunk_size: int, device: str, rng_seed: int):
    rng = np.random.default_rng(rng_seed)
    if len(cloud) > chunk_size:
        chosen = rng.choice(len(cloud), chunk_size, replace=False)
        cloud = cloud[chosen]
    elif len(cloud) < chunk_size:
        chosen = rng.choice(len(cloud), chunk_size - len(cloud), replace=True)
        cloud = np.concatenate([cloud, cloud[chosen]], axis=0)
    reordered = np.concatenate([cloud[:, -1:], cloud[:, :-1]], axis=1)
    source = np.full((chunk_size, max_points), PAD_INDEX, dtype=np.int64)
    numbers = np.ones((chunk_size, max_points), dtype=np.float32)
    width = min(reordered.shape[1], max_points)
    source[:, :width] = NUMBER_INDEX
    numbers[:, :width] = reordered[:, :width].astype(np.float32)
    return (
        torch.from_numpy(source).unsqueeze(0).to(device),
        torch.from_numpy(numbers).unsqueeze(0).to(device),
    )


def failure_result(filename: str, reason: str) -> dict:
    return {
        "filename": filename,
        "r2": None,
        "mse": None,
        "rmse": None,
        "mse_n": None,
        "exact_match": False,
        "exact_status": "not_scored",
        "functional_score": False,
        "status": "failure",
        "failure_reason": reason,
        "infrastructure_error": False,
        "pred_expr": "",
    }


def candidate_selection_supports(seed: int, equation: str, cycles: int, size: int = 500):
    """Deterministic arm-independent policy-panel rows for proposal selection."""
    supports, digests = [], {}
    panel_size = POLICY_ROWS[1] - POLICY_ROWS[0]
    for cycle in range(cycles):
        cycle_support = {}
        for phase in ("pre", "post"):
            key = f"{seed}:{equation}:{cycle}:{phase}".encode()
            local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
            indices = np.random.default_rng(local_seed).choice(panel_size, min(size, panel_size), replace=False)
            indices = np.asarray(indices, dtype=np.int64)
            cycle_support[phase] = indices
            digests[f"cycle{cycle + 1}_{phase}"] = hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
        supports.append(cycle_support)
    return supports, digests


def _evaluate(model, vocab, config, data: dict[str, dict], equations: tuple[str, ...], method: str):
    results = []
    trainer_class = {
        "PIGP+DPO": SymbolicDPOTrainer,
        "PIGP+GRPO": SymbolicGRPOTrainer,
        "PIGP": PIGPBaselineTrainer,
    }[method]
    score_key = {"PIGP+DPO": "dpo_r2", "PIGP+GRPO": "grpo_r2", "PIGP": "pigp_r2"}[method]
    for name in tqdm(equations, desc=method):
        item = data[name]
        trainer = None
        predictor = None
        selection_supports, support_digests = candidate_selection_supports(
            config.seed, name, config.dpo_num_cycles
        )
        rng_schedule = {
            "derivation": "uint32(first_8_bytes(SHA256(colon_join(parts))), little_endian)",
            "transformer_context": stable_seed(config.seed, name, "transformer_context"),
            "gp_per_cycle": [stable_seed(config.seed, name, cycle, "gp") for cycle in range(config.dpo_num_cycles)],
        }
        if method == "PIGP+DPO":
            rng_schedule["dpo_shuffle_per_cycle"] = [
                stable_seed(config.seed, name, cycle, method) for cycle in range(config.dpo_num_cycles)
            ]
        elif method == "PIGP+GRPO":
            rng_schedule["grpo_group_per_step"] = [
                [stable_seed(config.seed, name, cycle, method, step) for step in range(config.grpo_steps_per_cycle)]
                for cycle in range(config.dpo_num_cycles)
            ]
        # The PIGP baseline takes no policy step, so it consumes neither the DPO
        # shuffle stream nor the GRPO group stream.
        try:
            started = time.time()
            context_seed = stable_seed(config.seed, name, "transformer_context")
            source, numbers = cloud_to_model_input(
                item["cloud"], config.max_input_points, config.chunk_size, config.device, context_seed
            )
            policy = copy.deepcopy(model)
            trainer = trainer_class(policy, vocab, config)
            predictor = BeamPredictor(policy, vocab, config.device)
            _, expression, selection_r2 = trainer.training_loop(
                predictor,
                source,
                numbers,
                item["points"],
                item["n_vars"],
                num_cycles=config.dpo_num_cycles,
                X=item["X"],
                y=item["y"],
                selection_supports=selection_supports,
                experiment_key=f"{config.seed}:{name}",
            )
            trainer.accounting.update(predictor.accounting)
            if expression is None:
                failure = failure_result(name, "no valid expression")
                failure.update(
                    candidate_selection_support_sha256=support_digests,
                    rng_schedule=rng_schedule,
                    training_accounting=trainer.accounting,
                )
                results.append(failure)
                continue
            result = evaluate_expression(expression, item["true_expr"], item["eval_X"], item["eval_y"])
            replayable_prediction = prediction_record(expression)
            result.update(
                filename=name,
                pred_expr=str(expression),
                prediction_record=replayable_prediction,
                prediction_ir_version=replayable_prediction["version"],
                prediction_source=replayable_prediction["source"],
                true_expr=str(item["true_expr"]),
                **{score_key: selection_r2},
                candidate_selection_support_sha256=support_digests,
                rng_schedule=rng_schedule,
                training_accounting=trainer.accounting,
            )
            logger.info(
                "{:<12} {} R2={:.6f} exact={} elapsed={:.1f}s",
                name,
                method,
                result["r2"],
                result["exact_match"],
                time.time() - started,
            )
            results.append(result)
        except Exception as error:
            logger.exception("{} failed under {}", name, method)
            failure = failure_result(name, f"{type(error).__name__}: {error}")
            failure["infrastructure_error"] = True
            accounting = dict(getattr(trainer, "accounting", {}))
            if predictor is not None:
                accounting.update(predictor.accounting)
            failure.update(
                candidate_selection_support_sha256=support_digests,
                rng_schedule=rng_schedule,
                training_accounting=accounting,
            )
            results.append(failure)
    return results, evaluate_batch(results)


def evaluate_dpo(model, vocab, config, data, equations):
    return _evaluate(model, vocab, config, data, equations, "PIGP+DPO")


def evaluate_grpo(model, vocab, config, data, equations):
    return _evaluate(model, vocab, config, data, equations, "PIGP+GRPO")


def evaluate_pigp(model, vocab, config, data, equations):
    return _evaluate(model, vocab, config, data, equations, "PIGP")
