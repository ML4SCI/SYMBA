"""Command-line runner for the paired DPO/GRPO experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from symba_grpo.benchmark import TEST_EQUATIONS
from symba_grpo.config import Config
from symba_grpo.experiment import (
    GP_ROWS,
    POLICY_ROWS,
    REPORTING_ROWS,
    add_target_noise,
    evaluate_dpo,
    evaluate_grpo,
    evaluate_pigp,
    load_benchmark,
    set_seeds,
)
from symba_grpo.models.seq2seq import build_model
from symba_grpo.tokenizers.vocab import get_default_decoder_vocab


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def load_sha256_manifest(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        entries[filename.strip()] = digest
    return entries


def canonical_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def protocol_settings(config: Config) -> dict:
    excluded = {"seed", "data_dir", "output_dir", "device"}
    return {key: value for key, value in asdict(config).items() if key not in excluded}


def asset_key(path: Path, repository_root: Path | None = None) -> str:
    """Return the repository-relative manifest key for an asset path."""
    root = (repository_root or Path.cwd()).resolve()
    return str(path.resolve().relative_to(root))


def data_hashes(data_dir: Path, equations: tuple[str, ...]) -> dict[str, str]:
    paths = [data_dir / "FeynmanEquations.csv"]
    paths.extend(data_dir / f"{name}.npy" for name in equations)
    return {asset_key(path): file_sha256(path) for path in paths}


def load_model(config: Config, checkpoint: Path):
    payload = torch.load(checkpoint, map_location=config.device, weights_only=True)
    model = build_model(config)
    model.load_state_dict(payload.get("state_dict", payload))
    return model.to(config.device).eval()


def _json_safe(value):
    if isinstance(value, float) and not (float("-inf") < value < float("inf")):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def assert_smoke_has_no_infrastructure_errors(method: str, rows: list[dict]) -> None:
    """Make a smoke run fail loudly on unexpected implementation exceptions."""
    failures = [row.get("filename", "<unknown>") for row in rows if row.get("infrastructure_error") is True]
    if failures:
        raise RuntimeError(f"infrastructure failure in {method} smoke run: {failures}")


def run(
    seed: int,
    data_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    source_manifest: Path,
    assets_manifest: Path,
    smoke: bool = False,
    noise_sigma: float = 0.0,
) -> Path:
    if seed not in (42, 43, 44, 45, 46, 47):
        raise ValueError("valid seeds are 42--47")
    vocab = get_default_decoder_vocab()
    config = Config(seed=seed, data_dir=str(data_dir), output_dir=str(output_dir))
    config.tgt_vocab_size = vocab.size
    equations = TEST_EQUATIONS[:1] if smoke else TEST_EQUATIONS
    if smoke:
        config.pop_size = 8
        config.num_generations = 1
        config.dpo_num_cycles = 1
        config.dpo_epochs_per_cycle = 1
        config.grpo_steps_per_cycle = 1
        config.grpo_group_size = 2

    expected_sources = {
        name.removeprefix("symba_grpo/"): digest
        for name, digest in load_sha256_manifest(source_manifest).items()
    }
    observed_sources = source_hashes()
    if observed_sources != expected_sources:
        raise RuntimeError("source manifest does not match imported package")
    expected_assets = load_sha256_manifest(assets_manifest)
    checkpoint_key = asset_key(checkpoint)
    if expected_assets.get(checkpoint_key) != file_sha256(checkpoint):
        raise RuntimeError("checkpoint is absent from or mismatched with the asset manifest")
    observed_data = data_hashes(data_dir, equations)
    if any(expected_assets.get(name) != digest for name, digest in observed_data.items()):
        raise RuntimeError("benchmark data are absent from or mismatched with the asset manifest")

    data = load_benchmark(data_dir, equations)
    if noise_sigma:
        data = add_target_noise(data, noise_sigma, seed)
    methods = {}
    arms = (("PIGP+DPO", evaluate_dpo), ("PIGP+GRPO", evaluate_grpo), ("PIGP", evaluate_pigp))
    for name, evaluator in arms:
        set_seeds(seed)
        model = load_model(config, checkpoint)
        started = time.time()
        rows, aggregate = evaluator(model, vocab, config, data, equations)
        if len(rows) != len(equations) or {row["filename"] for row in rows} != set(equations):
            raise RuntimeError(f"coverage failure for {name}")
        if smoke:
            assert_smoke_has_no_infrastructure_errors(name, rows)
        methods[name] = {
            "per_equation": rows,
            "aggregate": aggregate,
            "elapsed_sec": time.time() - started,
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    context_rows = {name: data[name]["context_row_indices"] for name in equations}
    settings = protocol_settings(config)
    panels = {
        "gp_rows": list(GP_ROWS),
        "policy_rows": list(POLICY_ROWS),
        "transformer_context_row_indices": context_rows,
        "group_reward_support": "one uniform sample within policy rows per group",
        "reporting_rows": list(REPORTING_ROWS),
    }
    accounting = {
        "beam_banks_per_cycle_per_method": 2,
        "gp_searches_per_cycle_per_method": 1,
        "pre_update_bank": "GP seeding and direct policy-panel selection",
        "post_update_bank": "direct policy-panel selection",
        "maximum_candidates_per_decode": config.num_equations,
        "gp_population": config.pop_size,
        "grpo_group_trajectories_per_cycle": config.grpo_group_size * config.grpo_steps_per_cycle,
        "grpo_policy_passes_per_group": config.grpo_policy_epochs,
        "compute_matching_scope": "beam banks, GP searches, GP population, and numerical panels",
    }
    protocol_name = "smoke" if smoke else (f"noise_sigma_{noise_sigma:g}" if noise_sigma else "frozen10_paired_v1")
    protocol_record = {
        "protocol": protocol_name,
        "equations": list(equations),
        "methods": [arm_name for arm_name, _ in arms],
        "settings": settings,
        "data_panels": panels,
        "search_accounting": accounting,
    }
    payload = {
        "protocol": protocol_name,
        "seed": seed,
        "test_equations": list(equations),
        "checkpoint_sha256": file_sha256(checkpoint),
        "source_sha256": observed_sources,
        "source_manifest_sha256": file_sha256(source_manifest),
        "assets_manifest_sha256": file_sha256(assets_manifest),
        "data_sha256": observed_data,
        "protocol_sha256": canonical_sha256(protocol_record),
        "settings": settings,
        "data_panels": panels,
        "search_accounting": accounting,
        "environment": {"device": config.device},
        "methods": methods,
    }
    if smoke:
        destination = output_dir / "smoke" / f"seed{seed}.json"
    elif noise_sigma:
        destination = output_dir / "noise" / f"sigma{noise_sigma:g}" / f"seed{seed}.json"
    else:
        destination = output_dir / "per_seed" / f"seed{seed}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=Path("manifests/source.sha256"))
    parser.add_argument("--assets-manifest", type=Path, default=Path("manifests/assets.sha256"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--noise-sigma", type=float, default=0.0)
    args = parser.parse_args()
    print(run(
        args.seed, args.data_dir, args.output_dir, args.checkpoint,
        args.source_manifest, args.assets_manifest, args.smoke, args.noise_sigma,
    ))


if __name__ == "__main__":
    main()
