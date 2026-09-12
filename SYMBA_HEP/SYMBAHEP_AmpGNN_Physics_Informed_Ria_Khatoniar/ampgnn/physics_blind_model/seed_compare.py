#!/usr/bin/env python3
"""Compare the physics-informed model against the physics-blind baseline across seeds.

The `physics` arm runs the unmodified `ampgnn.train`; the `blind` arm runs
`ampgnn.physics_blind_model.train`. The seed drives both weight init and the
stratified train/val/test split, so the spread across seeds covers split
variance as well as optimisation noise. On small test splits that variance is
substantial, so prefer the mean over seeds to any single run.

Example:
  python -m ampgnn.physics_blind_model.seed_compare --ranks 2_to_2 --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data_july2026"
DEFAULT_OUT = ROOT / "runs" / "baseline_seed_compare"

ARMS = ("physics", "blind")

METRIC_RES = {
    "test_seq_acc": re.compile(r"test/seq_acc\s+([0-9.]+)"),
    "test_token_acc": re.compile(r"test/token_acc\s+([0-9.]+)"),
    "test_loss": re.compile(r"test/loss\s+([0-9.]+)"),
    "val_seq_acc": re.compile(r"val_seq_acc=([0-9.]+)"),
    "val_token_acc": re.compile(r"val_token_acc=([0-9.]+)"),
}


def parse_metrics(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, rx in METRIC_RES.items():
        matches = rx.findall(text)
        if matches:
            out[name] = float(matches[-1])
    return out


def build_cmd(arm: str, seed: int, out_dir: Path, args: argparse.Namespace) -> List[str]:
    entry = "ampgnn.train" if arm == "physics" else "ampgnn.physics_blind_model.train"
    cmd = [
        sys.executable, "-m", entry,
        "--model", args.model,
        "--data_dir", str(args.data_dir),
        "--out_dir", str(out_dir),
        "--orders", "tree",
        "--ranks", args.ranks,
        "--simplify_targets",
        "--no-canonicalize_targets",
        "--canonicalize_eval",
        "--dec_use_len_mask",
        "--length_loss_weight", str(args.length_loss_weight),
        "--loss_mode", "ce",
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--warmup_steps", str(args.warmup_steps),
        "--scheduler", "cosine_warmup",
        "--gradient_clip_val", str(args.gradient_clip_val),
        "--label_smoothing", str(args.label_smoothing),
        "--max_seq_len_cap", str(args.max_seq_len_cap),
        "--dec_max_len", str(args.dec_max_len),
        "--accelerator", args.accelerator,
        "--devices", str(args.devices),
        "--precision", args.precision,
        "--num_workers", str(args.num_workers),
        "--print_every_n_epochs", str(args.print_every_n_epochs),
        "--seed", str(seed),
        "--fail_log_dir", str(out_dir / "failures"),
    ]
    if arm == "physics":
        cmd += [
            "--mass_rewrite_mode", "leg_sets",
            "--phys_decode", "masses",
            "--max_perms", str(args.max_perms),
            "--no-auto_resume",
        ]
    else:
        cmd += ["--no-mass_rewrite"]
    return cmd


def run_one(arm: str, seed: int, args: argparse.Namespace, results_dir: Path) -> Dict[str, Any]:
    run_id = f"{arm}_seed{seed}"
    out_dir = results_dir / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_cmd(arm, seed, out_dir, args)

    env = os.environ.copy()
    parent = str(ROOT.parent)
    env["PYTHONPATH"] = parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    t0 = time.time()
    print(f"\n[{datetime.now():%H:%M:%S}] START {run_id}", flush=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write("# " + " ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT.parent), env=env,
                              stdout=logf, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    metrics = parse_metrics(log_path.read_text(encoding="utf-8", errors="replace"))

    rec: Dict[str, Any] = {
        "run_id": run_id,
        "arm": arm,
        "seed": seed,
        **metrics,
        "exit_code": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "out_dir": str(out_dir),
        "log_path": str(log_path),
        "ok": proc.returncode == 0 and "test_seq_acc" in metrics,
    }
    status = "OK" if rec["ok"] else f"FAIL(exit={proc.returncode})"
    print(
        f"[{datetime.now():%H:%M:%S}] DONE  {run_id}  {status}  "
        f"test_seq={metrics.get('test_seq_acc')}  test_tok={metrics.get('test_token_acc')}  "
        f"({elapsed/60:.1f} min)",
        flush=True,
    )
    return rec


def _mean_std(values: List[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f} +/- {statistics.stdev(values):.4f}"


def write_summary(results: List[Dict[str, Any]], results_dir: Path) -> None:
    csv_path = results_dir / "summary.csv"
    fields = ["arm", "seed", "test_seq_acc", "test_token_acc", "test_loss",
              "val_seq_acc", "val_token_acc", "elapsed_sec", "ok", "exit_code", "out_dir"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(results, key=lambda r: (r.get("arm", ""), r.get("seed", 0))):
            w.writerow(r)

    print("\n=== SUMMARY (mean +/- stdev over seeds) ===", flush=True)
    for arm in ARMS:
        rows = [r for r in results if r.get("arm") == arm and r.get("ok")]
        if not rows:
            continue
        seqs = [float(r["test_seq_acc"]) for r in rows]
        toks = [float(r["test_token_acc"]) for r in rows]
        print(f"  {arm:8s} n={len(rows)}  "
              f"test_seq_acc={_mean_std(seqs)}  test_token_acc={_mean_std(toks)}", flush=True)

    phys = [float(r["test_seq_acc"]) for r in results if r.get("arm") == "physics" and r.get("ok")]
    blind = [float(r["test_seq_acc"]) for r in results if r.get("arm") == "blind" and r.get("ok")]
    if phys and blind:
        print(f"\n  physics - blind (seq acc): "
              f"{statistics.mean(phys) - statistics.mean(blind):+.4f}", flush=True)
    print(f"\nWrote {csv_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--results_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", default="QED")
    ap.add_argument("--ranks", default="2_to_2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--gradient_clip_val", type=float, default=0.5)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--max_perms", type=int, default=6)
    ap.add_argument("--max_seq_len_cap", type=int, default=1024)
    ap.add_argument("--dec_max_len", type=int, default=1024)
    ap.add_argument("--length_loss_weight", type=float, default=1.0)
    ap.add_argument("--accelerator", default="gpu")
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--print_every_n_epochs", type=int, default=25)
    ap.add_argument("--skip_done", action=argparse.BooleanOptionalAction, default=True,
                    help="Skip runs that already have an OK row in results.jsonl.")
    args = ap.parse_args()

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_dir / "results.jsonl"

    print(f"Arms: {args.arms}  seeds: {args.seeds}  "
          f"{args.model} {args.ranks}  epochs={args.epochs}", flush=True)

    done_ids = set()
    results: List[Dict[str, Any]] = []
    if args.skip_done and jsonl_path.is_file():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            results.append(rec)
            if rec.get("ok"):
                done_ids.add(rec["run_id"])
        print(f"Resuming: {len(done_ids)} finished runs already in {jsonl_path}", flush=True)

    with open(jsonl_path, "a", encoding="utf-8") as jf:
        for seed in args.seeds:
            for arm in args.arms:
                if f"{arm}_seed{seed}" in done_ids:
                    print(f"SKIP {arm}_seed{seed} (already done)", flush=True)
                    continue
                rec = run_one(arm, seed, args, results_dir)
                results.append(rec)
                jf.write(json.dumps(rec) + "\n")
                jf.flush()

    write_summary(results, results_dir)


if __name__ == "__main__":
    main()
