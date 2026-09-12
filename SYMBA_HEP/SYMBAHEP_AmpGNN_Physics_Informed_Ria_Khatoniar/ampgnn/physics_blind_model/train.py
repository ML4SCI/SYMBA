#!/usr/bin/env python3
"""Train the physics-blind baseline.

Mirrors `ampgnn.train` for everything that should be held constant (data split,
targets, decoder, losses, metrics, optimiser) and differs only in that the input
carries no physics: symbol-identity graph features, no cross edges, no anchor
distances, no leg-permutation augmentation, no physics decode mask.

Example:
  python -m ampgnn.physics_blind_model.train \
    --model QED --data_dir ampgnn/data_july2026 \
    --out_dir ampgnn/runs/qed_2to2_blind --ranks 2_to_2 \
    --epochs 200 --label_smoothing 0.03 --gradient_clip_val 1.0 --weight_decay 0.03
"""

from __future__ import annotations

import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from ..train import CompactEpochPrinter, GraphExprDataModule, _parse_model_list
from .data import MAX_PERMS, build_blind_datasets
from .model import LitPhysicsBlind


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="'QED', 'QCD', 'EW', a comma-separated list, or 'all'.")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--orders", default="tree")
    ap.add_argument("--ranks", default=None)
    ap.add_argument("--n_initial", type=int, default=2)
    ap.add_argument("--compact_header_stub", action="store_true")
    ap.add_argument("--subsample", type=int, default=None)

    ap.add_argument("--mass_rewrite", action=argparse.BooleanOptionalAction, default=False,
                    help="Rewrite masses to diagram-derived leg slots. Off by default: the "
                         "slot names are themselves a physics prior, so the baseline predicts "
                         "raw species (m_u, m_d, ...). Turn on to match a mass-rewritten run.")
    ap.add_argument("--mass_rewrite_mode", choices=["species", "leg_sets"], default="leg_sets")
    ap.add_argument("--simplify_targets", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--canonicalize_targets", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--canonicalize_eval", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max_seq_len_cap", type=int, default=1024)

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--accumulate_grad_batches", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--scheduler", default="cosine_warmup")
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--t_0", type=int, default=10)
    ap.add_argument("--t_mult", type=int, default=2)
    ap.add_argument("--eta_min", type=float, default=0.0)
    ap.add_argument("--gradient_clip_val", type=float, default=0.5)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--loss_mode", choices=["ce", "ctc"], default="ce")
    ap.add_argument("--length_loss_weight", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=None)

    ap.add_argument("--enc_hid", type=int, default=128)
    ap.add_argument("--enc_layers", type=int, default=None)
    ap.add_argument("--enc_heads", type=int, default=None)
    ap.add_argument("--enc_dropout", type=float, default=None)
    ap.add_argument("--d_model", type=int, default=None)
    ap.add_argument("--dec_nhead", type=int, default=None)
    ap.add_argument("--dec_layers", type=int, default=None)
    ap.add_argument("--dec_dropout", type=float, default=None)
    ap.add_argument("--dec_max_len", type=int, default=None)
    ap.add_argument("--dec_use_len_mask", action="store_true", default=True)
    ap.add_argument("--no-dec_use_len_mask", dest="dec_use_len_mask", action="store_false")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--devices", default="auto")
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--strategy", default="auto")
    ap.add_argument("--log_every_n_steps", type=int, default=50)
    ap.add_argument("--print_every_n_epochs", type=int, default=5)
    ap.add_argument("--enable_progress_bar", action="store_true")
    ap.add_argument("--enable_model_summary", action="store_true")
    ap.add_argument("--enable_logger", action="store_true")
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--fail_log_dir", default=None)
    ap.add_argument("--fail_log_max", type=int, default=64)
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    if args.dec_max_len is None:
        args.dec_max_len = int(args.max_seq_len_cap)
    elif args.dec_max_len < args.max_seq_len_cap:
        print(f"[WARN] --dec_max_len ({args.dec_max_len}) < --max_seq_len_cap "
              f"({args.max_seq_len_cap}); raising it to avoid truncation.")
        args.dec_max_len = int(args.max_seq_len_cap)

    pl.seed_everything(args.seed)

    orders = None if args.orders == "all" else [s.strip() for s in args.orders.split(",") if s.strip()]
    ranks = None if not args.ranks else [s.strip() for s in args.ranks.split(",") if s.strip()]
    model_list = _parse_model_list(args.model)

    print("[baseline] physics-blind: symbol-identity features, no cross edges, "
          "no anchor distances, no leg permutations, no physics decode mask.")
    if args.mass_rewrite:
        print("[baseline][WARN] --mass_rewrite is on, so targets keep physics-derived mass "
              "slots. Use --no-mass_rewrite for a fully physics-free baseline.")
    print(f"[data] training on physics models: {model_list}")

    train_ds, val_ds, test_ds, vocab = build_blind_datasets(
        model_list, args.data_dir,
        n_initial=args.n_initial,
        max_seq_len_cap=args.max_seq_len_cap,
        seed=args.seed,
        orders=orders, ranks=ranks,
        mass_rewrite=args.mass_rewrite,
        mass_rewrite_mode=args.mass_rewrite_mode,
        canonicalize_targets=args.canonicalize_targets,
        compact_header_stub=args.compact_header_stub,
        simplify_targets=args.simplify_targets,
        subsample=args.subsample,
    )

    dm = GraphExprDataModule((train_ds, val_ds, test_ds), batch_size=args.batch_size,
                             num_workers=args.num_workers, seed=args.seed,
                             max_perms=MAX_PERMS, pad_id=vocab.pad)

    graphs0, *_rest = train_ds[0]
    sample_graph = next(g for g in graphs0 if g is not None)
    in_dim = sample_graph.x.size(1)
    edge_dim = (
        sample_graph.edge_attr.size(1)
        if getattr(sample_graph, "edge_attr", None) is not None and sample_graph.edge_attr.numel() > 0
        else 1
    )
    print(f"[baseline] in_dim={in_dim} edge_dim={edge_dim} vocab={len(vocab.stoi)}")

    lit = LitPhysicsBlind(
        in_dim=in_dim, edge_dim=edge_dim, vocab_size=len(vocab.stoi),
        pad_id=vocab.pad, bos_id=vocab.bos, eos_id=vocab.eos,
        lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler,
        warmup_steps=args.warmup_steps, t_0=args.t_0, t_mult=args.t_mult, eta_min=args.eta_min,
        loss_mode=args.loss_mode, label_smoothing=args.label_smoothing,
        length_loss_weight=args.length_loss_weight, max_steps=args.max_steps,
        enc_hid=args.enc_hid, enc_layers=args.enc_layers, enc_heads=args.enc_heads,
        enc_dropout=args.enc_dropout, d_model=args.d_model, dec_nhead=args.dec_nhead,
        dec_layers=args.dec_layers, dec_dropout=args.dec_dropout,
        dec_max_len=args.dec_max_len, dec_use_len_mask=args.dec_use_len_mask,
        failure_log_dir=args.fail_log_dir, failure_log_max=args.fail_log_max,
    )
    lit.attach_vocab(vocab)
    lit.canonicalize_commutative = bool(args.canonicalize_eval)

    os.makedirs(args.out_dir, exist_ok=True)
    model_tag = "_".join(model_list) if len(model_list) > 1 else model_list[0]
    cbs = [
        ModelCheckpoint(
            dirpath=args.out_dir,
            filename=f"{model_tag}-blind" + "-{epoch:02d}-{val_seq_acc:.4f}",
            save_top_k=1, monitor="val/seq_acc", mode="max", save_last=True,
            auto_insert_metric_name=False,
        )
    ]
    if args.print_every_n_epochs > 0:
        cbs.append(CompactEpochPrinter(every_n_epochs=args.print_every_n_epochs))
    if args.enable_logger:
        cbs.append(LearningRateMonitor(logging_interval="step"))
    if args.early_stop:
        cbs.append(EarlyStopping(monitor="val/seq_acc", patience=args.patience, mode="max"))

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        default_root_dir=args.out_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=cbs,
        enable_checkpointing=True,
        enable_progress_bar=args.enable_progress_bar,
        enable_model_summary=args.enable_model_summary,
        logger=args.enable_logger,
        deterministic=False,
    )

    if args.test_only:
        trainer.test(lit, datamodule=dm, ckpt_path=args.resume_from or "best")
        return

    trainer.fit(lit, datamodule=dm, ckpt_path=args.resume_from)
    trainer.test(lit, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
