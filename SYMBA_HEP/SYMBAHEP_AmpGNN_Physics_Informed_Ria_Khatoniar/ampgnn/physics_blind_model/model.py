"""Lightning module for the physics-blind baseline.

Subclasses the parent `LitAmpGNN` so the decoder, CE/CTC losses, length head,
accuracy metrics, failure logging and optimiser/scheduler are bit-for-bit the
same. The only change is the encoder: anchor-distance features (hop distances to
the amplitude-side external legs) are physics-motivated, so they are removed and
the encoder is left with plain graph topology.
"""

from __future__ import annotations

from typing import Literal, Optional

from ..model import AmpGNN, LitAmpGNN


class LitPhysicsBlind(LitAmpGNN):
    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        vocab_size: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        lr: float,
        weight_decay: float,
        scheduler: str,
        warmup_steps: int,
        t_0: int = 10,
        t_mult: int = 2,
        eta_min: float = 0.0,
        loss_mode: Literal["ce", "ctc"] = "ce",
        label_smoothing: float = 0.0,
        length_loss_weight: float = 1.0,
        max_steps: Optional[int] = None,
        failure_log_dir: Optional[str] = None,
        failure_log_max: int = 64,
        enc_hid: int = 128,
        enc_layers: Optional[int] = None,
        enc_heads: Optional[int] = None,
        enc_dropout: Optional[float] = None,
        d_model: Optional[int] = None,
        dec_nhead: Optional[int] = None,
        dec_layers: Optional[int] = None,
        dec_dropout: Optional[float] = None,
        dec_max_len: Optional[int] = None,
        dec_use_len_mask: bool = False,
    ):
        # LitAmpGNN always builds its encoder with anchor distances enabled, and
        # GraphEncoder refuses to build without an anchor count, so pass a
        # placeholder and swap in the anchor-free network below.
        super().__init__(
            in_dim=in_dim, edge_dim=edge_dim, vocab_size=vocab_size, pad_id=pad_id,
            bos_id=bos_id, eos_id=eos_id, lr=lr, weight_decay=weight_decay,
            scheduler=scheduler, warmup_steps=warmup_steps, t_0=t_0, t_mult=t_mult,
            eta_min=eta_min, loss_mode=loss_mode, label_smoothing=label_smoothing,
            length_loss_weight=length_loss_weight, max_steps=max_steps,
            failure_log_dir=failure_log_dir, failure_log_max=failure_log_max,
            enc_hid=enc_hid, enc_layers=enc_layers, enc_heads=enc_heads,
            enc_dropout=enc_dropout, d_model=d_model, dec_nhead=dec_nhead,
            dec_layers=dec_layers, dec_dropout=dec_dropout, dec_max_len=dec_max_len,
            dec_use_len_mask=dec_use_len_mask,
            num_anchors=1, rank_anchor_dims=None,
        )
        self.save_hyperparameters()
        self.model = AmpGNN(
            in_dim=in_dim, edge_dim=edge_dim, vocab_size=vocab_size, pad_id=pad_id,
            enc_hid=enc_hid, enc_layers=enc_layers, enc_heads=enc_heads,
            enc_dropout=enc_dropout, d_model=d_model, dec_nhead=dec_nhead,
            dec_layers=dec_layers, dec_dropout=dec_dropout, dec_max_len=dec_max_len,
            dec_use_len_mask=dec_use_len_mask,
            enc_add_node_pos=False,
            enc_add_anchor_dists=False,
            enc_num_anchors=None,
            enc_rank_anchor_dims=None,
        )
        # The baseline never applies the physics allowlist at decode time.
        self.phys_decode_mode = "none"
