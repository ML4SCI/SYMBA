"""Physics-informed token constraints for parallel token decoding."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence

import torch

_MASS_TOKEN_RE = re.compile(r"^M(i?[A-Za-z0-9]+)?$")


def is_mass_token(tok: str) -> bool:
    return bool(_MASS_TOKEN_RE.match(tok))


def build_batch_allow_mask(
    meta_list: Sequence[Dict[str, Any]],
    vocab,
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    """Return [B, V] bool mask: True = token allowed for that example."""
    mode = str(mode).lower().strip()
    vocab_size = len(vocab.stoi)
    allow = torch.ones(len(meta_list), vocab_size, dtype=torch.bool, device=device)

    if mode in ("", "none"):
        return allow

    mass_token_ids = {tok: tid for tok, tid in vocab.stoi.items() if is_mass_token(tok)}

    for b, meta in enumerate(meta_list):
        if mode == "masses":
            allowed_masses = set(meta.get("slot_map", {}).values())
            for tok, tid in mass_token_ids.items():
                if tok not in allowed_masses:
                    allow[b, tid] = False

    return allow


def apply_physics_allowlist(
    logits: torch.Tensor,
    meta_list: Optional[Sequence[Dict[str, Any]]],
    vocab,
    mode: str,
) -> torch.Tensor:
    """Zero out logits for physics-disallowed tokens. Used in train/val/test alike."""
    mode = str(mode).lower().strip()
    if mode in ("", "none") or meta_list is None or len(meta_list) == 0:
        return logits

    allow = build_batch_allow_mask(meta_list, vocab, mode, logits.device)
    return logits.masked_fill(~allow.unsqueeze(1), -1e4)
