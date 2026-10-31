"""Frozen experiment configuration."""

from dataclasses import dataclass

import torch


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    seed: int
    data_dir: str
    output_dir: str
    device: str | None = None
    chunk_size: int = 400
    max_input_points: int = 11
    embedding_size: int = 64
    nhead: int = 8
    num_encoder_layers: int = 2
    num_decoder_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    input_emb_size: int = 64
    src_vocab_size: int = 3
    tgt_vocab_size: int = 59
    pop_size: int = 500
    cxpb: float = 0.7
    mutpb: float = 0.2
    num_generations: int = 30
    gp_verbose: bool = False
    beam_size: int = 5
    num_equations: int = 20
    dpo_beta: float = 0.01
    dpo_beta_multiplier: float = 5.0
    dpo_num_cycles: int = 2
    dpo_lr: float = 1e-4
    dpo_epochs_per_cycle: int = 6
    dpo_top_n: int = 10
    dpo_compare_n: int = 100
    dpo_batch_size: int = 16
    grpo_group_size: int = 16
    grpo_clip_eps: float = 0.2
    grpo_temperature: float = 1.0
    grpo_lr: float = 1e-4
    grpo_steps_per_cycle: int = 6
    grpo_policy_epochs: int = 2
    grpo_max_len: int = 64

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = get_device()
