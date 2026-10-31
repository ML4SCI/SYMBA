"""Encoder-decoder seq2seq transformer for symbolic regression.

Ported from Aryamaan's xval_transformers/model/seq2seq.py with support
for both xVal (continuous) and vanilla (P1000 discrete) input modes.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.pos_embedding[:, :x.size(1), :])

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        legacy_key = f"{prefix}pe"
        expected_key = f"{prefix}pos_embedding"
        if expected_key not in state_dict and legacy_key in state_dict:
            state_dict[expected_key] = state_dict.pop(legacy_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class LinearPointEmbedder(nn.Module):
    """Embeds xVal-style point data: token embeddings scaled by num_array."""

    def __init__(self, vocab_size: int, input_emb_size: int, emb_size: int,
                 max_input_points: int, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_emb_size)
        self.input_size = max_input_points * input_emb_size
        self.fc1 = nn.Linear(self.input_size, emb_size)
        self.fc2 = nn.Linear(emb_size, emb_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, num_array: Tensor) -> Tensor:
        out = self.embedding(tokens.long())  # (B, N, max_pts, input_emb)
        bs, n = out.shape[0], out.shape[1]
        out = out * num_array.unsqueeze(-1)
        out = out.view(bs, n, -1)
        out = self.activation(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)


class Seq2SeqTransformer(nn.Module):
    """Seq2Seq transformer supporting xVal input encoding."""

    def __init__(
        self,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 6,
        emb_size: int = 64,
        nhead: int = 8,
        src_vocab_size: int = 3,
        tgt_vocab_size: int = 59,
        input_emb_size: int = 64,
        max_input_points: int = 11,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = LinearPointEmbedder(
            src_vocab_size, input_emb_size, emb_size, max_input_points, dropout
        )
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(
        self,
        src: Tensor,
        num_array: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ) -> Tensor:
        src_emb = self.src_tok_emb(src, num_array)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb, tgt_emb, src_mask, tgt_mask, None,
            src_padding_mask, tgt_padding_mask, memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(
        self, src: Tensor, num_array: Tensor,
        src_mask: Tensor, src_key_padding_mask: Tensor = None,
    ) -> Tensor:
        return self.transformer.encoder(
            self.src_tok_emb(src, num_array), src_mask, src_key_padding_mask,
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor) -> Tensor:
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )


def create_masks(src, tgt, pad_idx=0, device="cpu"):
    """Create source and target masks matching Aryamaan's exact implementation.

    src_mask:         (S, S) bool all-False (no encoder self-attn masking)
    tgt_mask:         (T, T) bool causal mask (True = masked)
    src_padding_mask: (B, S) bool all-False (no encoder padding masking)
    tgt_padding_mask: (B, T) bool (True where tgt == pad_idx)
    """
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = torch.triu(
        torch.ones((tgt_seq_len, tgt_seq_len), device=device, dtype=torch.bool),
        diagonal=1,
    )

    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).bool()
    src_padding_mask = torch.zeros((src.shape[0], src_seq_len), device=device).bool()
    tgt_padding_mask = (tgt == pad_idx)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def build_model(config) -> nn.Module:
    """Build a Seq2SeqTransformer from Config."""
    return Seq2SeqTransformer(
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        emb_size=config.embedding_size,
        nhead=config.nhead,
        src_vocab_size=config.src_vocab_size,
        tgt_vocab_size=config.tgt_vocab_size,
        input_emb_size=config.input_emb_size,
        max_input_points=config.max_input_points,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    )
