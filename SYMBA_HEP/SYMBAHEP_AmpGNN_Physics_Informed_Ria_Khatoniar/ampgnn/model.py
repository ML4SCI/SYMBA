from typing import Optional, Dict, Tuple, Literal, List, Any
import os, json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import TransformerConv, GlobalAttention
from torch_geometric.utils import degree

from .physics_decode import apply_physics_allowlist
from .expr_canonical import safe_canonicalize_tokens
from .term_slot_objective import structured_slot_objective


class GraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hid: int = 128,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        struct_pe: Literal["none", "deg"] = "deg",
        pr_alpha: float = 0.85,
        add_node_pos: bool = True,
        add_anchor_dists: bool = False,
        num_anchors: Optional[int] = None,
        rank_anchor_dims: Optional[Dict[str, int]] = None,
        cross_edge_idx: int = -1,
        external_flag_idx: int = 0,
        side_idx: int = -1,
        anchor_exclude_cross: bool = True,
        anchor_max_hops: Optional[int] = None,
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, hid)
        self.struct_pe = struct_pe
        self.pr_alpha = pr_alpha
        self.add_node_pos = add_node_pos
        self.add_anchor_dists = add_anchor_dists
        self.num_anchors = num_anchors
        self.rank_anchor_dims = dict(rank_anchor_dims) if rank_anchor_dims else None
        self.cross_edge_idx = cross_edge_idx
        self.external_flag_idx = external_flag_idx
        self.side_idx = side_idx
        self.anchor_exclude_cross = anchor_exclude_cross
        self.anchor_max_hops = anchor_max_hops


        self.deg_proj = nn.Linear(1, hid) if self.struct_pe != "none" else None


        self.anchor_projs: Optional[nn.ModuleDict] = None
        self.anchor_proj: Optional[nn.Linear] = None
        if self.add_anchor_dists:
            if self.rank_anchor_dims is not None:
                self.anchor_projs = nn.ModuleDict({
                    rk: nn.Linear(int(k), hid) for rk, k in self.rank_anchor_dims.items()
                })
            else:
                if self.num_anchors is None or self.num_anchors <= 0:
                    raise ValueError(
                        "num_anchors must be set (>0) when add_anchor_dists=True "
                        "and rank_anchor_dims is not provided."
                    )
                self.anchor_proj = nn.Linear(int(self.num_anchors), hid)

        self.layers = nn.ModuleList([
            TransformerConv(
                hid, hid // heads, heads=heads,
                edge_dim=edge_dim, dropout=dropout,
                beta=True
            ) for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.pool_gate = nn.Sequential(nn.Linear(hid, 1), nn.Sigmoid())
        self.pool = GlobalAttention(self.pool_gate)

    def _anchor_contribution(self, anchor_dists: torch.Tensor,
                             rank_key: Optional[str]) -> torch.Tensor:
        if self.anchor_projs is not None:
            if rank_key is None or rank_key not in self.anchor_projs:
                raise KeyError(
                    f"GraphEncoder was built with per-rank anchor projections for "
                    f"{sorted(self.anchor_projs.keys())} but got rank_key={rank_key!r}"
                )
            return self.anchor_projs[rank_key](anchor_dists)
        assert self.anchor_proj is not None
        return self.anchor_proj(anchor_dists)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                batch: torch.Tensor, anchor_dists: torch.Tensor | None = None,
                rank_key: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        N = x.size(0)
        h = self.proj(x)

        if self.deg_proj is not None:
            logdeg = torch.log1p(degree(edge_index[0], N, dtype=x.dtype)).unsqueeze(-1)
            h = h + self.deg_proj(logdeg)

        if self.add_anchor_dists:
            if anchor_dists is None:
                raise ValueError("anchor_dists is required when add_anchor_dists=True")
            h = h + self._anchor_contribution(anchor_dists.to(x.dtype), rank_key)

        for conv, norm in zip(self.layers, self.norms):
            h_res = h
            h = conv(h, edge_index, edge_attr)
            h = F.gelu(h)
            h = norm(h + h_res)
        g = self.pool(h, batch)
        return h, g


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()

        assert dim % 2 == 0, "RoPE head_dim must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        cos = freqs.cos().to(dtype)[None, None, :, :]
        sin = freqs.sin().to(dtype)[None, None, :, :]
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd  = x[..., 1::2]
    x_even_rot = x_even * cos - x_odd * sin
    x_odd_rot  = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., ::2] = x_even_rot
    out[..., 1::2] = x_odd_rot
    return out


class CausalBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0
        assert self.head_dim % 2 == 0, "RoPE requires even head_dim"
        self.causal = causal

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.head_dim)

    def _split_heads(self, x):
        B, T, D = x.shape
        x = x.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        return x

    def _merge_heads(self, x):
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        T = x.size(1)


        cos, sin = self.rope.get_cos_sin(seq_len=T, device=x.device, dtype=x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)


        attn_mask = None
        is_causal = self.causal
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.to(dtype=torch.bool)

            attn_mask = (~key_padding_mask)[:, None, None, :]

            if self.causal and T > 0:
                causal_keep = torch.ones((T, T), device=x.device, dtype=torch.bool).tril()
                attn_mask = attn_mask & causal_keep
            is_causal = False


        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        attn = self._merge_heads(attn)
        x = x + self.dropout(self.proj(attn))


        h = self.ln2(x)
        x = x + self.dropout(self.mlp(h))
        return x


class TokenDecoder(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.d_model = d_model
        self.max_len = max_len


        self.graph_proj = nn.Linear(d_model, d_model)


        self.pos_emb = nn.Embedding(max_len, d_model)


        self.blocks = nn.ModuleList([
            CausalBlock(d_model, nhead, mlp_ratio=4.0, dropout=dropout, causal=False)
            for _ in range(num_layers)
        ])


        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(d_model)
        self.to_logits = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        g: torch.Tensor,
        seq_len: int,
        mem: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, D = g.shape
        T = int(min(seq_len, self.max_len))
        if T <= 0:

            return g.new_zeros((B, 0, self.vocab_size))

        device = g.device


        base = self.graph_proj(g).unsqueeze(1)


        tok_key_padding_mask = None
        if lengths is not None:
            lengths = lengths.to(device=device)
            if lengths.dim() != 1 or lengths.numel() != B:
                raise ValueError(f"lengths must have shape [B] (got {tuple(lengths.shape)})")
            lengths = lengths.to(dtype=torch.long).clamp(min=1, max=T)
            pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            tok_key_padding_mask = pos >= lengths.unsqueeze(1)


        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x = base + self.pos_emb(pos_ids)


        mem_nodes = None
        mem_key_padding_mask = None
        if mem is not None:
            H_all, B_all = mem
            if H_all is not None and H_all.numel() > 0:

                H_all = H_all.to(device=device, dtype=x.dtype)
                B_all = B_all.to(device=device)


                B_batch = B
                sorted_idx = torch.argsort(B_all)
                H_sorted = H_all[sorted_idx]
                B_sorted = B_all[sorted_idx]

                counts = torch.bincount(B_sorted, minlength=B_batch)
                max_nodes = int(counts.max().item())
                if max_nodes > 0:
                    mem_nodes = x.new_zeros(B_batch, max_nodes, self.d_model)
                    mem_key_padding_mask = torch.ones(
                        B_batch, max_nodes, dtype=torch.bool, device=device
                    )
                    start = 0
                    for b in range(B_batch):
                        c = int(counts[b].item())
                        if c == 0:
                            continue
                        end = start + c
                        mem_nodes[b, :c] = H_sorted[start:end]
                        mem_key_padding_mask[b, :c] = False
                        start = end


        for blk, xattn, xnorm in zip(self.blocks, self.cross_attn, self.cross_norms):
            x = blk(x, key_padding_mask=tok_key_padding_mask)
            if mem_nodes is not None:
                h = xnorm(x)
                attn_out, _ = xattn(
                    h,
                    mem_nodes,
                    mem_nodes,
                    key_padding_mask=mem_key_padding_mask,
                    need_weights=False,
                )
                x = x + self.dropout(attn_out)

        x = self.norm(x)
        logits = self.to_logits(x)
        return logits


class StructuredTermSlotDecoder(nn.Module):
    """Canonical slot parallel decoder over algebraic monomial/factor slots."""

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        coefficient_size: int,
        symbol_size: int,
        coupling_size: int,
        max_exponent: int,
        num_slots: int,
        num_factors: int,
        den_slots: int,
        den_terms: int,
        den_term_factors: int,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.max_exponent = int(max_exponent)
        self.num_slots = int(num_slots)
        self.num_factors = int(num_factors)
        self.den_slots = int(den_slots)
        self.den_terms = int(den_terms)
        self.den_term_factors = int(den_term_factors)
        for name in (
            "max_exponent", "num_slots", "num_factors", "den_slots",
            "den_terms", "den_term_factors",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

        self.graph_proj = nn.Linear(d_model, d_model)
        self.global_query = nn.Parameter(torch.empty(1, d_model))
        self.num_queries = nn.Parameter(torch.empty(self.num_slots, d_model))
        self.den_queries = nn.Parameter(torch.empty(self.den_slots, d_model))
        self.den_term_queries = nn.Parameter(torch.empty(self.den_terms, d_model))
        self.type_embeddings = nn.Parameter(torch.empty(4, d_model))
        self.num_factor_embeddings = nn.Parameter(torch.empty(self.num_factors, d_model))
        self.den_factor_embeddings = nn.Parameter(
            torch.empty(self.den_term_factors, d_model)
        )
        for parameter in (
            self.global_query, self.num_queries, self.den_queries,
            self.den_term_queries, self.type_embeddings,
            self.num_factor_embeddings, self.den_factor_embeddings,
        ):
            nn.init.normal_(parameter, std=0.02)

        self.blocks = nn.ModuleList([
            CausalBlock(d_model, nhead, mlp_ratio=4.0, dropout=dropout, causal=False)
            for _ in range(num_layers)
        ])
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.num_factor_norm = nn.LayerNorm(d_model)
        self.den_factor_norm = nn.LayerNorm(d_model)

        exponent_classes = self.max_exponent + 1
        self.global_coefficient = nn.Linear(d_model, coefficient_size)
        self.global_coupling = nn.Linear(d_model, coupling_size)
        self.global_power = nn.Linear(d_model, exponent_classes)
        self.num_active = nn.Linear(d_model, 2)
        self.num_coefficient = nn.Linear(d_model, coefficient_size)
        self.num_factor_active = nn.Linear(d_model, 2)
        self.num_symbol = nn.Linear(d_model, symbol_size)
        self.num_exponent = nn.Linear(d_model, exponent_classes)
        self.den_active = nn.Linear(d_model, 2)
        self.den_power = nn.Linear(d_model, exponent_classes)
        self.den_term_active = nn.Linear(d_model, 2)
        self.den_term_coefficient = nn.Linear(d_model, coefficient_size)
        self.den_factor_active = nn.Linear(d_model, 2)
        self.den_symbol = nn.Linear(d_model, symbol_size)
        self.den_exponent = nn.Linear(d_model, exponent_classes)

    def _batch_memory(self, mem, batch_size, device, dtype):
        if mem is None:
            return None, None
        H_all, B_all = mem
        if H_all is None or H_all.numel() == 0:
            return None, None
        H_all = H_all.to(device=device, dtype=dtype)
        B_all = B_all.to(device=device)
        order = torch.argsort(B_all)
        H_all, B_all = H_all[order], B_all[order]
        counts = torch.bincount(B_all, minlength=batch_size)
        width = int(counts.max().item())
        nodes = H_all.new_zeros((batch_size, width, self.d_model))
        padding = torch.ones(batch_size, width, dtype=torch.bool, device=device)
        starts = torch.cumsum(counts, dim=0) - counts
        positions = torch.arange(B_all.numel(), device=device) - torch.repeat_interleave(
            starts, counts
        )
        nodes[B_all, positions] = H_all
        padding[B_all, positions] = False
        return nodes, padding

    def forward(self, g: torch.Tensor, mem=None) -> Dict[str, torch.Tensor]:
        B = g.size(0)
        global_q = self.global_query + self.type_embeddings[0]
        num_q = self.num_queries + self.type_embeddings[1]
        den_q = self.den_queries + self.type_embeddings[2]
        den_term_q = (
            self.den_queries[:, None, :]
            + self.den_term_queries[None, :, :]
            + self.type_embeddings[3]
        ).reshape(self.den_slots * self.den_terms, self.d_model)
        queries = torch.cat((global_q, num_q, den_q, den_term_q), dim=0)
        x = self.graph_proj(g).unsqueeze(1) + queries.unsqueeze(0)

        mem_nodes, mem_padding = self._batch_memory(mem, B, g.device, x.dtype)
        for block, cross_attn, cross_norm in zip(
            self.blocks, self.cross_attn, self.cross_norms
        ):
            x = block(x)
            if mem_nodes is not None:
                h = cross_norm(x)
                attended, _ = cross_attn(
                    h, mem_nodes, mem_nodes,
                    key_padding_mask=mem_padding,
                    need_weights=False,
                )
                x = x + self.dropout(attended)
        x = self.norm(x)

        offset = 0
        h_global = x[:, offset]
        offset += 1
        h_num = x[:, offset:offset + self.num_slots]
        offset += self.num_slots
        h_den = x[:, offset:offset + self.den_slots]
        offset += self.den_slots
        h_den_terms = x[:, offset:].reshape(
            B, self.den_slots, self.den_terms, self.d_model
        )

        h_num_factors = self.num_factor_norm(
            h_num[:, :, None, :] + self.num_factor_embeddings[None, None, :, :]
        )
        h_den_factors = self.den_factor_norm(
            h_den_terms[:, :, :, None, :]
            + self.den_factor_embeddings[None, None, None, :, :]
        )
        return {
            "global_coefficient_logits": self.global_coefficient(h_global),
            "global_coupling_logits": self.global_coupling(h_global),
            "global_power_logits": self.global_power(h_global),
            "num_active_logits": self.num_active(h_num),
            "num_coefficient_logits": self.num_coefficient(h_num),
            "num_factor_active_logits": self.num_factor_active(h_num_factors),
            "num_symbol_logits": self.num_symbol(h_num_factors),
            "num_exponent_logits": self.num_exponent(h_num_factors),
            "den_active_logits": self.den_active(h_den),
            "den_power_logits": self.den_power(h_den),
            "den_term_active_logits": self.den_term_active(h_den_terms),
            "den_term_coefficient_logits": self.den_term_coefficient(h_den_terms),
            "den_factor_active_logits": self.den_factor_active(h_den_factors),
            "den_symbol_logits": self.den_symbol(h_den_factors),
            "den_exponent_logits": self.den_exponent(h_den_factors),
        }


class AmpGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        graph_feat_dim: int,
        vocab_size: int,
        pad_id: int,
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
        output_mode: Literal["sequence", "term_slots"] = "sequence",
        slot_coefficient_size: int = 0,
        slot_symbol_size: int = 0,
        slot_coupling_size: int = 0,
        slot_max_exponent: int = 8,
        slot_num_slots: int = 64,
        slot_num_factors: int = 3,
        slot_den_slots: int = 4,
        slot_den_terms: int = 5,
        slot_den_term_factors: int = 3,
        enc_struct_pe: Literal["none","deg"] = "deg",
        enc_pr_alpha: float = 0.85,
        enc_add_node_pos: bool = False,
        enc_add_anchor_dists: bool = True,
        enc_num_anchors: Optional[int] = None,
        enc_rank_anchor_dims: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.edge_dim = edge_dim
        self.graph_feat_dim = int(graph_feat_dim)
        self.output_mode = str(output_mode)
        if self.output_mode not in ("sequence", "term_slots"):
            raise ValueError(f"unsupported output_mode {output_mode!r}")
        self.dec_use_len_mask = bool(dec_use_len_mask) and self.output_mode == "sequence"


        enc_kwargs: Dict[str, object] = {}
        if enc_layers  is not None: enc_kwargs["layers"]  = enc_layers
        if enc_heads   is not None: enc_kwargs["heads"]   = enc_heads
        if enc_dropout is not None: enc_kwargs["dropout"] = enc_dropout
        enc_kwargs["struct_pe"]    = enc_struct_pe
        enc_kwargs["pr_alpha"]     = enc_pr_alpha
        enc_kwargs["add_node_pos"] = enc_add_node_pos
        enc_kwargs["add_anchor_dists"] = enc_add_anchor_dists
        enc_kwargs["num_anchors"] = enc_num_anchors
        enc_kwargs["rank_anchor_dims"] = enc_rank_anchor_dims
        self._enc_rank_anchor_dims = dict(enc_rank_anchor_dims) if enc_rank_anchor_dims else None
        enc_kwargs["cross_edge_idx"] = -1
        enc_kwargs["external_flag_idx"] = 0
        enc_kwargs["side_idx"] = -1
        enc_kwargs["anchor_exclude_cross"] = True

        self.encoder = GraphEncoder(in_dim=in_dim, edge_dim=edge_dim, hid=enc_hid, **enc_kwargs)


        D = 256 if d_model is None else int(d_model)
        self.d_model = D
        self.proj = nn.Linear(enc_hid, D)
        self.graph_phys_proj = (
            nn.Linear(self.graph_feat_dim, D) if self.graph_feat_dim > 0 else None
        )
        self.mem_proj = nn.Linear(enc_hid, D)
        self.mix_gate = nn.Linear(D, 1)
        self.mix_norm = nn.LayerNorm(D)


        dec_kwargs: Dict[str, object] = {}
        if dec_nhead   is not None: dec_kwargs["nhead"]      = dec_nhead
        if dec_layers  is not None: dec_kwargs["num_layers"] = dec_layers
        if dec_dropout is not None: dec_kwargs["dropout"]    = dec_dropout
        if dec_max_len is not None: dec_kwargs["max_len"]    = dec_max_len

        if self.output_mode == "sequence":
            self.decoder = TokenDecoder(
                vocab_size=vocab_size, pad_id=pad_id, d_model=D, **dec_kwargs
            )
            self.max_dec_len = self.decoder.max_len
            self.length_head = (
                nn.Linear(D, self.max_dec_len) if self.dec_use_len_mask else None
            )
        else:
            if min(slot_coefficient_size, slot_symbol_size, slot_coupling_size) <= 0:
                raise ValueError("term_slots requires non-empty structured vocabularies")
            self.decoder = StructuredTermSlotDecoder(
                d_model=D,
                nhead=int(dec_kwargs.get("nhead", 8)),
                num_layers=int(dec_kwargs.get("num_layers", 4)),
                dropout=float(dec_kwargs.get("dropout", 0.0)),
                coefficient_size=slot_coefficient_size,
                symbol_size=slot_symbol_size,
                coupling_size=slot_coupling_size,
                max_exponent=slot_max_exponent,
                num_slots=slot_num_slots,
                num_factors=slot_num_factors,
                den_slots=slot_den_slots,
                den_terms=slot_den_terms,
                den_term_factors=slot_den_term_factors,
            )
            self.max_dec_len = dec_max_len or 4096
            self.length_head = None

    def _encode_and_mix(
        self,
        batches_per_p,
        idx_per_p,
        mask,
        rank_key: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        device = mask.device
        B, P = mask.shape
        D = self.proj.out_features
        dest_dtype = next(self.parameters()).dtype


        G = torch.zeros(B, P, D, device=device, dtype=dest_dtype)


        H_chunks: List[torch.Tensor] = []
        B_chunks: List[torch.Tensor] = []

        for p, batch_p in enumerate(batches_per_p):
            if batch_p is None:
                continue
            batch_p = batch_p.to(device)

            x = batch_p.x
            edge_attr = getattr(batch_p, "edge_attr", None)
            if edge_attr is None:
                E = batch_p.edge_index.size(1)
                edge_attr = torch.zeros(
                    E,
                    getattr(self, "edge_dim", 1),
                    device=device,
                    dtype=x.dtype,
                )
            elif edge_attr.dtype != x.dtype:
                edge_attr = edge_attr.to(x.dtype)


            anchor_d = getattr(batch_p, "anchor_dists", None)
            h_p, g_p = self.encoder(x, batch_p.edge_index, edge_attr,
                                    batch_p.batch, anchor_dists=anchor_d,
                                    rank_key=rank_key)


            g_p = self.proj(g_p).to(dtype=dest_dtype)
            if self.graph_phys_proj is not None:
                graph_features = getattr(batch_p, "graph_features", None)
                if graph_features is None:
                    raise ValueError(
                        "graph_features is required when graph_feat_dim is non-zero"
                    )
                graph_features = graph_features.reshape(-1, self.graph_feat_dim)
                if graph_features.size(0) != g_p.size(0):
                    raise ValueError(
                        "batched graph_features rows must match encoded graph count: "
                        f"{graph_features.size(0)} != {g_p.size(0)}"
                    )
                g_p = g_p + self.graph_phys_proj(
                    graph_features.to(device=device, dtype=dest_dtype)
                )
            h_p = self.mem_proj(h_p).to(dtype=dest_dtype)


            idx = idx_per_p[p].to(device)
            G[idx, p, :] = g_p


            global_batch = idx_per_p[p][batch_p.batch].to(device)
            H_chunks.append(h_p)
            B_chunks.append(global_batch)

        if H_chunks:
            H_all = torch.cat(H_chunks, dim=0)
            B_all = torch.cat(B_chunks, dim=0)
        else:
            H_all = torch.zeros(0, D, device=device, dtype=dest_dtype)
            B_all = torch.zeros(0, dtype=torch.long, device=device)


        scores = self.mix_gate(G).squeeze(-1)
        if mask.dtype != torch.bool:
            mask_bool = mask > 0
        else:
            mask_bool = mask
        scores = scores.masked_fill(~mask_bool, float("-inf"))
        weights = torch.softmax(scores.float(), dim=1).to(G.dtype).unsqueeze(-1)
        g_mix = (G * weights).sum(dim=1)

        return g_mix, (H_all, B_all)

    @torch.no_grad()
    def encode_mix(self, batches_per_p, idx_per_p, mask, rank_key: Optional[str] = None):
        g_mix, _ = self._encode_and_mix(batches_per_p, idx_per_p, mask, rank_key=rank_key)
        return self.mix_norm(g_mix)

    @torch.no_grad()
    def encode_mix_with_memory(self, batches_per_p, idx_per_p, mask, rank_key: Optional[str] = None):
        g_mix, mem = self._encode_and_mix(batches_per_p, idx_per_p, mask, rank_key=rank_key)
        return self.mix_norm(g_mix), mem

    def forward(self, batches_per_p, idx_per_p=None, mask=None, y_in=None,
                rank_key: Optional[str] = None):
        if self.output_mode == "sequence" and y_in is None:
            raise ValueError("sequence decoding requires y_in to size the decode width")
        device = mask.device if y_in is None else y_in.device
        mask = mask.to(device)


        g_mix, mem = self._encode_and_mix(batches_per_p, idx_per_p, mask, rank_key=rank_key)
        g_mix = self.mix_norm(g_mix)


        if self.output_mode == "term_slots":
            return self.decoder(g_mix, mem=mem)

        T = y_in.size(1)
        out: Dict[str, torch.Tensor] = {}
        lengths = None
        if self.dec_use_len_mask:
            out["length_logits"] = self.length_head(g_mix)
            with torch.no_grad():
                lengths = (y_in != self.pad_id).sum(dim=1).to(dtype=torch.long)
                lengths = lengths.clamp(min=1, max=int(T))

        logits = self.decoder(g_mix, T, mem=mem, lengths=lengths)
        out["logits"] = logits
        return out


class LitAmpGNN(pl.LightningModule):
    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        vocab_size: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        lr: int,
        weight_decay: float,
        scheduler: str,
        warmup_steps: int,
        graph_feat_dim: int = 0,
        t_0: int = 10,
        t_mult: int = 2,
        eta_min: float = 0.0,
        loss_mode: Literal["ce", "ctc"] = "ctc",
        label_smoothing: float = 0.0,
        length_loss_weight: float = 0.1,
        max_steps: int = None,
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
        output_mode: Literal["sequence", "term_slots"] = "sequence",
        slot_coefficient_size: int = 0,
        slot_symbol_size: int = 0,
        slot_coupling_size: int = 0,
        slot_max_exponent: int = 8,
        slot_num_slots: int = 64,
        slot_num_factors: int = 3,
        slot_den_slots: int = 4,
        slot_den_terms: int = 5,
        slot_den_term_factors: int = 3,
        slot_degree_loss_weight: float = 0.1,
        slot_no_object_weight: float = 0.1,
        slot_symbol_degree2: Optional[List[int]] = None,
        slot_zero_coefficient_id: int = 0,
        num_anchors: Optional[int] = None,
        rank_anchor_dims: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AmpGNN(
            in_dim=in_dim, edge_dim=edge_dim, graph_feat_dim=graph_feat_dim,
            vocab_size=vocab_size, pad_id=pad_id,
            enc_hid=enc_hid, enc_layers=enc_layers, enc_heads=enc_heads, enc_dropout=enc_dropout,
            d_model=d_model, dec_nhead=dec_nhead, dec_layers=dec_layers, dec_dropout=dec_dropout,
            dec_max_len=dec_max_len, dec_use_len_mask=dec_use_len_mask,
            output_mode=output_mode,
            slot_coefficient_size=slot_coefficient_size,
            slot_symbol_size=slot_symbol_size,
            slot_coupling_size=slot_coupling_size,
            slot_max_exponent=slot_max_exponent,
            slot_num_slots=slot_num_slots,
            slot_num_factors=slot_num_factors,
            slot_den_slots=slot_den_slots,
            slot_den_terms=slot_den_terms,
            slot_den_term_factors=slot_den_term_factors,
            enc_add_node_pos=False, enc_add_anchor_dists=True, enc_num_anchors=num_anchors,
            enc_rank_anchor_dims=rank_anchor_dims,
        )
        self.label_smoothing = label_smoothing
        self.length_loss_weight = length_loss_weight
        self.failure_log_dir = failure_log_dir
        self.failure_log_max= failure_log_max
        self._val_fail = []
        self._test_fail = []
        self._vocab = None
        self._slot_vocab = None
        self._detok = None
        self.phys_decode_mode = "none"
        self.canonicalize_commutative = True
        self.register_buffer("bos_id_t", torch.tensor(bos_id, dtype=torch.long), persistent=False)
        self.register_buffer("eos_id_t", torch.tensor(eos_id, dtype=torch.long), persistent=False)
        self.register_buffer("pad_id_t", torch.tensor(pad_id, dtype=torch.long), persistent=False)
        self.bos_id = int(self.bos_id_t.item())
        self.eos_id = int(self.eos_id_t.item())
        self.pad_id = int(self.pad_id_t.item())
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.t_0 = t_0
        self.t_mult = t_mult
        self.eta_min = eta_min
        self.max_steps = max_steps

        self.loss_mode = str(loss_mode).lower().strip()
        if self.loss_mode not in ("ce", "ctc"):
            raise ValueError(f"loss_mode must be 'ce' or 'ctc' (got {loss_mode!r})")
        self.output_mode = str(output_mode)
        if self.output_mode == "term_slots" and self.loss_mode != "ce":
            raise ValueError("term_slots supports CE only")
        self.slot_degree_loss_weight = float(slot_degree_loss_weight)
        self.slot_no_object_weight = float(slot_no_object_weight)
        if self.slot_degree_loss_weight < 0:
            raise ValueError("slot_degree_loss_weight must be non-negative")
        if self.slot_no_object_weight <= 0:
            raise ValueError("slot_no_object_weight must be positive")
        self.slot_zero_coefficient_id = int(slot_zero_coefficient_id)
        degree_values = slot_symbol_degree2 or [0] * max(int(slot_symbol_size), 1)
        if self.output_mode == "term_slots" and len(degree_values) != int(slot_symbol_size):
            raise ValueError("slot_symbol_degree2 must match slot_symbol_size")
        self.register_buffer(
            "slot_symbol_degree2",
            torch.tensor(degree_values, dtype=torch.float32),
            persistent=True,
        )

        self.ctc_loss_fn = nn.CTCLoss(blank=self.pad_id, reduction="none", zero_infinity=True)

    @staticmethod
    def _mask_from_targets(y: torch.Tensor, pad_id: int) -> torch.Tensor:
        return (y != pad_id)

    def _ids_to_tok_strings(self, ids: List[int]) -> List[str]:
        if self._vocab is None:
            return [str(i) for i in ids]
        itos = getattr(self._vocab, "itos", None)
        if itos is None:
            return [str(i) for i in ids]
        return [itos[int(i)] for i in ids]

    @staticmethod
    def _trim_sequence_ids(ids: torch.Tensor, eos_id: int, pad_id: int) -> List[int]:
        """Return one semantic sequence, including EOS and excluding trailing PAD."""
        out: List[int] = []
        for token in ids.detach().cpu().tolist():
            token = int(token)
            if token == int(pad_id):
                break
            out.append(token)
            if token == int(eos_id):
                break
        return out

    def _canonicalize_sequence_ids(self, ids: List[int], eos_id: int) -> List[str]:
        """Canonicalize expression content while preserving EOS as a boundary token."""
        has_eos = bool(ids) and int(ids[-1]) == int(eos_id)
        content_ids = ids[:-1] if has_eos else ids
        canonical = safe_canonicalize_tokens(self._ids_to_tok_strings(content_ids))
        if has_eos:
            canonical.extend(self._ids_to_tok_strings([int(eos_id)]))
        return canonical

    def _canonical_token_and_seq_acc(
        self,
        pred: torch.Tensor,
        y_out: torch.Tensor,
        pad_id: int,
        eos_id: int,
        keep_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if keep_loss is not None:
                keep_loss = keep_loss.to(device=y_out.device, dtype=torch.bool)

            B = y_out.size(0)
            tok_correct = 0
            tok_total = 0
            seq_correct = 0
            seq_total = 0
            for b in range(B):
                if keep_loss is not None and not bool(keep_loss[b].item()):
                    continue
                tgt_ids = self._trim_sequence_ids(y_out[b], eos_id=eos_id, pad_id=pad_id)
                prd_ids = self._trim_sequence_ids(pred[b], eos_id=eos_id, pad_id=pad_id)
                canon_tgt = self._canonicalize_sequence_ids(tgt_ids, eos_id=eos_id)
                canon_prd = self._canonicalize_sequence_ids(prd_ids, eos_id=eos_id)
                seq_correct += int(canon_tgt == canon_prd)
                seq_total += 1
                tok_total += len(canon_tgt)
                tok_correct += sum(
                    i < len(canon_prd) and canon_tgt[i] == canon_prd[i]
                    for i in range(len(canon_tgt))
                )

            token_acc = torch.tensor(
                float(tok_correct) / float(max(tok_total, 1)),
                device=y_out.device,
            )
            seq_acc = torch.tensor(
                float(seq_correct) / float(max(seq_total, 1)),
                device=y_out.device,
            )
        return token_acc, seq_acc

    @staticmethod
    def _token_and_seq_acc(
        logits: torch.Tensor,
        y_out: torch.Tensor,
        pad_id: int,
        eos_id: int,
        keep_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            mask = LitAmpGNN._build_eval_mask(y_out, eos_id=eos_id, pad_id=pad_id)
            if keep_loss is not None:
                keep_loss = keep_loss.to(device=y_out.device, dtype=torch.bool)
                mask = mask & keep_loss.unsqueeze(1)

            correct = (pred == y_out) & mask
            token_acc = correct.sum(dtype=torch.float32) / mask.sum().clamp_min(1)

            per_ex_seq = ((correct | ~mask).all(dim=1)).float()
            if keep_loss is not None and keep_loss.any():
                seq_acc = per_ex_seq[keep_loss].mean()
            elif keep_loss is not None:
                seq_acc = torch.zeros((), device=y_out.device)
            else:
                seq_acc = per_ex_seq.mean()

        return token_acc, seq_acc

    @staticmethod
    def _ctc_best_path_to_padded(
        pred_ids: torch.Tensor,
        out_len: int,
        *,
        pad_id: int,
        blank_id: int,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        B, _T = pred_ids.shape
        out = pred_ids.new_full((B, int(out_len)), int(pad_id))
        for b in range(B):
            s = torch.unique_consecutive(pred_ids[b])
            s = s[s != int(blank_id)]
            if eos_id is not None:
                eos_pos = torch.nonzero(s == int(eos_id), as_tuple=False)
                if eos_pos.numel() > 0:
                    s = s[: int(eos_pos[0].item()) + 1]
            L = min(int(out_len), int(s.numel()))
            if L > 0:
                out[b, :L] = s[:L]
        return out

    @staticmethod
    def _token_and_seq_acc_ctc(
        logits: torch.Tensor,
        y_out: torch.Tensor,
        pad_id: int,
        eos_id: int,
        keep_loss: Optional[torch.Tensor] = None,
        blank_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if blank_id is None:
                blank_id = pad_id
            pred_raw = logits.argmax(dim=-1)
            pred = LitAmpGNN._ctc_best_path_to_padded(
                pred_raw,
                out_len=y_out.size(1),
                pad_id=pad_id,
                blank_id=blank_id,
                eos_id=eos_id,
            )

            mask = LitAmpGNN._build_eval_mask(y_out, eos_id=eos_id, pad_id=pad_id)
            if keep_loss is not None:
                keep_loss = keep_loss.to(device=y_out.device, dtype=torch.bool)
                mask = mask & keep_loss.unsqueeze(1)

            correct = (pred == y_out) & mask
            token_acc = correct.sum(dtype=torch.float32) / mask.sum().clamp_min(1)

            per_ex_seq = ((correct | ~mask).all(dim=1)).float()
            if keep_loss is not None and keep_loss.any():
                seq_acc = per_ex_seq[keep_loss].mean()
            elif keep_loss is not None:
                seq_acc = torch.zeros((), device=y_out.device)
            else:
                seq_acc = per_ex_seq.mean()
        return token_acc, seq_acc

    def _ce_loss(self, logits: torch.Tensor, targets: torch.Tensor, keep_loss: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = targets.shape
        V = logits.size(-1)

        per_tok = F.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1),
            ignore_index=self.pad_id,
            reduction="none",
            label_smoothing=self.label_smoothing,
        ).view(B, T)

        mask = (targets != self.pad_id)
        if keep_loss is not None:
            keep_loss = keep_loss.to(device=targets.device, dtype=torch.bool)
            mask = mask & keep_loss.unsqueeze(1)

        denom = mask.sum(dtype=torch.float32).clamp_min(1.0)
        loss = (per_tok * mask.to(per_tok.dtype)).sum(dtype=torch.float32) / denom
        return loss

    def _ctc_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        keep_loss: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _V = logits.shape
        device = logits.device


        log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1).contiguous()

        if input_lengths is None:
            input_lengths = torch.full((B,), T, device=device, dtype=torch.long)
        else:
            input_lengths = input_lengths.to(device=device, dtype=torch.long).clamp(min=1, max=T)


        with torch.no_grad():
            target_lengths = (targets != self.pad_id).sum(dim=1).to(device=device, dtype=torch.long)
            target_lengths = torch.minimum(target_lengths, input_lengths)
            target_lengths = target_lengths.clamp(min=1)


        targets = targets.to(device=device, dtype=torch.long)
        flat: List[torch.Tensor] = []
        for b in range(B):
            lb = int(target_lengths[b].item())
            flat.append(targets[b, :lb])
        targets_1d = torch.cat(flat, dim=0) if flat else targets.new_empty((0,), dtype=torch.long)

        per_ex = self.ctc_loss_fn(log_probs, targets_1d, input_lengths, target_lengths)

        if keep_loss is not None:
            keep = keep_loss.to(device=device, dtype=torch.bool)
            per_ex = per_ex[keep]
            target_lengths = target_lengths[keep]

        if per_ex.numel() == 0:
            return log_probs.new_zeros(())


        denom = target_lengths.to(dtype=per_ex.dtype).sum().clamp_min(1.0)
        return per_ex.sum() / denom

    @staticmethod
    def _rank_key_from_batch(batch) -> Optional[str]:
        if len(batch) < 7:
            return None
        meta_list = batch[6]
        if not meta_list:
            return None
        return meta_list[0].get("rank")

    @staticmethod
    def _process_key_from_batch(batch) -> Optional[str]:
        if len(batch) < 7:
            return None
        meta_list = batch[6]
        if not meta_list:
            return None
        m = meta_list[0]
        model = m.get("model")
        rank = m.get("rank")
        if model is None or rank is None:
            return None
        return f"{model}_{rank}"

    @staticmethod
    def _meta_from_batch(batch) -> Optional[List[Dict[str, Any]]]:
        if len(batch) < 7:
            return None
        meta_list = batch[6]
        return meta_list if meta_list else None

    def _apply_phys_mask(self, logits: torch.Tensor, meta_list: Optional[List[Dict[str, Any]]]) -> torch.Tensor:
        mode = getattr(self, "phys_decode_mode", "none")
        if mode in ("", "none") or self._vocab is None:
            return logits
        return apply_physics_allowlist(logits, meta_list, self._vocab, mode)

    def _step_common(self, batch):
        batches_per_p, idx_per_p, masks, y_in, y_out, keep_loss, *_ = batch
        rank_key = self._rank_key_from_batch(batch)
        process_key = self._process_key_from_batch(batch)
        meta_list = self._meta_from_batch(batch)
        decoder_input = None if self.output_mode == "term_slots" else y_in
        out = self.model.forward(
            batches_per_p, idx_per_p, masks, decoder_input, rank_key=rank_key
        )
        if self.output_mode == "term_slots":
            if not meta_list or any("term_slot_target" not in meta for meta in meta_list):
                raise RuntimeError("term-slot batch is missing structured targets")
            slot_targets = (
                batch[7]
                if len(batch) >= 8
                else [meta["term_slot_target"] for meta in meta_list]
            )
            slot_result = structured_slot_objective(
                out,
                slot_targets,
                keep_loss=keep_loss,
                symbol_degree2=self.slot_symbol_degree2,
                zero_coefficient_id=self.slot_zero_coefficient_id,
                label_smoothing=self.label_smoothing,
                no_object_weight=self.slot_no_object_weight,
            )
            main_loss = slot_result["main_loss"]
            degree_loss = slot_result["degree_loss"]
            loss = main_loss + self.slot_degree_loss_weight * degree_loss
            return (
                loss,
                main_loss,
                slot_result["token_acc"],
                slot_result["seq_acc"],
                y_in.size(0),
                process_key,
                int(slot_result["token_weight"]),
                degree_loss,
                slot_result["degree_valid"],
            )
        raw_logits = out["logits"]
        logits = self._apply_phys_mask(raw_logits, meta_list)

        # CE/CTC loss uses unmasked logits: with label smoothing, the -1e4 fill on
        # phys-masked classes would otherwise add a huge constant offset to the loss.
        # The mask still applies to accuracy metrics and decoding below.
        if self.loss_mode == "ctc":
            main_loss = self._ctc_loss(raw_logits, y_out, keep_loss=keep_loss)
            tok_acc, seq_acc = self._token_and_seq_acc_ctc(
                logits, y_out, self.pad_id, self.eos_id, keep_loss=keep_loss, blank_id=self.pad_id
            )
        else:
            main_loss = self._ce_loss(raw_logits, y_out, keep_loss=keep_loss)
            pred = logits.argmax(dim=-1)
            if getattr(self, "canonicalize_commutative", False):
                tok_acc, seq_acc = self._canonical_token_and_seq_acc(
                    pred, y_out, self.pad_id, self.eos_id, keep_loss=keep_loss
                )
            else:
                tok_acc, seq_acc = self._token_and_seq_acc(
                    logits, y_out, self.pad_id, self.eos_id, keep_loss=keep_loss
                )

        loss = main_loss
        length_logits = out.get("length_logits", None)
        if length_logits is not None and getattr(self, "length_loss_weight", 0.0) > 0.0:

            with torch.no_grad():
                nonpad = (y_out != self.pad_id)
                lengths = nonpad.sum(dim=1)
                lengths = lengths.clamp(min=1, max=length_logits.size(1))
                length_targets = lengths - 1

            loss_len = F.cross_entropy(length_logits, length_targets)
            loss = loss + self.length_loss_weight * loss_len

        token_weight = self._token_metric_weight(y_out, keep_loss)
        zero = loss.detach().new_zeros(())
        return (
            loss, main_loss, tok_acc, seq_acc, y_in.size(0), process_key,
            token_weight, zero, zero,
        )

    def _log_per_rank(
        self, stage: str, process_key: Optional[str], loss, main, tok, seq, bs,
        token_weight: int,
    ):
        if process_key is None:
            return
        self.log(f"{stage}/loss/{process_key}", loss, on_epoch=True, sync_dist=True, batch_size=bs)
        metric_name = "ctc_loss" if self.loss_mode == "ctc" else "ce_loss"
        self.log(f"{stage}/{metric_name}/{process_key}", main, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log(
            f"{stage}/token_acc/{process_key}", tok, on_epoch=True, sync_dist=True,
            batch_size=max(token_weight, 1),
        )
        self.log(f"{stage}/seq_acc/{process_key}", seq, on_epoch=True, sync_dist=True, batch_size=bs)

    def training_step(self, batch, _):
        loss, main, tok, seq, bs, process_key, token_weight, degree_loss, degree_valid = self._step_common(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        if self.loss_mode == "ctc":
            self.log("train/ctc_loss", main, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        else:
            self.log("train/ce_loss", main, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log(
            "train/token_acc", tok, on_epoch=True, sync_dist=True,
            batch_size=max(token_weight, 1),
        )
        self.log("train/seq_acc", seq, on_epoch=True, sync_dist=True, batch_size=bs)
        if self.output_mode == "term_slots":
            self.log("train/degree_loss", degree_loss, on_epoch=True, sync_dist=True, batch_size=bs)
            self.log("train/degree_valid", degree_valid, on_epoch=True, sync_dist=True, batch_size=bs)
        self._log_per_rank("train", process_key, loss, main, tok, seq, bs, token_weight)
        return loss

    def validation_step(self, batch, _):
        loss, main, tok, seq, bs, process_key, token_weight, degree_loss, degree_valid = self._step_common(batch)
        if getattr(self.model, "dec_use_len_mask", False):
            tok, seq = self._generate_metrics(batch)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        if self.loss_mode == "ctc":
            self.log("val/ctc_loss", main, on_epoch=True, sync_dist=True, batch_size=bs)
        else:
            self.log("val/ce_loss", main, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log(
            "val/token_acc", tok, on_epoch=True, prog_bar=True, sync_dist=True,
            batch_size=max(token_weight, 1),
        )
        self.log("val/seq_acc", seq, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=bs)
        if self.output_mode == "term_slots":
            self.log("val/degree_loss", degree_loss, on_epoch=True, sync_dist=True, batch_size=bs)
            self.log("val/degree_valid", degree_valid, on_epoch=True, sync_dist=True, batch_size=bs)
        self._log_per_rank("val", process_key, loss, main, tok, seq, bs, token_weight)

    @torch.no_grad()
    def generate(
        self,
        batch_or_group,
        max_len,
        rank_key: Optional[str] = None,
        meta_list: Optional[List[Dict[str, Any]]] = None,
    ):
        batches_per_p, idx_per_p, mask = batch_or_group


        g_mix, mem = self.model.encode_mix_with_memory(batches_per_p, idx_per_p, mask, rank_key=rank_key)
        B = g_mix.size(0)
        device = g_mix.device


        length_pred = None
        T_max = max_len
        if getattr(self.model, "length_head", None) is not None:
            length_logits = self.model.length_head(g_mix)
            length_pred = length_logits.argmax(dim=-1) + 1
            if max_len is not None:
                length_pred = torch.clamp(length_pred, max=max_len)
            T_max = int(length_pred.max().item())

        if T_max is None or T_max <= 0:
            return torch.full((B, 0), fill_value=self.pad_id, dtype=torch.long, device=device)


        dec_lengths = None
        if getattr(self.model, "dec_use_len_mask", False) and length_pred is not None:
            dec_lengths = length_pred

        logits = self.model.decoder(g_mix, T_max, mem=mem, lengths=dec_lengths)
        logits = self._apply_phys_mask(logits, meta_list)
        seq = logits.argmax(dim=-1)


        if self.loss_mode == "ctc":
            out_len = T_max if max_len is None else int(max_len)
            out = self._ctc_best_path_to_padded(
                seq,
                out_len=out_len,
                pad_id=self.pad_id,
                blank_id=self.pad_id,
                eos_id=self.eos_id,
            )
            if length_pred is not None:

                for b in range(B):
                    tb = int(length_pred[b].item())
                    tb = max(1, min(tb, out_len))
                    out[b, tb:] = self.pad_id
            return out


        if length_pred is not None:

            out_len = T_max if max_len is None else max_len
            out = torch.full((B, out_len), fill_value=self.pad_id, dtype=torch.long, device=device)
            for b in range(B):
                tb = int(length_pred[b].item())
                tb = max(1, min(tb, out_len))
                out[b, :tb] = seq[b, :tb]
            return out
        else:
            if max_len is not None and seq.size(1) > max_len:
                return seq[:, :max_len]
            return seq

    @staticmethod
    def _acc_from_pred_ids(
        pred: torch.Tensor,
        y_out: torch.Tensor,
        pad_id: int,
        eos_id: int,
        keep_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if keep_loss is not None:
                keep_loss = keep_loss.to(device=y_out.device, dtype=torch.bool)

            tok_correct = 0
            tok_total = 0
            seq_correct = 0
            seq_total = 0
            for b in range(y_out.size(0)):
                if keep_loss is not None and not bool(keep_loss[b].item()):
                    continue
                tgt_ids = LitAmpGNN._trim_sequence_ids(y_out[b], eos_id=eos_id, pad_id=pad_id)
                prd_ids = LitAmpGNN._trim_sequence_ids(pred[b], eos_id=eos_id, pad_id=pad_id)
                tok_total += len(tgt_ids)
                tok_correct += sum(
                    i < len(prd_ids) and tgt_ids[i] == prd_ids[i]
                    for i in range(len(tgt_ids))
                )
                seq_correct += int(tgt_ids == prd_ids)
                seq_total += 1

            token_acc = torch.tensor(
                float(tok_correct) / float(max(tok_total, 1)), device=y_out.device
            )
            seq_acc = torch.tensor(
                float(seq_correct) / float(max(seq_total, 1)), device=y_out.device
            )
        return token_acc, seq_acc

    @torch.no_grad()
    def _generate_metrics(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        batches_per_p, idx_per_p, masks, y_in, y_out, keep_loss, *_ = batch
        rank_key = self._rank_key_from_batch(batch)
        meta_list = self._meta_from_batch(batch)
        pred = self.generate(
            (batches_per_p, idx_per_p, masks),
            max_len=self.model.max_dec_len,
            rank_key=rank_key,
            meta_list=meta_list,
        )
        if getattr(self, "canonicalize_commutative", False):
            return self._canonical_token_and_seq_acc(
                pred, y_out, self.pad_id, self.eos_id, keep_loss=keep_loss
            )
        return self._acc_from_pred_ids(pred, y_out, self.pad_id, self.eos_id, keep_loss=keep_loss)

    @staticmethod
    def _build_eval_mask(y_out: torch.Tensor, eos_id: int, pad_id: int) -> torch.Tensor:
        B, T = y_out.shape
        device = y_out.device
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        is_pad = (y_out == pad_id)
        is_eos = (y_out == eos_id)
        first_pad = torch.where(is_pad, pos, T).min(dim=1).values
        first_eos = torch.where(is_eos, pos, T).min(dim=1).values
        stop = torch.minimum(first_pad, torch.clamp(first_eos + 1, max=T))
        return pos < stop.unsqueeze(1)

    def _token_metric_weight(
        self,
        y_out: torch.Tensor,
        keep_loss: Optional[torch.Tensor] = None,
    ) -> int:
        mask = self._build_eval_mask(y_out, eos_id=self.eos_id, pad_id=self.pad_id)
        if keep_loss is not None:
            keep = keep_loss.to(device=y_out.device, dtype=torch.bool)
            mask = mask & keep.unsqueeze(1)
        return int(mask.sum().item())

    def attach_vocab(self, vocab: Any, detokenize: Optional[Any] = None) -> None:
        self._vocab = vocab
        self._slot_vocab = getattr(vocab, "term_slot_vocabulary", None)
        if detokenize is not None:
            self._detok = detokenize
        elif hasattr(vocab, "detokenize"):
            self._detok = getattr(vocab, "detokenize")
        elif hasattr(vocab, "decode"):
            self._detok = lambda ids: vocab.decode(ids)
        else:
            itos = getattr(vocab, "itos", None)
            if itos is not None:
                self._detok = lambda ids: "".join(itos[i] for i in ids)
            else:
                self._detok = lambda ids: " ".join(str(i) for i in ids)

    def _ids_to_text(self, ids: List[int]) -> str:
        ids = [i for i in ids if i != self.pad_id]
        if self.bos_id in ids:
            try:
                ids = ids[ids.index(self.bos_id) + 1 :]
            except ValueError:
                pass
        if self.eos_id in ids:
            ids = ids[: ids.index(self.eos_id)]
        return self._detok(ids) if self._detok is not None else " ".join(map(str, ids))

    def _readable_slot_prediction(self, prediction: Dict[str, object]) -> Dict[str, object]:
        if not prediction or self._slot_vocab is None:
            return prediction
        coefficient_names = {
            value: key for key, value in self._slot_vocab.coefficient_to_id.items()
        }
        symbol_names = {value: key for key, value in self._slot_vocab.symbol_to_id.items()}
        coupling_names = {
            value: key for key, value in self._slot_vocab.coupling_to_id.items()
        }

        def monomial(value):
            coefficient, factors = value
            return {
                "coefficient": coefficient_names.get(int(coefficient), "[UNK]"),
                "factors": [
                    {
                        "symbol": symbol_names.get(int(symbol), "[UNK]"),
                        "exponent": int(exponent),
                    }
                    for symbol, exponent in factors
                ],
            }

        global_prediction = prediction["global"]
        return {
            "global": {
                "coefficient": coefficient_names.get(
                    int(global_prediction["coefficient"]), "[UNK]"
                ),
                "coupling": coupling_names.get(
                    int(global_prediction["coupling"]), "[UNK]"
                ),
                "coupling_power": int(global_prediction["coupling_power"]),
            },
            "numerator_terms": [
                monomial(value) for value in prediction["numerator_terms"]
            ],
            "denominator_factors": [
                {
                    "power": int(power),
                    "terms": [monomial(value) for value in terms],
                }
                for power, terms in prediction["denominator_factors"]
            ],
        }

    @torch.no_grad()
    def _maybe_collect_failures(self, stage: str, batch) -> None:
        if self.failure_log_dir is None:
            return
        if getattr(self, "global_rank", 0) != 0:
            return
        buf = self._val_fail if stage == "val" else self._test_fail
        if len(buf) >= int(self.failure_log_max):
            return

        batches_per_p, idx_per_p, masks, y_in, y_out, keep_loss, *_ = batch
        rank_key = self._rank_key_from_batch(batch)
        meta_list = self._meta_from_batch(batch)
        if self.output_mode == "term_slots":
            out = self.model.forward(
                batches_per_p, idx_per_p, masks, None, rank_key=rank_key
            )
            slot_result = structured_slot_objective(
                out,
                (
                    batch[7]
                    if len(batch) >= 8
                    else [meta["term_slot_target"] for meta in meta_list]
                ),
                keep_loss=keep_loss,
                symbol_degree2=self.slot_symbol_degree2,
                zero_coefficient_id=self.slot_zero_coefficient_id,
                label_smoothing=self.label_smoothing,
                no_object_weight=self.slot_no_object_weight,
                return_predictions=True,
            )
            for index, exact in enumerate(slot_result["exact_flags"]):
                if exact is None or exact or len(buf) >= int(self.failure_log_max):
                    continue
                buf.append({
                    "stage": stage,
                    "epoch": int(getattr(self, "current_epoch", -1)),
                    "global_step": int(getattr(self, "global_step", -1)),
                    "batch_index": int(index),
                    "target_slots": meta_list[index].get(
                        "term_slot_target_raw", meta_list[index]["term_slot_target"]
                    ),
                    "pred_slots": self._readable_slot_prediction(
                        slot_result["predictions"][index]
                    ),
                })
            return
        if getattr(self.model, "dec_use_len_mask", False):
            pred = self.generate(
                (batches_per_p, idx_per_p, masks),
                max_len=self.model.max_dec_len,
                rank_key=rank_key,
                meta_list=meta_list,
            )
        else:
            out = self.model.forward(batches_per_p, idx_per_p, masks, y_in, rank_key=rank_key)
            logits = self._apply_phys_mask(out["logits"], meta_list)
            pred_raw = logits.argmax(dim=-1)
            if self.loss_mode == "ctc":
                pred = self._ctc_best_path_to_padded(
                    pred_raw,
                    out_len=y_out.size(1),
                    pad_id=self.pad_id,
                    blank_id=self.pad_id,
                    eos_id=self.eos_id,
                )
            else:
                pred = pred_raw

        if keep_loss is not None:
            keep_loss = keep_loss.to(device=y_out.device, dtype=torch.bool)

        incorrect: List[int] = []
        trimmed: Dict[int, Tuple[List[int], List[int]]] = {}
        for b in range(y_out.size(0)):
            if keep_loss is not None and not bool(keep_loss[b].item()):
                continue
            tgt_ids = self._trim_sequence_ids(y_out[b], eos_id=self.eos_id, pad_id=self.pad_id)
            prd_ids = self._trim_sequence_ids(pred[b], eos_id=self.eos_id, pad_id=self.pad_id)
            trimmed[b] = (tgt_ids, prd_ids)
            if getattr(self, "canonicalize_commutative", False):
                canon_tgt = self._canonicalize_sequence_ids(tgt_ids, eos_id=self.eos_id)
                canon_prd = self._canonicalize_sequence_ids(prd_ids, eos_id=self.eos_id)
                equal = canon_tgt == canon_prd
            else:
                equal = tgt_ids == prd_ids
            if not equal:
                incorrect.append(b)

        for i in incorrect:
            if len(buf) >= int(self.failure_log_max):
                break
            tgt_ids, prd_ids = trimmed[i]
            buf.append({
                "stage": stage,
                "epoch": int(getattr(self, "current_epoch", -1)),
                "global_step": int(getattr(self, "global_step", -1)),
                "batch_index": int(i),
                "target_text": self._ids_to_text(tgt_ids),
                "pred_text":   self._ids_to_text(prd_ids),
                "target_ids": tgt_ids,
                "pred_ids": prd_ids,
            })

    def on_validation_epoch_start(self) -> None:
        self._val_fail = []

    def on_test_epoch_start(self) -> None:
        self._test_fail = []

    def on_validation_epoch_end(self) -> None:
        self._flush_failures("val", self._val_fail)

    def on_test_epoch_end(self) -> None:
        self._flush_failures("test", self._test_fail)

    def _flush_failures(self, stage: str, buf: List[Dict[str, Any]]) -> None:
        if self.failure_log_dir is None or not buf:
            return
        if getattr(self, "global_rank", 0) != 0:
            return
        os.makedirs(self.failure_log_dir, exist_ok=True)
        path = os.path.join(self.failure_log_dir, f"{stage}_failures.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            for rec in buf:
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")

    def test_step(self, batch, batch_idx):
        loss, main, tok, seq, bs, process_key, token_weight, degree_loss, degree_valid = self._step_common(batch)
        if getattr(self.model, "dec_use_len_mask", False):
            tok, seq = self._generate_metrics(batch)
        self.log("test/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        if self.loss_mode == "ctc":
            self.log("test/ctc_loss", main, on_epoch=True, sync_dist=True, batch_size=bs)
        else:
            self.log("test/ce_loss", main, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log(
            "test/token_acc", tok, on_epoch=True, prog_bar=True, sync_dist=True,
            batch_size=max(token_weight, 1),
        )
        self.log("test/seq_acc", seq, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=bs)
        if self.output_mode == "term_slots":
            self.log("test/degree_loss", degree_loss, on_epoch=True, sync_dist=True, batch_size=bs)
            self.log("test/degree_valid", degree_valid, on_epoch=True, sync_dist=True, batch_size=bs)
        self._log_per_rank("test", process_key, loss, main, tok, seq, bs, token_weight)
        self._maybe_collect_failures("test", batch)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        sched = self.scheduler
        if sched == "none":
            return opt

        eta_min_ratio = 0.0
        if self.lr and self.lr > 0 and self.eta_min and self.eta_min > 0:
            eta_min_ratio = float(self.eta_min) / float(self.lr)
            eta_min_ratio = max(0.0, min(1.0, eta_min_ratio))

        def build_warmup_decay(total_steps: int, warmup_steps: int, mode: str):
            floor = eta_min_ratio
            def lr_lambda(step: int):
                if warmup_steps > 0 and step < warmup_steps:
                    return float(step + 1) / float(max(1, warmup_steps))
                progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                progress = max(0.0, min(1.0, progress))
                if mode == "linear":
                    base = 1.0 - progress
                elif mode == "cosine":
                    base = 0.5 * (1.0 + math.cos(math.pi * progress))
                else:
                    base = 1.0
                return floor + (1.0 - floor) * base
            return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        total_steps = self.max_steps
        interval = "epoch"
        monitor = None

        if sched in {"cosine_warmup","linear_warmup","onecycle","cosine_warm_restarts"}:

            if total_steps is None and self.trainer is not None and self.trainer.estimated_stepping_batches is not None:
                total_steps = int(self.trainer.estimated_stepping_batches)
            if total_steps is None:

                sched = "cosine"
            else:
                current_step = int(getattr(self, "global_step", 0) or 0)
                print(
                    f"[sched] scheduler={sched!r} total_steps={total_steps} "
                    f"warmup_steps={self.warmup_steps} current_global_step={current_step} "
                    f"lr={self.lr} eta_min={self.eta_min} eta_min_ratio={eta_min_ratio:.4g}"
                )
                if self.warmup_steps >= total_steps:
                    print(
                        f"[sched][WARN] warmup_steps ({self.warmup_steps}) >= total_steps "
                        f"({total_steps}). The LR will never finish warming up and will "
                        "never enter the cosine/linear decay phase. Reduce --warmup_steps "
                        "or train for more --epochs."
                    )
                if current_step >= total_steps:
                    print(
                        f"[sched][WARN] current global_step ({current_step}) >= total_steps "
                        f"({total_steps}). The LR schedule is already past its end and the LR "
                        "will be pinned at eta_min for the rest of training. This usually means "
                        "--epochs was reduced on resume. Extend --epochs or pass "
                        "--no-auto_resume to rebuild the schedule from scratch."
                    )

        if sched == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs if self.trainer else 100, eta_min=self.eta_min)
            interval = "epoch"
        elif sched == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=self.step_size, gamma=self.gamma)
            interval = "epoch"
        elif sched == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
            interval = "epoch"
            monitor = "val/loss"
        elif sched == "cosine_warmup":
            scheduler = build_warmup_decay(total_steps, self.warmup_steps, mode="cosine")
            interval = "step"
        elif sched == "linear_warmup":
            scheduler = build_warmup_decay(total_steps, self.warmup_steps, mode="linear")
            interval = "step"
        elif sched == "cosine_warm_restarts":
            max_epochs = self.trainer.max_epochs if self.trainer else 100
            steps_per_epoch = max(1, total_steps // max_epochs) if total_steps else 1
            t0_steps = self.t_0 * steps_per_epoch
            warm_restarts = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=t0_steps, T_mult=self.t_mult, eta_min=self.eta_min,
            )
            if self.warmup_steps > 0:
                warmup_sched = torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=1e-8, end_factor=1.0, total_iters=self.warmup_steps,
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    opt, schedulers=[warmup_sched, warm_restarts], milestones=[self.warmup_steps],
                )
            else:
                scheduler = warm_restarts
            interval = "step"
        elif sched == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=self.lr, total_steps=total_steps or 100, pct_start=float(self.warmup_steps)/(total_steps or 100)
            )
            interval = "step"
        else:
            return opt

        if monitor is not None:
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": interval,
                    "monitor": monitor,
                },
            }
        else:
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": interval,
                },
            }
