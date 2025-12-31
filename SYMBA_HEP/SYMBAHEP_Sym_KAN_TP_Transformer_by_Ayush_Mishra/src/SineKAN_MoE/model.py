import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math

class SineKAN1D(nn.Module):
    """
    Input:  (..., input_dim)  e.g., (B, L)
    Output: (..., output_dim) e.g., (B, O)
    """
    def __init__(self, input_dim, output_dim, device='cuda', grid_size=8, is_first=False, add_bias=True, norm_freq=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.get_default_dtype()

        # Frequencies as a 1D learnable vector
        freq = torch.arange(1, grid_size + 1, dtype=dtype, device=device)
        if norm_freq:
            freq = freq / ((grid_size + 1) ** (0 if is_first else 1))
        self.freq = nn.Parameter(freq)  # (G,)

        # Phase matrix (K, G) precomputed
        input_phase = torch.linspace(0, math.pi, input_dim, dtype=dtype, device=device) * input_dim  # (K,)
        grid_phase = torch.arange(1, grid_size + 1, dtype=dtype, device=device) / (grid_size + 1) * grid_size  # (G,)
        phase = input_phase[:, None] + grid_phase[None, :]  # (K, G)
        self.register_buffer('phase', phase)

        # Amplitudes as (O, K, G). Final matmul done with nn.functional.linear on flattened KG.
        if is_first:
            amp = torch.empty(output_dim, input_dim, grid_size, dtype=dtype, device=device).normal_(0, 0.4)
        else:
            amp = torch.empty(output_dim, input_dim, grid_size, dtype=dtype, device=device).uniform_(-1, 1)
        grid_norm = torch.arange(1, grid_size + 1, dtype=dtype, device=device)  # (G,)
        amp = amp / output_dim / grid_norm[None, None, :]
        self.amplitudes = nn.Parameter(amp)  # (O, K, G)

        if add_bias:
            self.bias = nn.Parameter(torch.ones(output_dim, dtype=dtype, device=device) / output_dim)
        else:
            self.register_parameter('bias', None)

    @property
    def _W(self):
        # (O, K*G)
        return self.amplitudes.reshape(self.output_dim, self.input_dim * self.grid_size)

    def forward(self, x):
        # Support arbitrary leading dims ending with input_dim
        out_shape = x.shape[:-1] + (self.output_dim,)
        x2 = x.reshape(-1, self.input_dim)  # (N, K)
        # Compute sin with minimal broadcasting: (N, K, G)
        s = torch.sin(x2[..., :, None] * self.freq[None, None, :] + self.phase[None, :, :])
        # Dense linear over flattened (K*G)
        y = nn.functional.linear(s.reshape(-1, self.input_dim * self.grid_size), self._W, self.bias)  # (N, O)
        return y.reshape(out_shape)


class SineKANSeqFeat(nn.Module):
    """
    Input:  (B, L, F) where F == input_dim
    Output: (B, L, O) broadcast along L
    """
    def __init__(self, input_dim, output_dim, device='cuda', grid_size=8, is_first=False, add_bias=True, norm_freq=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size

        device = torch.device(device)
        dtype = torch.get_default_dtype()

        freq = torch.arange(1, grid_size + 1, dtype=dtype, device=device)
        if norm_freq:
            freq = freq / ((grid_size + 1) ** (0 if is_first else 1))
        self.freq = nn.Parameter(freq)  # (G,)

        input_phase = torch.linspace(0, math.pi, input_dim, dtype=dtype, device=device) * input_dim  # (F,)
        grid_phase = torch.arange(1, grid_size + 1, dtype=dtype, device=device) / (grid_size + 1) * grid_size  # (G,)
        phase = input_phase[:, None] + grid_phase[None, :]  # (F, G)
        self.register_buffer('phase', phase)

        if is_first:
            amp = torch.empty(output_dim, input_dim, grid_size, dtype=dtype, device=device).normal_(0, 0.4)
        else:
            amp = torch.empty(output_dim, input_dim, grid_size, dtype=dtype, device=device).uniform_(-1, 1)
        grid_norm = torch.arange(1, grid_size + 1, dtype=dtype, device=device)
        amp = amp / output_dim / grid_norm[None, None, :]
        self.amplitudes = nn.Parameter(amp)  # (O, F, G)

        if add_bias:
            self.bias = nn.Parameter(torch.ones(output_dim, dtype=dtype, device=device) / output_dim)
        else:
            self.register_parameter('bias', None)

    @property
    def _W(self):
        return self.amplitudes.reshape(self.output_dim, self.input_dim * self.grid_size)

    def forward(self, x):
        B, L, F = x.shape
        assert F == self.input_dim
        # (B, L, F, G)
        s = torch.sin(x.unsqueeze(-1) * self.freq.view(1, 1, 1, -1) + self.phase.view(1, 1, F, -1))
        # Dense linear per time step
        y = nn.functional.linear(s.reshape(B, L, F * self.grid_size), self._W, self.bias).reshape(B, L, self.output_dim)
        return y

class MoeLayer(nn.Module):
  def __init__(self , d_model , n_experts , k):
    super().__init__()
    self.n_experts = n_experts
    self.experts = nn.ModuleList([SineKAN1D(d_model , d_model , grid_size=8) for i in range(self.n_experts)])
    self.gate = nn.Linear(d_model , self.n_experts)
    self.k = k

  def forward(self , x):
    gate_logits = self.gate(x)
    weights , selected_experts = torch.topk(gate_logits , k = self.k)
    weights = F.softmax(weights , dim=-1)
    out = torch.zeros_like(x)
    for i , current_expert in enumerate(self.experts):
      batch_idx , seq_idx , k_idx = torch.where(selected_experts == i)
      token_x = x[batch_idx, seq_idx]
      token_w = weights[batch_idx, seq_idx, k_idx].unsqueeze(-1)
      expert_out = current_expert(token_x)
      out[batch_idx, seq_idx] += token_w * expert_out
    return out


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model , n_heads, max_seq_len , bias = False) -> None:
    super().__init__()
    assert d_model % n_heads == 0 , "d_model not divisible by n_heads"
    self.d_model = d_model
    self.n_heads = n_heads
    self.bias = bias
    self.qw = nn.Linear(self.d_model , self.d_model , bias = self.bias)
    self.kw = nn.Linear(self.d_model , self.d_model , bias = self.bias)
    self.vw = nn.Linear(self.d_model , self.d_model , bias = self.bias)
    self.project = nn.Linear(self.d_model , self.d_model)
    self.rope = RotaryPositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)

  # def cross_forward()

  def forward(self, q , k , v ,  mask = None):
    ## X dimension ==> (batch_size, seq_len, dim)

    ## (batch_dim, seq_len, d_out)
    q = self.qw(q)
    k = self.kw(k)
    v = self.vw(v)

    k = self.rope(k)
    v = self.rope(v)

    # (batch seq_len d_out) -> (batch n_heads seq_len d_out)
    q = rearrange(q , "b s (h d) -> b h s d", h = self.n_heads)
    k = rearrange(k , "b s (h d) -> b h s d", h = self.n_heads)
    v = rearrange(v , "b s (h d) -> b h s d", h = self.n_heads)

    attention_scores = (q @ k.transpose(-2 , -1)) / (k.shape[-1]**0.5)

    # if mask:
    #   masks = torch.triu(torch.ones(seq_len , seq_len) , diagonal=1)
    #   attention_scores = attention_scores.masked_fill(masks == 1 , float("-inf"))
    #   attention_weights = F.softmax(attention_scores/(k.shape[-1]**0.5) , dim=-1)
    # else:
    #   attention_weights = F.softmax(attention_scores/(k.shape[-1]**0.5) , dim=-1)


    if mask is not None:
      attention_scores = attention_scores.masked_fill(mask == 0 , -1e9)
    attention_weights = F.softmax(attention_scores , dim = -1)

    context_vector = attention_weights @ v

    context_vector = rearrange(context_vector , "b h s d -> b s (h d)")
    return self.project(context_vector)


## Expansion and Contraction Module

class FeedForward(nn.Module):
  def __init__(self, d_in) -> None:
    super().__init__()
    self.ff = nn.Sequential(
        nn.Linear(d_in , 4 * d_in),
        nn.GELU(),
        nn.Linear(4 * d_in , d_in)
    )

  def forward(self, x):
    return self.ff(x)
  
class Encoder(nn.Module):
  def __init__(self, d_model , n_heads , max_seq_len , dropout_ratio = 0.2 , bias = False) -> None:
    super().__init__()
    self.attention = MultiHeadAttention(d_model=d_model , n_heads=n_heads , max_seq_len=max_seq_len , bias = bias)
    self.ff = FeedForward(d_in=d_model)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout_ratio)

  def forward(self , x , src_mask):
    attention_out = self.attention(x , x , x , src_mask)
    x = x + self.norm1(self.dropout(attention_out))
    ff_out = self.ff(x)
    x = x + self.norm2(self.dropout(ff_out))
    return x
  
class Decoder(nn.Module):
  def __init__(self, d_model , max_seq_len , n_heads , dropout_ratio = 0.2 , bias = False) -> None:
    super().__init__()
    self.attention = MultiHeadAttention(d_model=d_model , n_heads=n_heads , max_seq_len=max_seq_len , bias = bias)
    self.ff = FeedForward(d_in=d_model)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout_ratio)

  def forward(self, x , enc_output , src_mask, tgt_mask):
    attention_outputs = self.attention(x , x, x, tgt_mask)
    x = x + self.norm1(self.dropout(attention_outputs))
    cross_attention_outputs = self.attention(x , enc_output , enc_output ,  src_mask)
    x = x + self.norm2(self.dropout(cross_attention_outputs))
    ff_output = self.ff(x)
    x = x + self.norm3(self.dropout(ff_output))
    return x


class PositionalEncoding(nn.Module):
  def __init__(self , d_model , max_seq_len):
    super().__init__()
    self.pos = torch.arange(0 , max_seq_len)
    self.theta = 1 / ((10000 ** (torch.arange(0 , d_model , 2))) / d_model)

    pe = torch.zeros(max_seq_len , d_model)

    pe[...,0::2] = torch.sin(self.pos[: , None] / self.theta)
    pe[...,1::2] = torch.cos(self.pos[: , None] / self.theta)

    self.register_buffer("pe" , pe)

  def forward(self, x):
    b , s , d = x.shape
    return x + self.pe[:s , :]

def get_mask(src , tgt):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  src_mask = (src != 0).to(device)
  src_mask = src_mask[: , None , None , :]
  tgt_mask = (tgt != 0)[: , None , : , None].to(device)
  seq_len = tgt_mask.shape[-2]
  causal_mask = torch.tril(torch.ones(1 , seq_len , seq_len)).bool().to(device)
  final_tgt_mask = tgt_mask & causal_mask
  return src_mask , final_tgt_mask.to(device)


class Transformer(nn.Module):
  def __init__(self , src_vocab_size , d_model , tgt_vocab_size , max_seq_len , n_heads , dropout_ratio , bias , n_encoders , n_decoders, n_experts , k) -> None:
    super().__init__()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.encod_embedding = nn.Embedding(src_vocab_size, d_model)
    self.decod_embedding = nn.Embedding(tgt_vocab_size, d_model)
    self.encoding = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)
    self.encoder = nn.ModuleList([Encoder(d_model=d_model, n_heads=n_heads, max_seq_len=max_seq_len , dropout_ratio=dropout_ratio , bias = bias) for i in range(n_encoders)])
    self.decoder = nn.ModuleList([Decoder(d_model=d_model, n_heads=n_heads , max_seq_len=max_seq_len , dropout_ratio=dropout_ratio , bias = bias) for i in range(n_encoders)])
    self.linear = nn.Linear(d_model , tgt_vocab_size)
    self.dropout = nn.Dropout(dropout_ratio)
    self.ff = MoeLayer(d_model=d_model , n_experts=n_experts , k=k)
    self.ff2 = FeedForward(d_model)

  def forward(self , src, tgt):
    src_mask , tgt_mask = get_mask(src , tgt)
    src_mask = src_mask.to(self.device)
    tgt_mask = tgt_mask.to(self.device)
    encod_embed = self.dropout(self.encod_embedding(src))
    decod_embed = self.dropout(self.decod_embedding(tgt))

    enc_output = encod_embed
    for encoder in self.encoder:
      enc_output = encoder(enc_output , src_mask)

    dec_output = decod_embed
    for decoder in self.decoder:
      dec_output = decoder(dec_output , enc_output , src_mask , tgt_mask )
    # print("Decoder output " , dec_output)
    out = self.ff(dec_output)

    # out = self.ff(dec_output)
    return self.linear(out)

class RotaryPositionalEncoding(nn.Module):
  def __init__(self, d_model , max_seq_len) -> None:
    super().__init__()
    self.max_seq_len = max_seq_len
    self.d_model = d_model
    self.half_d  = d_model // 2
    self.pos = torch.arange(0 , max_seq_len)
    self.theta = 1 / (10000 ** ((2 * torch.arange(0 , self.half_d)) / self.d_model))
    self.angles = self.pos[: , None] * self.theta[None , :]
    sin = torch.sin(self.angles)
    cos = torch.cos(self.angles)
    self.register_buffer("sin" , sin)
    self.register_buffer("cos" , cos)

  def forward(self , x):
    assert x.shape[-2] <= self.max_seq_len , "seq len should be less than max_seq_len"
    seq_len = x.shape[-2]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    x_rot_1 = x1 * self.cos[:seq_len , :]  - x2 * self.sin[:seq_len , :]
    x_rot_2 = x1 * self.sin[:seq_len , :]  + x2 * self.cos[:seq_len , :]

    out = torch.zeros_like(x)
    out[..., 0::2] = x_rot_1
    out[..., 1::2] = x_rot_2
    return out

