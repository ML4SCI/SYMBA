# from flash_attn.bert_padding import unpad_input, pad_input
# from flash_attn import flash_attn_varlen_qkvpacked_func,flash_attn_varlen_kvpacked_func
# import torch
# import torch.nn as nn
# from typing import Optional, Union
# import copy

# from .sinekan import KANFeedForwardBlock
# from .utils import TokenEmbedding, PositionalEncoding

# class FlashMHA(nn.Module):
#     """
#     Multi-Head Attention module using FlashAttention backend for efficient computation.

#     Args:
#         embed_dim (int): Total dimension of the model.
#         num_heads (int): Number of attention heads.
#         dropout (float, optional): Dropout probability applied to attention weights. Defaults to 0.0.
#         **factory_kwargs: Additional keyword arguments passed to nn.Linear projections.
#     """

#     def __init__(
#         self,
#         embed_dim: int,
#         num_heads: int,
#         dropout: float = 0.0,
#         **factory_kwargs
#     ):
#         super().__init__()
#         assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.dropout = dropout

#         self.q_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
#         self.k_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
#         self.v_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
#         self.out_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)

#         self.factory_kwargs = factory_kwargs

#     def forward(
#         self,
#         q: torch.Tensor,
#         k: torch.Tensor = None,
#         v: torch.Tensor = None,
#         padding_mask: torch.Tensor = None,
#         is_cross: bool = False,
#         causal: bool = False
#     ) -> torch.Tensor:
#         """
#         Forward pass for Flash Multi-Head Attention.

#         Args:
#             q (Tensor): Query tensor of shape (batch, seq_len, embed_dim).
#             k (Tensor, optional): Key tensor. Defaults to None (for self-attention).
#             v (Tensor, optional): Value tensor. Defaults to None (for self-attention).
#             padding_mask (Tensor, optional): Mask tensor with True/1 for valid tokens. Defaults to None.
#             is_cross (bool, optional): Indicates whether this is cross-attention. Defaults to False.
#             causal (bool, optional): Whether to apply causal masking. Defaults to False.

#         Returns:
#             Tensor: Output tensor of shape (batch, seq_len, embed_dim).
#         """
#         b, l_q, _ = q.shape
#         device = q.device

#         if not is_cross:
#             # Self-attention mode: Q = K = V
#             k, v = q, q

#         # Project inputs
#         q_proj = self.q_proj(q)
#         k_proj = self.k_proj(k)
#         v_proj = self.v_proj(v)

#         # Handle padding mask
#         if padding_mask is None:
#             padding_mask = torch.ones((b, k.shape[1]), dtype=torch.bool, device=device)
#         else:
#             if padding_mask.shape != (b, k.shape[1]):
#                 raise ValueError(
#                     f"padding_mask shape {padding_mask.shape} does not match "
#                     f"expected shape ({b}, {k.shape[1]})"
#                 )

#         # Set dropout to 0 during evaluation
#         dropout_p = self.dropout if self.training else 0.0

#         if not is_cross:
#             # Self-attention mode
#             qkv = torch.stack([q_proj, k_proj, v_proj], dim=1)  # (b, 3, l, d)
#             qkv = qkv.transpose(1, 2)  # (b, l, 3, d)

#             qkv, indices, cu_seqlens, max_s, _ = unpad_input(qkv, padding_mask)
#             qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)

#             out_unpad = flash_attn_varlen_qkvpacked_func(
#                 qkv,
#                 cu_seqlens,
#                 max_s,
#                 dropout_p=dropout_p,
#                 softmax_scale=None,
#                 causal=causal
#             )
#         else:
#             # Cross-attention mode
#             q_mask = torch.ones((b, l_q), dtype=torch.bool, device=device)
#             q_packed, q_indices, cu_q, max_q, _ = unpad_input(q_proj, q_mask)

#             kv = torch.stack([k_proj, v_proj], dim=2)  # (b, l_k, 2, d)
#             kv, _, cu_k, max_k, _ = unpad_input(kv, padding_mask)

#             q_packed = q_packed.view(-1, self.num_heads, self.head_dim)
#             kv = kv.view(-1, 2, self.num_heads, self.head_dim)

#             out_unpad = flash_attn_varlen_kvpacked_func(
#                 q_packed,
#                 kv,
#                 cu_q,
#                 cu_k,
#                 max_q,
#                 max_k,
#                 dropout_p=dropout_p,
#                 softmax_scale=None,
#                 causal=causal
#             )

#         # Re-pad output and apply final projection
#         out = out_unpad.reshape(-1, self.embed_dim)
#         out = pad_input(out, q_indices if is_cross else indices, b, l_q)

#         return self.out_proj(out)

# class TransformerEncoderLayer(nn.Module):
#     def __init__(
#         self,
#         embed_size: int,
#         nhead: int,
#         dim_feedforward: int = 2048,
#         dropout: float = 0.1,
#         activation: nn.Module = torch.nn.functional.gelu,
#         layer_norm_eps: float = 1e-5,
#         norm_first: bool = False,
#         bias: bool = True,
#         device: Union[int, str, None] = None,
#         dtype=None,
#     ):
#         super().__init__()
#         factory_kwargs = {"device": device, "dtype": dtype}

#         self.self_attn = FlashMHA(
#             embed_size,
#             nhead,
#             dropout=dropout,
#             **factory_kwargs
#         )

#         self.linear1 = nn.Linear(embed_size, dim_feedforward, bias=bias, **factory_kwargs)
#         self.linear2 = nn.Linear(dim_feedforward, embed_size, bias=bias, **factory_kwargs)

#         self.norm1 = nn.LayerNorm(embed_size, eps=layer_norm_eps, bias=bias, **factory_kwargs)
#         self.norm2 = nn.LayerNorm(embed_size, eps=layer_norm_eps, bias=bias, **factory_kwargs)

#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout = nn.Dropout(dropout)

#         self.activation = activation
#         self.norm_first = norm_first

#     def _sa_block(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor], is_causal: bool) -> torch.Tensor:
#         x = self.self_attn(x, x, x, causal=is_causal, padding_mask=padding_mask)
#         return self.dropout1(x)

#     def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         return self.dropout2(x)

#     def forward(
#         self,
#         src: torch.Tensor,
#         src_pad_mask: Optional[torch.Tensor] = None,
#         is_causal: bool = False
#     ) -> torch.Tensor:
#         """
#         Args:
#             src: Input tensor of shape (batch_size, seq_len, embed_size)
#             src_pad_mask: Padding mask of shape (batch_size, seq_len)
#             is_causal: Whether to apply causal masking
#         """
#         x = src
#         if self.norm_first:
#             x = x + self._sa_block(self.norm1(x), src_pad_mask, is_causal)
#             x = x + self._ff_block(self.norm2(x))
#         else:
#             x = self.norm1(x + self._sa_block(x, src_pad_mask, is_causal))
#             x = self.norm2(x + self._ff_block(x))
#         return x


# class TransformerDecoderLayer(nn.Module):
#     def __init__(
#         self,
#         embed_size: int,
#         nhead: int,
#         dim_feedforward: int = 2048,
#         is_kan: bool = False,
#         kan_grid_size: int = 8,
#         kan_ff_dims: Optional[list] = None,
#         dropout: float = 0.1,
#         activation: nn.Module = torch.nn.functional.gelu,
#         layer_norm_eps: float = 1e-5,
#         norm_first: bool = False,
#         bias: bool = True,
#         device: Union[int, str, None] = None,
#         dtype=None,
#     ):
#         super().__init__()
#         factory_kwargs = {"device": device, "dtype": dtype}

#         self.is_kan = is_kan

#         self.self_attn = FlashMHA(embed_size, nhead, dropout=dropout, **factory_kwargs)
#         self.cross_attn = FlashMHA(embed_size, nhead, dropout=dropout, **factory_kwargs)

#         if self.is_kan:
#             self.kan_ff = KANFeedForwardBlock(embed_size, kan_ff_dims, grid_size=kan_grid_size, device=device)
#         else:
#             self.linear1 = nn.Linear(embed_size, dim_feedforward, bias=bias, **factory_kwargs)
#             self.linear2 = nn.Linear(dim_feedforward, embed_size, bias=bias, **factory_kwargs)
#             self.dropout = nn.Dropout(dropout)

#         self.norm1 = nn.LayerNorm(embed_size, eps=layer_norm_eps, bias=bias, **factory_kwargs)
#         self.norm2 = nn.LayerNorm(embed_size, eps=layer_norm_eps, bias=bias, **factory_kwargs)

#         if self.is_kan and not norm_first:
#             self.norm3 = nn.LayerNorm(kan_ff_dims[-1], eps=layer_norm_eps, bias=bias, **factory_kwargs)
#         else:
#             self.norm3 = nn.LayerNorm(embed_size, eps=layer_norm_eps, bias=bias, **factory_kwargs)

#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)

#         self.activation = activation
#         self.norm_first = norm_first

#     def _sa_block(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor], is_causal: bool) -> torch.Tensor:
#         x = self.self_attn(x, padding_mask=padding_mask, causal=is_causal)
#         return self.dropout1(x)

#     def _cross_attn_block(
#         self,
#         x: torch.Tensor,
#         memory: torch.Tensor,
#         padding_mask: Optional[torch.Tensor],
#         is_causal: bool
#     ) -> torch.Tensor:
#         x = self.cross_attn(x, k=memory, v=memory, padding_mask=padding_mask, is_cross=True, causal=is_causal)
#         return self.dropout2(x)

#     def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
#         if self.is_kan:
#             x = self.kan_ff(x)
#         else:
#             x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         return self.dropout3(x)

#     def forward(
#         self,
#         tgt: torch.Tensor,
#         memory: torch.Tensor,
#         tgt_padding_mask: Optional[torch.Tensor] = None,
#         memory_padding_mask: Optional[torch.Tensor] = None,
#         tgt_is_causal: bool = True,
#         memory_is_causal: bool = False,
#     ) -> torch.Tensor:
#         """
#         Args:
#             tgt: Target sequence tensor (batch_size, tgt_len, embed_size)
#             memory: Encoder output tensor (batch_size, src_len, embed_size)
#             tgt_padding_mask: Mask for target tokens (batch_size, tgt_len)
#             memory_padding_mask: Mask for memory tokens (batch_size, src_len)
#             tgt_is_causal: Whether to apply causal masking to self-attn
#             memory_is_causal: Whether to apply causal masking to cross-attn
#         """
#         x = tgt
#         if self.norm_first:
#             x = x + self._sa_block(self.norm1(x), tgt_padding_mask, tgt_is_causal)
#             x = x + self._cross_attn_block(self.norm2(x), memory, memory_padding_mask, memory_is_causal)
#             x = self._ff_block(self.norm3(x)) if self.is_kan else x + self._ff_block(self.norm3(x))
#         else:
#             x = self.norm1(x + self._sa_block(x, tgt_padding_mask, tgt_is_causal))
#             x = self.norm2(x + self._cross_attn_block(x, memory, memory_padding_mask, memory_is_causal))
#             x = self.norm3(self._ff_block(x)) if self.is_kan else self.norm3(x + self._ff_block(x))

#         return x

# class TransformerEncoder(nn.Module):
#     """Stack of Transformer encoder layers."""

#     def __init__(self, encoder_layer: nn.Module, num_layers: int):
#         super().__init__()
#         self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

#     def forward(self, src, src_padding_mask=None, is_causal=False):
#         output = src
#         for layer in self.layers:
#             output = layer(output, src_pad_mask=src_padding_mask, is_causal=is_causal)
#         return output


# class TransformerDecoder(nn.Module):
#     """Stack of Transformer decoder layers with optional KAN layer."""

#     def __init__(self, decoder_layer: nn.Module, decoder_kan_layer: Optional[nn.Module], num_layers: int):
#         super().__init__()
#         self.layers = nn.ModuleList()

#         if decoder_kan_layer is not None:
#             num_layers -= 1

#         self.layers.extend([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

#         if decoder_kan_layer is not None:
#             self.layers.append(decoder_kan_layer)

#     def forward(
#         self,
#         tgt,
#         memory,
#         tgt_padding_mask=None,
#         memory_padding_mask=None,
#         tgt_is_causal=True,
#         memory_is_causal=False,
#     ):
#         output = tgt
#         for layer in self.layers:
#             output = layer(
#                 output,
#                 memory,
#                 tgt_padding_mask=tgt_padding_mask,
#                 memory_padding_mask=memory_padding_mask,
#                 tgt_is_causal=tgt_is_causal,
#                 memory_is_causal=memory_is_causal,
#             )
#         return output


# class Transformer(nn.Module):
#     """Transformer model composed of encoder and decoder stacks."""

#     def __init__(
#         self,
#         embed_size: int,
#         nhead: int,
#         num_encoder_layers: int = 6,
#         num_decoder_layers: int = 6,
#         dim_feedforward: int = 2048,
#         is_kan: bool = False,
#         kan_ff_dims: Optional[list] = None,
#         kan_grid_size: Optional[int] = None,
#         dropout: float = 0.1,
#         activation=nn.functional.gelu,
#         layer_norm_eps: float = 1e-5,
#         norm_first: bool = False,
#         bias: bool = True,
#         device: Union[int, str, None] = 'cpu',
#         dtype=None,
#     ):
#         super().__init__()

#         encoder_layer = TransformerEncoderLayer(
#             embed_size,
#             nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation=activation,
#             layer_norm_eps=layer_norm_eps,
#             norm_first=norm_first,
#             bias=bias,
#             device=device,
#             dtype=dtype
#         )
#         self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

#         decoder_layer = TransformerDecoderLayer(
#             embed_size,
#             nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation=activation,
#             layer_norm_eps=layer_norm_eps,
#             norm_first=norm_first,
#             bias=bias,
#             device=device,
#             dtype=dtype
#         )

#         decoder_kan_layer = None
#         if is_kan:
#             if kan_grid_size is None or kan_grid_size <= 0:
#                 raise ValueError("kan_grid_size must be a positive integer when is_kan is True")
#             if kan_ff_dims is None:
#                 raise ValueError("kan_ff_dims is required when is_kan is True")

#             decoder_kan_layer = TransformerDecoderLayer(
#                 embed_size,
#                 nhead,
#                 dim_feedforward=dim_feedforward,
#                 is_kan=True,
#                 kan_ff_dims=kan_ff_dims,
#                 kan_grid_size=kan_grid_size,
#                 dropout=dropout,
#                 activation=activation,
#                 layer_norm_eps=layer_norm_eps,
#                 norm_first=norm_first,
#                 bias=bias,
#                 device=device,
#                 dtype=dtype
#             )

#         self.decoder = TransformerDecoder(decoder_layer, decoder_kan_layer, num_decoder_layers)

#     def forward(
#         self,
#         src,
#         tgt,
#         src_padding_mask=None,
#         tgt_padding_mask=None,
#         memory_padding_mask=None,
#         src_is_causal=False,
#         tgt_is_causal=True,
#         memory_is_causal=False,
#     ):
#         memory = self.encoder(src, src_padding_mask=src_padding_mask, is_causal=src_is_causal)
#         output = self.decoder(
#             tgt,
#             memory,
#             tgt_padding_mask=tgt_padding_mask,
#             memory_padding_mask=memory_padding_mask,
#             tgt_is_causal=tgt_is_causal,
#             memory_is_causal=memory_is_causal,
#         )
#         return output


# class Model(nn.Module):
#     """
#     Transformer-based model for sequence-to-sequence tasks.

#     Args:
#         num_encoder_layers (int): Number of encoder layers.
#         num_decoder_layers (int): Number of decoder layers.
#         embed_size (int): Embedding dimension.
#         nhead (int): Number of attention heads.
#         src_vocab_size (int): Source vocabulary size.
#         tgt_vocab_size (int): Target vocabulary size.
#         dim_feedforward (int, optional): Feedforward network dimension. Defaults to 512.
#         dropout (float, optional): Dropout rate. Defaults to 0.1.
#         is_kan (bool, optional): Use KAN in the decoder. Defaults to False.
#         kan_ff_dims (list, optional): Feedforward dimensions for KAN. Required if is_kan is True.
#         kan_grid_size (int, optional): Grid size for KAN.
#         device (str, optional): Device to run the model on. Defaults to 'cpu'.
#     """

#     def __init__(
#         self,
#         num_encoder_layers: int,
#         num_decoder_layers: int,
#         embed_size: int,
#         nhead: int,
#         src_vocab_size: int,
#         tgt_vocab_size: int,
#         dim_feedforward: int = 512,
#         dropout: float = 0.1,
#         is_pre_norm: bool = False,
#         is_kan: bool = False,
#         kan_ff_dims: Optional[list] = None,
#         kan_grid_size: int = 8,
#         device: Union[int, str, None] = None,
#         dtype=None,
#     ):
#         super().__init__()
#         self.transformer = Transformer(
#             embed_size=embed_size,
#             nhead=nhead,
#             num_encoder_layers=num_encoder_layers,
#             num_decoder_layers=num_decoder_layers,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             norm_first=is_pre_norm,
#             is_kan=is_kan,
#             kan_ff_dims=kan_ff_dims,
#             kan_grid_size=kan_grid_size,
#             device=device,
#             dtype=dtype
#         )

#         self.generator = nn.Linear(kan_ff_dims[-1] if is_kan else embed_size, tgt_vocab_size)

#         self.src_tok_emb = TokenEmbedding(src_vocab_size, embed_size)
#         self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, embed_size)
#         self.positional_encoding = PositionalEncoding(embed_size, dropout=dropout)
    
#     def encode(
#             self,
#             src: torch.Tensor,
#             src_padding_mask: Optional[torch.Tensor] = None,
#             src_is_causal: bool = False,
#         ) -> torch.Tensor:

#         """
#         Runs the source sequence through the encoder stack.

#         Args:
#             src: (B, S, E) tensor  — sequence length × batch × embedding
#             src_padding_mask: (B, S) mask where False/0
#             src_is_causal: if True, apply a causal mask inside the encoder

#         Returns:
#             memory: (B, S, E) – encoder outputs 
#         """
#         src = self.positional_encoding(self.src_tok_emb(src))            
#         return self.transformer.encoder(src, src_padding_mask=src_padding_mask, is_causal=src_is_causal)

#     def decode(
#         self,
#         tgt: torch.Tensor,
#         memory: torch.Tensor,
#         tgt_padding_mask: Optional[torch.Tensor] = None,
#         memory_padding_mask: Optional[torch.Tensor] = None,
#         tgt_is_causal: bool = True,
#         memory_is_causal: bool = False,
#     ) -> torch.Tensor:
#         """
#         Runs the target sequence through the decoder stack using encoder memory.

#         Args:
#             tgt: (B, T, E) tensor
#             memory: (B, S, E) – output of `self.encode`
#             tgt_padding_mask: (B, T) – mask where False/0
#             memory_padding_mask: (B, S) – mask where False/0
#             tgt_is_causal: apply causal mask to prevent peeking ahead
#             memory_is_causal: causal mask between tgt & memory

#         Returns:
#             output: (B, T, E) – decoder outputs ready for projection to logits
#         """
#         tgt = self.positional_encoding(self.tgt_tok_emb(tgt))
#         return self.transformer.decoder(
#             tgt,
#             memory,
#             tgt_padding_mask=tgt_padding_mask,
#             memory_padding_mask=memory_padding_mask,
#             tgt_is_causal=tgt_is_causal,
#             memory_is_causal=memory_is_causal,
#         )

#     def forward(
#         self,
#         src: torch.Tensor,
#         tgt: torch.Tensor,
#         src_padding_mask: Optional[torch.Tensor] = None,
#         tgt_padding_mask: Optional[torch.Tensor] = None,
#         memory_padding_mask: Optional[torch.Tensor] = None,
#         src_is_causal: bool = False,
#         tgt_is_causal: bool = True,
#         memory_is_causal: bool = False,
#     ):
#         """
#         Forward pass of the model.

#         Args:
#             src (Tensor): Source input.
#             tgt (Tensor): Target input.
#             src_padding_mask (Tensor, optional): Source padding mask.
#             tgt_padding_mask (Tensor, optional): Target padding mask.
#             memory_padding_mask (Tensor, optional): Memory padding mask.
#             src_is_causal (bool, optional): Whether the source is causal. Defaults to False.
#             tgt_is_causal (bool, optional): Whether the target is causal. Defaults to True.
#             memory_is_causal (bool, optional): Whether memory is causal. Defaults to False.

#         Returns:
#             Tensor: Output logits.
#         """
#         src_emb = self.positional_encoding(self.src_tok_emb(src))
#         tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

#         output = self.transformer(
#             src_emb,
#             tgt_emb,
#             src_padding_mask=src_padding_mask,
#             tgt_padding_mask=tgt_padding_mask,
#             memory_padding_mask=memory_padding_mask,
#             src_is_causal=src_is_causal,
#             tgt_is_causal=tgt_is_causal,
#             memory_is_causal=memory_is_causal,
#         )

#         return self.generator(output)