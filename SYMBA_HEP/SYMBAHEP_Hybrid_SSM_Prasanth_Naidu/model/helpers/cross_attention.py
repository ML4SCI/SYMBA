from typing import Optional
from functools import partial

import torch
import torch.nn as nn

from mamba_ssm.utils.generation import InferenceParams
from torchscale.architecture.config import DecoderConfig
from torchscale.component.multihead_attention import MultiheadAttention
from torchscale.component.relative_position_bias import RelativePositionBias

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class CrossAttentionWrapper(nn.Module):

    def __init__(
        self,
        layer_idx=0,
        d_model: int = 512,
        n_head: int = 8,
        rms_norm: bool = True,
        norm_epsilon: float = 1e-5,
        residual_in_fp32=True,
        fused_add_norm=True,
        attention_dropout=0,
        **kwargs,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        args = DecoderConfig(
            rel_pos_buckets=32, max_rel_pos=128, decoder_attention_heads=n_head
        )

        self.attention = MultiheadAttention(
            args,
            d_model,
            num_heads=n_head,
            dropout=attention_dropout,
            self_attention=False,
            encoder_decoder_attention=True,
            subln=args.subln,
        )

        self.cross_attn_relative_position = RelativePositionBias(
            num_buckets=args.rel_pos_buckets,
            max_distance=args.max_rel_pos,
            n_heads=args.decoder_attention_heads,
        )

        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm,
            eps=norm_epsilon,  # **factory_kwargs
        )

        self.norm = norm_cls(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        inference_params: InferenceParams = None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )

        bsz, slen, _ = hidden_states.shape

        incremental_state = None

        if inference_params is not None and inference_params.seqlen_offset == 0:
            slen = 1
            incremental_state = {}

        elif inference_params is not None and inference_params.seqlen_offset > 0:
            slen = inference_params.seqlen_offset
            incremental_state = inference_params.key_value_memory_dict[self.layer_idx]

        cross_attn_rel_pos = self.cross_attn_relative_position(
            batch_size=bsz,
            qlen=slen,
            klen=context.size(1),
        )

        if inference_params is not None and inference_params.seqlen_offset != 0:
            cross_attn_rel_pos = cross_attn_rel_pos[-1:, :, :]

        hidden_states, attn = self.attention.forward(
            query=hidden_states,
            key=context,
            value=context,
            key_padding_mask=context_mask,
            incremental_state=incremental_state,
            # rel_pos=cross_attn_rel_pos,  # none in the default torchscale config
        )

        if inference_params is not None and inference_params.seqlen_offset == 0:
            inference_params.key_value_memory_dict[self.layer_idx] = incremental_state

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return None