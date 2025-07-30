from typing import Optional
from functools import partial

import torch
import torch.nn as nn

from mamba_ssm.utils.generation import InferenceParams
from flash_attn.modules.mha import MHA
from x_transformers.x_transformers import Attention 
from x_transformers.x_transformers import RelativePositionBias as xRelativePositionBias 


from math import sqrt

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class FlashCrossAttentionWrapper(nn.Module):

    def __init__(
        self,
        layer_idx=0,
        d_model: int = 512,
        n_head: int = 8,
        rms_norm: bool = True,
        norm_epsilon: float = 1e-5,
        residual_in_fp32=True,
        fused_add_norm=True,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.attention = Attention(
            dim=d_model,
            dim_head=d_model // n_head,
            heads=n_head,
            causal=False,  # x attention should not be causal
            dropout=dropout,
            flash=True,
        )

        self.rel_pos = xRelativePositionBias(
            scale= (1 / d_model // n_head) ** 0.5,
            heads=n_head,
        )

        # self.attention = MHA(
        #     embed_dim=d_model,
        #     num_heads=n_head,
        #     cross_attn=True,
        #     # causal=True,
        #     dropout=dropout,
        #     layer_idx=layer_idx,
        #     # use_flash_attn=True, # not supported if cross_attn
        # )

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

        cache = None

        if inference_params is not None and inference_params.seqlen_offset > 0:
            cache = inference_params.key_value_memory_dict[self.layer_idx]
            cache.seqlen_offset = inference_params.seqlen_offset

        hidden_states, cache = self.attention.forward(
            x=hidden_states,
            mask=mask,
            context=context,
            context_mask=context_mask,
            # rel_pos=self.rel_pos,
            return_intermediates=True,
            cache=cache,
        )

        if inference_params is not None:
            inference_params.key_value_memory_dict[self.layer_idx] = cache
            # hidden_states = (
            #     hidden_states[:, -1:, :]
            #     if inference_params.seqlen_offset > 0
            #     else hidden_states
            # )

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return None