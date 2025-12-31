from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.generation import InferenceParams


import json
from transformers.optimization import get_inverse_sqrt_schedule

import evaluate


from .helpers.flash_cross_attention import FlashCrossAttentionWrapper
from .helpers.cross_attention import CrossAttentionWrapper
from .helpers.ffn import FeedForwardWrapper
from .helpers.mamba import MambaDecoder, MixerModel

# change the hardcoded 300 in the below code
class MambaEncDec(nn.Module):
    is_encoder_decoder = True
    is_concat = False  # FIXME remove
    model_name = "mamba_encdec"
    configs = {
        "default": {
            "enc_n_layer": 4,
            # mamba config
            "d_model": 512,
            "n_layer": 6,
            "rms_norm": True,
            "fused_add_norm": True,
            "use_fast_path": False,
            # "learning_rate": 7e-4,
            # "warmup_steps": 4000,
            # "weight_decay": 0.001,
            # "devices": 'cuda:0'
        }
    }

    def __init__(
        self,
        config=None,
        # tokenizer=PreTrainedTokenizerFast,
        src_vocab_size=459,
        tgt_vocab_size=59,
        d_model=None,
        dec_n_layer=None,
        enc_n_layer=None,
        rms_norm=None,
        fused_add_norm=None,
        use_fast_path=None,
        dropout=None,
        use_padding=None,
        precision="32-true",
        test_per_sample=True,
        test=False,
        test_suffix="",
        **kwargs,
    ):
        super().__init__()

        self.config = MambaConfig(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_layer=dec_n_layer,
            rms_norm=rms_norm,
            fused_add_norm=fused_add_norm,
            # use_fast_path=use_fast_path,
            ssm_cfg={"dropout": dropout},
        )

        self.encoder = MixerModel(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_layer=enc_n_layer,
            rms_norm=rms_norm,
            fused_add_norm=fused_add_norm,
            use_fast_path=use_fast_path,
            ssm_cfg={"dropout": dropout},
            layer_dict={},
        )

        self.layers = (0, 3, 6, 9, 12, 15)
        x_attention_layers = [
            (i, FlashCrossAttentionWrapper) for i in (1, 4, 7, 10, 13, 16)
        ]
        ffn_layers = [(i, FeedForwardWrapper) for i in (2, 5, 8, 11, 14, 17)]

        layer_dict = dict(x_attention_layers + ffn_layers)

        self.decoder = MambaDecoder(
            config=self.config,
            layer_dict=layer_dict,
            layer_kwargs={"dropout":0.1}
        )
        self.generator = self.decoder.generator
        # self.tokenizer = tokenizer
        self.bleu = evaluate.load("sacrebleu")
        self.config = config
        self.use_padding = use_padding
        dtype_map = {
            "bf16-mixed": torch.bfloat16,
            "16-true": torch.float16,
            "32-true": torch.float32,
        }
        self.precision = dtype_map[precision]

        if test:
            # self.comet = load_comet()
            self.test_per_sample = test_per_sample
            self.test_res = []
            self.test_suffix = test_suffix

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.decoder.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self,
        context_tokens,
        input_ids,
        source_attention_mask=None,
        target_attention_mask=None,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
    ):
        
        b, l = source_attention_mask.shape
        # source_attention_mask = source_attention_mask.reshape(b,l).to(torch.bool)
        source_attention_mask = source_attention_mask.to(torch.bool)
        target_attention_mask = target_attention_mask.to(torch.bool)

        source_vec = self.encoder.forward(
            input_ids=context_tokens,
            mask=source_attention_mask,
        )
        # print(source_vec.dtype, source_attention_mask.dtype)
        cache = self.allocate_inference_cache(
            batch_size=b,
            max_seqlen=300 + l + 1,  # source + BOS
            dtype=self.precision,
        )
        inference_params = InferenceParams(
            max_seqlen=300 + l + 1,
            max_batch_size=b,
            key_value_memory_dict=cache,
        )
        
        # batch, seqlen, dim = self.decoder.backbone.embedding.forward(input_ids).shape
        # conv_state, ssm_state = self.decoder.backbone.layers[0].mixer._get_states_from_cache(inference_params, b)
        # inference_params = None
        # print(conv_state.type(),input_ids.type(), source_vec.type())
        # print(source_attention_mask.type(), target_attention_mask.type())
        # print(position_ids.type())
        # print(num_last_tokens)

        out = self.decoder.forward(
            input_ids,
            context=source_vec,
            context_mask=source_attention_mask,
            attention_mask=target_attention_mask,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
        )
        return self.generator(out)

    def encode(self, src, source_attention_mask):
        memory = self.encoder.forward(
            input_ids=src,
            mask=source_attention_mask,
        )
        
        return memory

    def decode(self, ys, memory, target_attention_mask, source_attention_mask):
        b, l = source_attention_mask.shape
        cache = self.allocate_inference_cache(
            batch_size=b,
            max_seqlen=300 + l + 1,  # source + BOS
            dtype=self.precision,
        )

        inference_params = InferenceParams(
            max_seqlen=300 + l + 1,
            max_batch_size=b,
            key_value_memory_dict=cache,
        )

        out = self.decoder.forward(
                input_ids=ys,
                context=memory,
                # position_ids=position_ids,
                context_mask=source_attention_mask,
                attention_mask=target_attention_mask,
                inference_params=inference_params,
                num_last_tokens=1,
        )
        return out

    def _reorder_cache(self, cache, beam_idx):
        for layer_idx in self.layers:
            device = cache[layer_idx][0].device
            # {0:(conv_state, ssm_state)}
            cache[layer_idx] = (
                cache[layer_idx][0].index_select(0, beam_idx.to(device)),
                cache[layer_idx][1].index_select(0, beam_idx.to(device)),
            )
        return cache