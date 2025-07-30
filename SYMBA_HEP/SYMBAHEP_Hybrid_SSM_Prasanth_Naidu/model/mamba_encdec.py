from typing import Any
# import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.generation import InferenceParams


# from utils.beam_search import BeamSearch
import json
# from utils.mt.comet import load_comet
from transformers.optimization import get_inverse_sqrt_schedule
# from transformers import PreTrainedTokenizerFast
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
    
    # def generator(self, logits):

    # def decode(self,)
    # def test_step(self, batch, batch_idx):
    #     """beam search with parallel formulation"""
    #     # num_beams = 1

    #     # # source_tokens, labels, source_attention_mask = (
    #     # #     batch["input_ids"],
    #     # #     batch["labels"],
    #     # #     batch["attention_mask"],
    #     # # )
    #     src_tokens, _, labels, source_attention_mask, _ = batch
    #     batch_size, seq_len = src_tokens.shape
    #     max_length = 350
    #     cache = self.allocate_inference_cache(
    #         batch_size=batch_size,
    #         max_seqlen=max_length + seq_len + 1,  # source + BOS
    #         dtype=self.precision,
    #         # dtype = ,
    #     )
    #     inference_params = InferenceParams(
    #         max_seqlen=max_length + seq_len + 1,
    #         max_batch_size=batch_size,
    #         key_value_memory_dict=cache,
    #     )

    #     done = torch.tensor([False] * batch_size).to(src_tokens.device)
    #     preds = (
    #         torch.ones((batch_size, 1), dtype=torch.long).to(src_tokens.device)
    #         * self.tokenizer.bos_token_id
    #     )

    #     source_vec = self.encoder.forward(
    #         input_ids=src_tokens,
    #         mask=source_attention_mask,
    #     )
    #     # print('Source vec:', source_vec)
    #     position_ids = None

    #     for idx in range(labels.size(1)):

    #         if idx > 0:
    #             last_tokens = preds[:, -1:]  # (B, 1)
    #             position_ids = torch.full(
    #                 (batch_size, 1),
    #                 inference_params.seqlen_offset,
    #                 dtype=torch.long,
    #                 device=src_tokens.device,
    #             )
    #         #### <DEBUG>
    #         # hidden_states = self.decoder.backbone.embedding.forward(preds)
    #         # batch, seqlen, dim = self.decoder.backbone.embedding.forward(preds).shape
    #         # # https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L128
    #         # conv_state, ssm_state = self.decoder.backbone.layers[0].mixer._get_states_from_cache(inference_params, batch)
    #         # xz = self.decoder.backbone.layers[0].mixer.in_proj(hidden_states.squeeze(1))
    #         # print('Conv_state: ',conv_state.dtype, conv_state.type())
    #         # print('xz type: ', xz.dtype, xz.type())
    #         # print('hidden_states: ', hidden_states.dtype, hidden_states.type())
    #         # print(conv_state.dtype, ssm_state.dtype, preds.dtype, position_ids.dtype, source_vec.type())
    #         # print(conv_state.type(),preds.type() if idx == 0 else last_tokens.type())
    #         # print(conv_state.type(), preds.type() if idx == 0 else last_tokens.type(), source_vec.type())
    #         # if idx != 0:
    #         #     print(last_tokens)
    #         # print(source_attention_mask.type(), target_attention_mask.type())
    #         # print(position_ids.type())
    #         # print(num_last_tokens)
    #         #### </ DEBUG>
    #         logits = self.decoder.forward(
    #             input_ids=preds if idx == 0 else last_tokens,
    #             context=source_vec,
    #             position_ids=position_ids,
    #             inference_params=inference_params,
    #             num_last_tokens=1,
    #         )

    #         next_token_logits = logits[:, -1, :]
    #         next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    #         preds = torch.cat((preds, next_token), dim=-1)
    #         inference_params.seqlen_offset += 1
    #         # print(next_token.dtype)
    #         is_eos = next_token == self.tokenizer.eos_token_id
    #         done = done | is_eos.squeeze(-1)

    #         if done.all():
    #             break

    #     # Create a cumulative sum mask where positions after EOS become True
    #     eos_token_id = self.tokenizer.eos_token_id
    #     eos_mask = (preds == eos_token_id).cumsum(dim=1) > 0
    #     preds[eos_mask] = self.tokenizer.pad_token_id
        
    #     # tpreds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
    #     # tlabels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
    #     # bleu_score = self.bleu.compute(predictions=tpreds, references=tlabels)["score"]

    #     # self.log("val_bleu", bleu_score, sync_dist=True)
    #     preds = preds.cpu()
    #     labels = labels.cpu()
        
    #     import gc
    #     # from itertools import islice
    #     # for ts in islice([src_tokens, source_attention_mask, source_vec, done, preds, next_token_logits, next_token, eos_mask],0):
    #     #     ts = ts.cpu()
    #     #     del ts
    #     #     gc.collect()
    #     # gc.collect()
    #     # del src_tokens, source_attention_mask, source_vec
    #     # del done, preds, next_token_logits, next_token
    #     del cache, logits, next_token_logits, inference_params
        
    #     if position_ids is not None:
    #         position_ids = position_ids.cpu()
    #         del position_ids
    #         # gc.collect()
    #     eos_mask = eos_mask.cpu()
    #     del eos_mask
    #     gc.collect()
        
    #     torch.cuda.empty_cache()
        # token_acc = (preds == labels).float().mean().item()
        
        # Sequence accuracy: Mean of sequences where all tokens match
        # seq_acc = (preds == labels).all(dim=1).float().mean().item()
        # return preds, labels #(token_acc, seq_acc)
        # batch_size, seq_len = source_tokens.shape
        # beam_size = num_beams * batch_size
        # input_ids = source_tokens.repeat_interleave(num_beams, dim=0)
        # source_attention_mask = source_attention_mask.repeat_interleave(
        #     num_beams, dim=0
        # )

        # maxseq_len = int(seq_len * 1.5)

        # # cache = self.allocate_inference_cache(
        # #     batch_size=beam_size,
        # #     max_seqlen=maxseq_len + seq_len,
        # #     dtype=self.precision,
        # # )
        # inference_params = InferenceParams(
        #     max_seqlen=maxseq_len + seq_len,
        #     max_batch_size=beam_size,
        #     # key_value_memory_dict=cache,
        # )

        # search = BeamSearch(
        #     tokenizer=self.tokenizer,
        #     batch_size=batch_size,
        #     num_beams=num_beams,
        #     max_length=maxseq_len + seq_len,
        #     device=input_ids.device,
        # )

        # source_vec = self.encoder.forward(
        #     input_ids=input_ids,
        #     mask=source_attention_mask,
        # )

        # position_ids = None
        # preds = (
        #     torch.ones((beam_size, 1), dtype=torch.long).to(input_ids.device)
        #     * self.tokenizer.bos_token_id
        # )

        # for idx in range(maxseq_len):
        #     if idx > 0:
        #         last_tokens = preds[:, -1:]  # (B, 1)
        #         position_ids = torch.full(
        #             (beam_size, 1),
        #             inference_params.seqlen_offset,
        #             dtype=torch.long,
        #             device=input_ids.device,
        #         )

        #     outputs = self.decoder.forward(
        #         input_ids=preds if idx == 0 else last_tokens,
        #         context=source_vec,
        #         position_ids=position_ids,
        #         inference_params=inference_params,
        #         num_last_tokens=1,
        #     )

        #     next_token_logits = outputs[:, -1, :]
        #     preds, cache = search.step(
        #         ids=preds,
        #         logits=next_token_logits,
        #         # cache=inference_params.key_value_memory_dict,
        #         # reorder_cache_fn=self._reorder_cache,
        #     )
        #     inference_params.seqlen_offset += 1
        #     # inference_params.key_value_memory_dict = cache

        #     # generated EOS for all beams
        #     if search.is_done:
        #         break

        # seqs = search.finalize(ids=preds)

        # eos_mask = (seqs == self.tokenizer.eos_token_id).cumsum(dim=1) > 0

        # seqs[eos_mask] = self.tokenizer.pad_token_id

        # tsrcs = self.tokenizer.batch_decode(source_tokens, skip_special_tokens=True)
        # tpreds = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
        # tlabels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # return seqs, labels
        # bleu_score = self.bleu.compute(predictions=tpreds, references=tlabels)["score"]
        # self.log("test_bleu", bleu_score, sync_dist=True)

        # res = self.comet.compute(
        #     sources=tsrcs,
        #     predictions=tpreds,
        #     references=tlabels,
        #     devices=self.config["devices"],
        #     progress_bar=False,
        # )

        # self.log("test_comet", res["mean_score"], sync_dist=True)

        # if self.test_per_sample:
        #     bleu_scores = [
        #         self.bleu.compute(predictions=[tpreds[i]], references=[tlabels[i]])[
        #             "score"
        #         ]
        #         for i in range(batch_size)
        #     ]
        #     self.test_res.append((tsrcs, tpreds, tlabels, bleu_scores, res["scores"]))

        # print(f"bleu: {bleu_score}, comet: {res['mean_score']}")
    #     return bleu_score, res["mean_score"]

    # def on_test_epoch_end(self):
    #     # if self.test_per_sample:
    #     if False:
    #         source, target = self.config["language_pair"]

    #         with open(
    #             f"mt/res/{self.config['dataset']}/{self.config['dataset']}-{source}-{target}-{self.model_name}-{self.test_suffix}.json",
    #             "w",
    #         ) as f:
    #             json.dump(self.test_res, f)
# def training_step(self, batch, batch_idx):
    #     # source, target, source_attention_mask = (
    #     #     batch["input_ids"],
    #     #     batch["labels"],
    #     #     batch["src_attention_mask"],
    #     # )
    #     source, target, _, source_attention_mask, _ = batch
    #     # source, target, source_attention_mask, 

    #     target_attention_mask = (
    #         (target != self.tokenizer.pad_token_id).to(torch.bool).to(source.device)
    #     )
    #     # print(source.type(), source_attention_mask.type())
        
    #     lm_logits = self.forward(
    #         context_tokens=source,
    #         source_attention_mask=source_attention_mask,
    #         target_attention_mask=target_attention_mask,
    #         input_ids=target,
    #     )
        
    #     logits = lm_logits[:, :-1].contiguous()
    #     labels = target[:, 1:].contiguous()

    #     loss = F.cross_entropy(
    #         logits.view(-1, logits.size(-1)),
    #         labels.view(-1),
    #         ignore_index=self.tokenizer.pad_token_id,
    #     )
    #     # self.log("train_loss", loss, sync_dist=True)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     # src_tokens, labels, source_attention_mask = (
    #     #     batch["input_ids"],
    #     #     batch["labels"],
    #     #     batch["src_attention_mask"],
    #     # )
    #     src_tokens, target, _, source_attention_mask, _ = batch

    #     batch_size, seq_len = src_tokens.shape
    #     max_length = 351

    #     target_attention_mask = (
    #         (target != self.tokenizer.pad_token_id).to(torch.bool).to(src_tokens.device)
    #     )
        
    #     lm_logits = self.forward(
    #         context_tokens=src_tokens,
    #         source_attention_mask=source_attention_mask,
    #         target_attention_mask=target_attention_mask,
    #         input_ids=target,
    #     )
        
    #     logits = lm_logits[:, :-1].contiguous()
    #     labels = target[:, 1:].contiguous()
        
    #     loss = F.cross_entropy(
    #         logits.view(-1, logits.size(-1)),
    #         labels.view(-1),
    #         ignore_index=self.tokenizer.pad_token_id,
    #     )

    #     return loss
    def _reorder_cache(self, cache, beam_idx):
        for layer_idx in self.layers:
            device = cache[layer_idx][0].device
            # {0:(conv_state, ssm_state)}
            cache[layer_idx] = (
                cache[layer_idx][0].index_select(0, beam_idx.to(device)),
                cache[layer_idx][1].index_select(0, beam_idx.to(device)),
            )
        return cache

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(
    #         self.parameters(),
    #         lr=self.config["learning_rate"],
    #         weight_decay=self.config["weight_decay"],
    #         fused=True,
    #     )

    #     scheduler = {
    #         "scheduler": get_inverse_sqrt_schedule(
    #             optimizer,
    #             num_warmup_steps=self.config["warmup_steps"],
    #         ),
    #         "interval": "step",
    #     }

    #     return {"optimizer": optimizer, "lr_scheduler": scheduler}