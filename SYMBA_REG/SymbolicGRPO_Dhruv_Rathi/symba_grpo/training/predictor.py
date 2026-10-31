"""Grammar-constrained beam decoding for complete prefix expressions."""

from typing import List, Tuple

import torch
import torch.nn.functional as F

import sympy as sp

from symba_grpo.tokenizers.prefix import NARGS, OPERATORS_INV, prefix_to_sympy
from symba_grpo.tokenizers.vocab import Vocab

DEFAULT_MAX_DECODE_LEN = 256


def allowed_next_token_ids(vocab: Vocab, ids: List[int], num_vars: int) -> List[int]:
    """Return grammar-valid next tokens for one partial prefix trajectory."""
    if not ids or ids[0] != vocab.bos_id:
        return []
    operators = set(OPERATORS_INV)
    digits = set(map(str, range(10)))
    terminals = {"E", "pi", *(f"s_{index}" for index in range(1, num_vars + 1))}
    starts = operators | {"s+", "s-"} | terminals
    pending = 1
    in_integer = False
    integer_digits = 0

    def consume(token: str) -> bool:
        nonlocal pending, in_integer, integer_digits
        if pending <= 0:
            return False
        if token in operators:
            pending += NARGS[token] - 1
        elif token in {"s+", "s-"}:
            in_integer = True
            integer_digits = 0
        elif token in terminals:
            pending -= 1
        else:
            return False
        return True

    for index in ids[1:]:
        token = vocab.decode_id(index)
        if token in {"<pad>", "<bos>", "<eos>", "<unk>"}:
            return []
        if in_integer:
            if token in digits:
                integer_digits += 1
                continue
            if integer_digits == 0:
                return []
            pending -= 1
            in_integer = False
            if not consume(token):
                return []
        elif not consume(token):
            return []

    if in_integer:
        if integer_digits == 0:
            allowed = digits
        else:
            after_integer = pending - 1
            allowed = digits | ({"<eos>"} if after_integer == 0 else starts)
    else:
        allowed = {"<eos>"} if pending == 0 else starts
    return sorted(vocab.symbol2id[token] for token in allowed if token in vocab.symbol2id)


def grammar_sequence_logprobs(
    logits: torch.Tensor,
    prefix_ids: torch.Tensor,
    target_ids: torch.Tensor,
    vocab: Vocab,
    num_vars: int,
    temperature: float = 1.0,
    padded: bool = False,
) -> torch.Tensor:
    """Score trajectories under the same constrained policy used to sample them.

    Each next-token distribution is renormalized over the grammar-valid actions
    for its exact prefix. Invalid target actions fail closed rather than being
    assigned probability under a different, unconstrained policy.
    """
    if logits.ndim != 3 or prefix_ids.shape != target_ids.shape or logits.shape[:2] != prefix_ids.shape:
        raise ValueError("logits and trajectory tensors have inconsistent shapes")
    value = int(num_vars)
    if not 1 <= value <= 11:
        raise ValueError(f"num_vars must lie in [1, 11], received {num_vars!r}")
    scale = max(float(temperature), 1e-6)
    totals = []
    for batch in range(prefix_ids.shape[0]):
        total = logits.new_zeros(())
        for position in range(prefix_ids.shape[1]):
            target = int(target_ids[batch, position].item())
            if padded and target == vocab.pad_id:
                continue
            prefix = prefix_ids[batch, : position + 1].detach().cpu().tolist()
            allowed = allowed_next_token_ids(vocab, prefix, value)
            if target not in allowed:
                raise ValueError("trajectory contains an action forbidden by the prefix grammar")
            allowed_tensor = torch.tensor(allowed, device=logits.device, dtype=torch.long)
            constrained = F.log_softmax(logits[batch, position, allowed_tensor] / scale, dim=-1)
            total = total + constrained[allowed.index(target)]
        totals.append(total)
    return torch.stack(totals)


def full_vocabulary_sequence_logprobs(
    logits: torch.Tensor,
    prefix_ids: torch.Tensor,
    target_ids: torch.Tensor,
    vocab: Vocab,
    num_vars: int,
    temperature: float = 1.0,
    padded: bool = False,
) -> torch.Tensor:
    """Score grammar-valid trajectories without legal-set renormalization."""
    if logits.ndim != 3 or prefix_ids.shape != target_ids.shape or logits.shape[:2] != prefix_ids.shape:
        raise ValueError("logits and trajectory tensors have inconsistent shapes")
    value = int(num_vars)
    if not 1 <= value <= 11:
        raise ValueError(f"num_vars must lie in [1, 11], received {num_vars!r}")
    log_probs = F.log_softmax(logits / max(float(temperature), 1e-6), dim=-1)
    totals = []
    for batch in range(prefix_ids.shape[0]):
        total = logits.new_zeros(())
        for position in range(prefix_ids.shape[1]):
            target = int(target_ids[batch, position].item())
            if padded and target == vocab.pad_id:
                continue
            prefix = prefix_ids[batch, : position + 1].detach().cpu().tolist()
            if target not in allowed_next_token_ids(vocab, prefix, value):
                raise ValueError("trajectory contains an action forbidden by the prefix grammar")
            total = total + log_probs[batch, position, target]
        totals.append(total)
    return torch.stack(totals)


def generate_square_subsequent_mask(sz: int, device) -> torch.Tensor:
    mask = torch.triu(torch.ones((sz, sz), device=device) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
    return mask


class Predictor:
    """Shared source encoder and prefix-grammar validation."""

    def __init__(self, model, vocab: Vocab, device: str = "cpu"):
        self.model = model
        self.vocab = vocab
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.accounting = {"beam_decoder_forwards": 0, "beam_returned_tokens": 0}

    def _encode_src(self, src, num_array):
        src_mask = torch.zeros(src.shape[1], src.shape[1], device=self.device).float()
        if src.dim() == 3:
            src_pad = (src == 0).all(dim=-1).float()
        else:
            src_pad = (src == 0).float()
        src_pad = src_pad.masked_fill(src_pad == 1, float("-inf")).to(self.device)
        return self.model.encode(src, num_array, src_mask, src_pad)

    @staticmethod
    def _validate_num_variables(num_vars: int) -> int:
        """Validate the equation metadata used by the prefix grammar mask."""
        value = int(num_vars)
        if not 1 <= value <= 11:
            raise ValueError(f"num_vars must lie in [1, 11], received {num_vars!r}")
        return value

class BeamPredictor(Predictor):
    """Beam search decoder."""

    def _valid_completion(self, sequence: torch.Tensor) -> bool:
        try:
            ids = sequence.squeeze(0).tolist()
            prefix_to_sympy(self.vocab.decode_generated(ids), evaluate=False)
            return True
        except Exception:
            return False

    @torch.no_grad()
    def beam_search(
        self,
        src: torch.Tensor,
        num_array: torch.Tensor,
        num_vars: int,
        beam_size: int = 5,
        num_equations: int = 20,
        max_len: int = DEFAULT_MAX_DECODE_LEN,
        length_penalty: float = 0.6,
    ) -> List[Tuple[torch.Tensor, float]]:
        src = src.to(self.device)
        num_array = num_array.to(self.device)

        memory = self._encode_src(src, num_array)
        num_vars = self._validate_num_variables(num_vars)

        beams = [(
            torch.full((1, 1), self.vocab.bos_id, dtype=torch.long, device=self.device),
            0.0,
        )]
        completed = []

        for _ in range(max_len - 1):
            new_beams = []
            for ys, score in beams:
                tgt_mask = generate_square_subsequent_mask(ys.size(1), self.device)
                self.accounting["beam_decoder_forwards"] += 1
                out = self.model.decode(ys, memory, tgt_mask)
                logits = self.model.generator(out[:, -1])
                allowed = allowed_next_token_ids(self.vocab, ys.squeeze(0).tolist(), num_vars)
                if not allowed:
                    continue
                # Preserve the likelihood geometry of the frozen checkpoint:
                # normalize over its full vocabulary, then make illegal actions
                # unavailable without renormalizing the remaining scores.
                log_probs = F.log_softmax(logits, dim=-1)
                masked = torch.full_like(logits, -torch.inf)
                masked[:, allowed] = log_probs[:, allowed]

                width = min(beam_size, len(allowed))
                top_lps, top_ids = masked.topk(width, dim=-1)
                for i in range(width):
                    nw = top_ids[0, i].item()
                    ns = score + top_lps[0, i].item()
                    new_seq = torch.cat(
                        [ys, torch.tensor([[nw]], device=self.device)], dim=1
                    )
                    if nw == self.vocab.eos_id:
                        if self._valid_completion(new_seq):
                            completed.append((new_seq, ns))
                    else:
                        new_beams.append((new_seq, ns))

            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]

            if len(completed) >= num_equations or not beams:
                break

        def _score(seq_and_logp):
            seq, logp = seq_and_logp
            length = seq.size(1)
            return logp / (length ** length_penalty)

        completed.sort(key=_score, reverse=True)
        return completed[:num_equations]

    def predict_candidates(
        self,
        src: torch.Tensor,
        num_array: torch.Tensor,
        num_vars: int,
        beam_size: int = 5,
        num_equations: int = 20,
    ) -> List[List[str]]:
        """Return list of decoded token lists."""
        results = self.beam_search(src, num_array, num_vars, beam_size, num_equations)
        candidates = []
        for seq_tensor, score in results:
            ids = seq_tensor.squeeze(0).tolist()
            try:
                tokens = self.vocab.decode_generated(ids)
                candidates.append(tokens)
                self.accounting["beam_returned_tokens"] += len(tokens)
            except ValueError:
                continue
        return candidates

    def predict_sympy(
        self,
        src: torch.Tensor,
        num_array: torch.Tensor,
        num_vars: int,
        beam_size: int = 5,
        num_equations: int = 20,
    ) -> List[sp.Expr]:
        """Return list of SymPy expressions from beam search."""
        candidates = self.predict_candidates(src, num_array, num_vars, beam_size, num_equations)
        exprs = []
        for tokens in candidates:
            try:
                expr = prefix_to_sympy(tokens, evaluate=False)
                exprs.append(expr)
            except Exception:
                continue
        return exprs
