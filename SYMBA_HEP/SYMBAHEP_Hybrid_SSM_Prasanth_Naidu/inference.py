import torch

from .constants import PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX

def greedy_decode(model, device, max_len, src, src_padding_mask, start_symbol, dtype=torch.bfloat16):
    """
    Performs greedy decoding to generate predictions.

    Args:
        src (Tensor): Source tensor of shape (batch_size, src_seq_len).
        src_padding_mask (Tensor): Source padding mask of shape (batch_size, src_seq_len).
        start_symbol (int): Start token index.

    Returns:
        Tensor: Generated token sequence of shape (batch_size, tgt_seq_len).
    """
    src = src.to(device)
    src_padding_mask = src_padding_mask.to(device)
    batch_size = src.size(0)

    ys = torch.full((batch_size, 1), start_symbol, dtype=torch.long, device=device)

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=dtype):
            memory = model.encode(src, src_padding_mask)

            for _ in range(max_len):
                tgt_padding_mask = (ys != PAD_IDX)

                out = model.decode(ys, memory, tgt_padding_mask, src_padding_mask)

                prob = model.generator(out[:, -1, :])  # (batch_size, vocab_size)
                _, next_word = torch.max(prob, dim=1)       # (batch_size,)
                next_word = next_word.unsqueeze(1)          # (batch_size, 1)

                ys = torch.cat([ys, next_word], dim=1)      

                if ((next_word == EOS_IDX) | (next_word == SEP_IDX)).all():
                    break

    return ys