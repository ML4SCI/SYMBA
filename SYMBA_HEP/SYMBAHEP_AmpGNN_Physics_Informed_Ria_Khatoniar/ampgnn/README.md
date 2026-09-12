# Data semantics (diagram-pair terms)

Each dataset row is one **diagram pair** (i ≤ j) of a process, in the format

```
Interaction: <in> to <out> : pair i-j : <diagram_i> : <diagram_j> : <M_i> : <M_j> : <term>
```

where the target `<term>` (always the last field) is the diagonal `|M_i|^2` when
i == j, or **one ordering of the interference term** `M_i * conj(M_j)` when
i < j (the reverse ordering is its complex conjugate and is not stored). Summing
a process's predicted terms plus the conjugates of its off-diagonal terms
reconstructs the full process squared amplitude `|sum_i M_i|^2`.

The pipeline maps this onto the pair-graph architecture directly: the amplitude
side of each pair graph is diagram *i*, the conjugate side is diagram *j*
(`build_pair_graphs(built_i, built_j, ...)`), joined by cross edges under the
valid leg-pairing permutations. Diagonal rows reduce to the original
same-diagram construction. The raw `M_i` / `M_j` amplitude fields are not used.
Legacy single-diagram rows (one `Vertex` field) still load, with the conjugate
side defaulting to diagram *i*.

> Notes: interference targets are generally **complex-valued** (expect `i`
> factors); mass canonicalization uses the union of both diagrams' internal
> species; error rows ("Error on pair ...", "Error evaluating ...") are filtered
> at load time. Check `--max_seq_len_cap` / `--dec_max_len` against the new
> target-length distribution — too small a cap silently filters long examples
> (logged as "Filtered by cap").

# Example run:

The default output is the non-autoregressive structured term-slot decoder with
cross-entropy. Use `--output_mode sequence --loss_mode ce` for the flat
non-autoregressive sequence baseline. Both modes use the same mass rewriting,
symbolic simplification, and canonicalization before the structured mode
decomposes the target into coefficient, numerator, and denominator slots.

```
!python -m ampgnn.train --model QED --data_dir ../data --out_dir runs/qed \
  --scheduler cosine_warmup --warmup_steps 2140 --gradient_clip_val 1. \
  --precision 16-mixed --max_perms 6 --label_smoothing 0.01 \
  --epochs 400 --batch_size 32 --lr 5e-4 --num_workers 11 \
  --weight_decay 0.05 --dec_use_len_mask --length_loss_weight 1.0 \
  --enc_hid 128 --enc_layers 3 --enc_heads 8 --enc_dropout 0.1 \
  --d_model 256 --dec_layers 3 --dec_nhead 8 --dec_dropout 0.1 \
  --devices auto --strategy auto --dec_max_len 1024 --max_seq_len_cap 1024
```

## Sequence termination: EOS-based vs. length-masked

`--dec_use_len_mask` toggles between two decoding regimes (the length-prediction
head and its auxiliary loss are enabled in both):

**Default (flag off) — EOS-based termination.** Train/val/test decode the full
(batch-padded) width and rely on the model emitting `[EOS]` to terminate:

- **Loss** is cross-entropy with `[PAD]` ignored (`ignore_index`), so the model is
  trained on every content token plus the final `[EOS]`, and is not penalized on
  padding after it.
- **Token / sequence accuracy** ignore everything beyond the first `[EOS]` in the
  target (see `_build_eval_mask`), so a sequence is correct only if every content
  token matches *and* `[EOS]` is predicted at the right position.

**With `--dec_use_len_mask` — length-masked decoding.** The decoder's non-causal
self-attention is masked to per-example lengths: gold lengths from `y_in` during
train/val/test, predicted lengths in `generate()`. Note this **leaks the gold
length at eval**, so its val/test numbers are not directly comparable to the
EOS-based ones.

> The `CausalBlock` `key_padding_mask` convention bug (it used the
> `nn.MultiheadAttention` "True = ignore" convention with
> `F.scaled_dot_product_attention`, which expects "True = keep") has been fixed,
> so the length-masked path now attends to the valid region instead of inverting
> it. `--length_loss_weight` controls the auxiliary length head (the example
> below uses `1.0`).
