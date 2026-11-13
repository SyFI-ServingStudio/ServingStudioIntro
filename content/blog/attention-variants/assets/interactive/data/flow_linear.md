# Kimi Delta Attention in one linear layer of GLM-5.3-Flash

All the data behind this figure in "Beyond GQA: Characterizing Modern Attention Variants with ServingStudio".

Costs are for one whole layer on one GPU, as in the other figures.

*Prefill: data flow*

- Main row: x (N × 4,096) → Projection → Short conv → Per-block preparation → State pass, block by block → Per-block read-out → Gated norm → Output projection → out (to the MoE FFN)
- Parallel branch: from Projection (f_a, g_a, b) → Gates → joins Per-block preparation (f, b), Delta-rule update and read-out (f), Gated norm (g)
- q goes from Short conv straight to Per-block read-out
- Q Kᵀ goes from Per-block preparation straight to Per-block read-out
- Write: Short conv (last 3 inputs) → Write conv → State, fixed in size
- Write: State pass, block by block (S) → Write S → State, fixed in size
- Short conv reads conv from the cache: pads with last 3
- State pass, block by block reads S from the cache: reads S once

*Prefill at Chunk N = 8,192, Prefix L = 128K: every operation*

| Operation | Kind | Formula | What it does | Operands [axis size] dtype | FLOPs | Bytes | Intensity |
|---|---|---|---|---|---|---|---|
| Projection | GEMM · weights | x · [W_q, W_k, W_v, W_b, W_fa, W_ga] | Turns each token's hidden state into a query, a key and a value for each of the 64 heads, plus the inputs of the gates: one write-strength logit b per head and two 128-number factors f_a, g_a. | Input: x [N 8K, D 4,096] BF16 \| Weight: W_in [D 4,096, 3·H·d+H+2r 24,896] BF16 \| Output: q, k, v [N 8K, 3·H·d 24,576] BF16; b [N 8K, H 64] BF16; f_a, g_a [N 8K, 2r 256] BF16 | 1.67 TFLOP | 648 MiB | 2,461 FLOP/B |
| Short conv | per element | Σ over the last 4 tokens, per channel; SiLU | Mixes each channel of q, k and v with the same channel of the 3 tokens before it. The chunk's first tokens take theirs from the conv state. | Input: q, k, v [N 8K, 3·H·d 24,576] BF16 \| Weight: w_conv [3·H·d 24,576, 4 taps 4] FP32 \| State: conv [3 tokens 3, 3·H·d 24,576] BF16 \| Output: q, k, v [N 8K, 3·H·d 24,576] BF16 | 2.42 GFLOP | 769 MiB | 3.00 FLOP/B |
| Gates | GEMM · weights | f = f_a · W_fb,  g = g_a · W_gb | Expands f_a into f, one number per key channel of each head, and g_a into the output gate g, which the gated norm applies after the core. f is not the decay yet: the core kernel turns it into α, between 0 and 1. b passes by unchanged; the core kernel turns it into β = sigmoid(b). | Input: f_a, g_a [N 8K, 2r 256] BF16 \| Weight: W_fb [r 128, H·d 8,192] BF16; W_gb [r 128, H·d 8,192] BF16 \| Output: f [N 8K, H 64, d_k 128] BF16; g [N 8K, H·d 8,192] BF16 | 34.4 GFLOP | 264 MiB | 124 FLOP/B |
| Per-block preparation | 64-token blocks | per block of 64:  α from f, β from b;  A = β K Kᵀ (lower);  (I + A)⁻¹;  w, u = (I + A)⁻¹ (β K, β V);  Q Kᵀ (lower) | Cuts the chunk into blocks of 64 tokens and works on all blocks at once. It turns f into the decay α and b into the write strength β = sigmoid(b). Inside each block it measures how the 64 writes overlap, K Kᵀ, and solves that triangular system, so the block's 64 one-token writes become one matrix update, w and u. It also takes Q Kᵀ inside the block for the output step. Nothing here reads S. | Input: q, k, v [N 8K, 3·H 192, d 128] BF16; f [N 8K, H 64, d_k 128] BF16; b [N 8K, H 64] BF16 \| Weight: A, bias [H·d 8,192] FP32 \| Output: w, u [N 8K, H 64, d_k+d_v 256] on chip; Q Kᵀ [N 8K, H 64, 64 64] on chip | 27.2 GFLOP | 513 MiB | 50.6 FLOP/B |
| State pass, block by block | S, block to block | for each block in order:  D = u − w S;  S ← Diag(α) S + Kᵀ D | The one serial step: it walks the blocks in order, carrying S from each to the next. For a block, w S is what S already stores for its keys; subtracting it from u leaves the correction D. Then S decays across the block and takes Kᵀ D. S is read once, before the first block, and the S after the last block is written back once. The S each block started from goes on to the output step. | Input: w, u [N 8K, H 64, d_k+d_v 256] on chip \| State: S [H 64, d_k 128, d_v 128] FP32 \| Output: S per block [N/64 128, H 64, d_k 128, d_v 128] on chip; D [N 8K, H 64, d_v 128] on chip; S [H 64, d_k 128, d_v 128] on chip | 34.4 GFLOP | 4.00 MiB | 8,192 FLOP/B |
| Per-block read-out | 64-token blocks | per block:  o = Q S_start + (Q Kᵀ, lower) D | Works on all blocks at once again. A token's output is what S held at its block's start, read with its query and decayed to its position, plus what the earlier tokens of its own block wrote, through Q Kᵀ and D. | Input: q [N 8K, H 64, d_k 128] reused; S per block [N/64 128, H 64, d_k 128, d_v 128] on chip; D [N 8K, H 64, d_v 128] on chip; Q Kᵀ [N 8K, H 64, 64 64] on chip \| Output: o [N 8K, H 64, d_v 128] BF16 | 25.8 GFLOP | 128 MiB | 192 FLOP/B |
| Gated norm | per element | o ÷ rms × γ ⊙ sigmoid(g) | Rescales each head's output, then lets the output gate g scale every channel. This gate is separate from the decay α. | Input: o [N 8K, H 64, d_v 128] BF16; g [N 8K, H·d 8,192] BF16 \| Weight: γ [d_v 128] BF16 \| Output: o [N 8K, H·d 8,192] BF16 | 604 MFLOP | 384 MiB | 1.50 FLOP/B |
| Output projection | GEMM · weights | concat(o) · W_o | Joins the 64 heads into 8,192 numbers and maps them back to 4,096. | Input: o [N 8K, H·d 8,192] BF16 \| Weight: W_o [H·d 8,192, D 4,096] BF16 \| Output: out [N 8K, D 4,096] BF16 | 550 GFLOP | 256 MiB | 2,048 FLOP/B |
| Write conv | state write | overwrite in place | Replaces all 3 conv columns with the chunk's last 3 inputs; the prefix's 3 are no longer needed. The conv kernel does this itself, after the convolution. | Input: q, k, v [3 3, 3·H·d 24,576] BF16 \| Output: conv [3 tokens 3, 3·H·d 24,576] BF16 | 0 | 288 KiB | 0.00 FLOP/B |
| Write S | state write | overwrite in place | Writes the S left after the chunk's last block over the old one. vLLM runs this as a small kernel of its own (scatter_states) after the chunked kernel. | Input: S [H 64, d_k 128, d_v 128] on chip \| Output: S [H 64, d_k 128, d_v 128] FP32 | 0 | 4.00 MiB | 0.00 FLOP/B |

*Decode: data flow*

- Main row: x (B × 4,096) → Projection → Short conv → Delta-rule update and read-out → Gated norm → Output projection → out (to the MoE FFN)
- Parallel branch: from Projection (f_a, g_a) → Gates → joins Per-block preparation (f, b), Delta-rule update and read-out (f), Gated norm (g)
- b goes from Projection straight to Delta-rule update and read-out
- Write: Short conv (inputs) → Write conv → State, fixed in size
- Write: Delta-rule update and read-out (S) → Write S → State, fixed in size
- Short conv reads conv from the cache: reads last 3
- Delta-rule update and read-out reads S from the cache: reads S

*Decode at Batch B = 32, Context L = 128K: every operation*

| Operation | Kind | Formula | What it does | Operands [axis size] dtype | FLOPs | Bytes | Intensity |
|---|---|---|---|---|---|---|---|
| Projection | GEMM · weights | x · [W_q, W_k, W_v, W_b, W_fa, W_ga] | Turns each token's hidden state into a query, a key and a value for each of the 64 heads, plus the inputs of the gates: one write-strength logit b per head and two 128-number factors f_a, g_a. | Input: x [B 32, D 4,096] BF16 \| Weight: W_in [D 4,096, 3·H·d+H+2r 24,896] BF16 \| Output: q, k, v [B 32, 3·H·d 24,576] BF16; b [B 32, H 64] BF16; f_a, g_a [B 32, 2r 256] BF16 | 6.53 GFLOP | 196 MiB | 31.7 FLOP/B |
| Short conv | per element | Σ over the last 4 tokens, per channel; SiLU | Mixes each channel of q, k and v with the same channel of the 3 tokens before it, which it takes from the conv state. | Input: q, k, v [B 32, 3·H·d 24,576] BF16 \| Weight: w_conv [3·H·d 24,576, 4 taps 4] FP32 \| State: conv [B 32, 3 tokens 3, 3·H·d 24,576] BF16 \| Output: q, k, v [B 32, 3·H·d 24,576] BF16 | 9.44 MFLOP | 7.88 MiB | 1.14 FLOP/B |
| Gates | GEMM · weights | f = f_a · W_fb,  g = g_a · W_gb | Expands f_a into f, one number per key channel of each head, and g_a into the output gate g, which the gated norm applies after the core. f is not the decay yet: the core kernel turns it into α, between 0 and 1. b skips this step and goes straight to the core kernel, which turns it into β = sigmoid(b) itself. | Input: f_a, g_a [B 32, 2r 256] BF16 \| Weight: W_fb [r 128, H·d 8,192] BF16; W_gb [r 128, H·d 8,192] BF16 \| Output: f [B 32, H 64, d_k 128] BF16; g [B 32, H·d 8,192] BF16 | 134 MFLOP | 5.02 MiB | 25.5 FLOP/B |
| Delta-rule update and read-out | GEMM · state | α = decay(f), β = sigmoid(b);  S ← Diag(α) S;  S ← S + β k (v − Sᵀk)ᵀ;  o = Sᵀ q | Turns f into the decay α and b into β. Then it decays each row of S by its α, reads what S stores for the key, Sᵀk, and writes the correction β(v − Sᵀk), so a key written again replaces its old value. Last it reads the output with q. One kernel, the same cost at any L. | Input: q, k, v [B 32, 3·H 192, d 128] BF16; f [B 32, H 64, d_k 128] BF16; b [B 32, H 64] BF16 \| Weight: A, bias [H·d 8,192] FP32 \| State: S [B 32, H 64, d_k 128, d_v 128] FP32 \| Output: o [B 32, H 64, d_v 128] BF16; S [B 32, H 64, d_k 128, d_v 128] on chip | 201 MFLOP | 131 MiB | 1.47 FLOP/B |
| Gated norm | per element | o ÷ rms × γ ⊙ sigmoid(g) | Rescales each head's output, then lets the output gate g scale every channel. This gate is separate from the decay α. | Input: o [B 32, H 64, d_v 128] BF16; g [B 32, H·d 8,192] BF16 \| Weight: γ [d_v 128] BF16 \| Output: o [B 32, H·d 8,192] BF16 | 2.36 MFLOP | 1.50 MiB | 1.50 FLOP/B |
| Output projection | GEMM · weights | concat(o) · W_o | Joins the 64 heads into 8,192 numbers and maps them back to 4,096. | Input: o [B 32, H·d 8,192] BF16 \| Weight: W_o [H·d 8,192, D 4,096] BF16 \| Output: out [B 32, D 4,096] BF16 | 2.15 GFLOP | 64.8 MiB | 31.6 FLOP/B |
| Write conv | state write | overwrite in place | Shifts the conv window by one: this token's inputs enter on the right, the oldest leave. The conv kernel does this itself. | Input: q, k, v [B 32, 3·H·d 24,576] on chip \| Output: conv [B 32, 3 tokens 3, 3·H·d 24,576] BF16 | 0 | 4.50 MiB | 0.00 FLOP/B |
| Write S | state write | overwrite in place | Writes the updated S over the old one. The recurrent kernel does this itself, at its end. | Input: S [B 32, H 64, d_k 128, d_v 128] on chip \| Output: S [B 32, H 64, d_k 128, d_v 128] FP32 | 0 | 128 MiB | 0.00 FLOP/B |

*Axes*

| Symbol | Meaning |
|---|---|
| B | requests |
| N | tokens in the chunk |
| N/64 | 64-token blocks in the chunk |
| 64 | tokens in one block |
| H | heads |
| d | head dim |
| d_k | key channels: one row of S each |
| d_v | value channels: one column of S each |
| r | rank of the gate factors |
| D | hidden size |

*Cache*

- Prefill: State, fixed in size; conv, 144 KiB: the prefix's last 3 pad the chunk; its last 3 replace them; S: 64 × 128 × 128 FP32 = 4 MiB; read once, written once
- Decode: State, fixed in size; conv: last 3 inputs, 3 × 24,576 BF16 = 144 KiB; S: 64 heads × 128 × 128, FP32 = 4 MiB
- 4.1 MiB per layer, whatever L is; 141 MiB per request over 34 KDA layers
