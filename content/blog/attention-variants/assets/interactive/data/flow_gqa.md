# GQA in one layer of Qwen3-Coder-480B-A35B

All the data behind this figure in "Beyond GQA: Characterizing Modern Attention Variants with ServingStudio".

*Prefill: data flow*

- Main row: x (N × 6,144) → Projection → QK-norm → RoPE → Scores → Causal mask → Softmax → Weighted sum → Output projection → out (to the MoE FFN)
- Write: RoPE (k) and Projection (v (no norm, no RoPE)) → Write K, V → KV cache, kept for the whole request
- Scores reads K from the cache: reads K: L earlier + N in chunk
- Weighted sum reads V from the cache: reads V: L earlier + N in chunk

*Prefill at Chunk N = 8,192, Prefix L = 128K: every operation*

| Operation | Kind | Formula | What it does | Operands [axis size] dtype | FLOPs | Bytes | Intensity |
|---|---|---|---|---|---|---|---|
| Projection | GEMM · weights | x · W_q, W_k, W_v | Turns each token's hidden state into a query, a key and a value. | Input: x [N 8K, D 6,144] BF16 \| Weight: W_q [D 6,144, H_q·d 12,288] BF16; W_k [D 6,144, H_kv·d 1,024] BF16; W_v [D 6,144, H_kv·d 1,024] BF16 \| Output: q [N 8K, H_q 96, d 128] BF16; k [N 8K, H_kv 8, d 128] BF16; v [N 8K, H_kv 8, d 128] BF16 | 1.44 TFLOP | 488 MiB | 2,820 FLOP/B |
| QK-norm | per element | q, k ÷ rms × γ | Rescales each head's q and k to the same size, so scores stay in range. | Input: q [N 8K, H_q 96, d 128] BF16; k [N 8K, H_kv 8, d 128] BF16 \| Weight: γ_q [d 128] BF16; γ_k [d 128] BF16 \| Output: q [N 8K, H_q 96, d 128] BF16; k [N 8K, H_kv 8, d 128] BF16 | 436 MFLOP | 416 MiB | 1.00 FLOP/B |
| RoPE | per element | rotate by position | Turns each pair of dimensions by an angle set by the token's position. cos, sin hold 64 values each, looked up by position. | Input: q [N 8K, H_q 96, d 128] BF16; k [N 8K, H_kv 8, d 128] BF16; cos, sin [N 8K, d 128] BF16 \| Output: q [N 8K, H_q 96, d 128] BF16; k [N 8K, H_kv 8, d 128] BF16 | 327 MFLOP | 418 MiB | 0.75 FLOP/B |
| Scores | GEMM · cache | q · Kᵀ ÷ √128 | Compares every query in the chunk with the keys of the L earlier tokens and of the chunk's own tokens. Each KV head serves 12 query heads. | Input: q [N 8K, H_q 96, d 128] BF16 \| Cache: K [L+N 136K, H_kv 8, d 128] FP8 \| Output: S [N 8K, H_q 96, L+N 136K] on chip | 27.2 TFLOP | 328 MiB | 79,123 FLOP/B |
| Causal mask | per element | score = −∞ where key > query | Hides the tokens that come later in the chunk, so each token sees only itself and earlier ones. | Input: S [N 8K, H_q 96, L+N 136K] on chip \| Output: S [N 8K, H_q 96, L+N 136K] on chip | 0 | 0, on chip | — |
| Softmax | per element | exp, ÷ row sum | Turns each row of scores into weights that add up to 1. | Input: S [N 8K, H_q 96, L+N 136K] on chip \| Output: P [N 8K, H_q 96, L+N 136K] on chip | 532 GFLOP | 0, on chip | — |
| Weighted sum | GEMM · cache | P · V | Adds up the values of the tokens it may see, each times its weight. | Input: P [N 8K, H_q 96, L+N 136K] on chip \| Cache: V [L+N 136K, H_kv 8, d 128] FP8 \| Output: o [N 8K, H_q 96, d 128] BF16 | 27.2 TFLOP | 328 MiB | 79,123 FLOP/B |
| Output projection | GEMM · weights | concat(o) · W_o | Joins the 96 heads into 12,288 numbers and maps them back to 6,144. | Input: o [N 8K, H_q·d 12,288] BF16 \| Weight: W_o [H_q·d 12,288, D 6,144] BF16 \| Output: out [N 8K, D 6,144] BF16 | 1.24 TFLOP | 432 MiB | 2,731 FLOP/B |
| Write K, V | cache write | quantize, append | Stores the chunk's keys and values in FP8, so later chunks and tokens can read them. | Input: k [N 8K, H_kv 8, d 128] BF16; v [N 8K, H_kv 8, d 128] BF16 \| Output: K [N 8K, H_kv 8, d 128] FP8; V [N 8K, H_kv 8, d 128] FP8 | 16.8 MFLOP | 48.0 MiB | 0.33 FLOP/B |

*Decode: data flow*

- Main row: x (B × 6,144) → Projection → QK-norm → RoPE → Scores → Softmax → Weighted sum → Output projection → out (to the MoE FFN)
- Write: RoPE (k) and Projection (v (no norm, no RoPE)) → Write K, V → KV cache, kept for the whole request
- Scores reads K from the cache: reads K: all L tokens
- Weighted sum reads V from the cache: reads V: all L tokens

*Decode at Batch B = 32, Context L = 128K: every operation*

| Operation | Kind | Formula | What it does | Operands [axis size] dtype | FLOPs | Bytes | Intensity |
|---|---|---|---|---|---|---|---|
| Projection | GEMM · weights | x · W_q, W_k, W_v | Turns each token's hidden state into a query, a key and a value. | Input: x [B 32, D 6,144] BF16 \| Weight: W_q [D 6,144, H_q·d 12,288] BF16; W_k [D 6,144, H_kv·d 1,024] BF16; W_v [D 6,144, H_kv·d 1,024] BF16 \| Output: q [B 32, H_q 96, d 128] BF16; k [B 32, H_kv 8, d 128] BF16; v [B 32, H_kv 8, d 128] BF16 | 5.64 GFLOP | 169 MiB | 31.8 FLOP/B |
| QK-norm | per element | q, k ÷ rms × γ | Rescales each head's q and k to the same size, so scores stay in range. | Input: q [B 32, H_q 96, d 128] BF16; k [B 32, H_kv 8, d 128] BF16 \| Weight: γ_q [d 128] BF16; γ_k [d 128] BF16 \| Output: q [B 32, H_q 96, d 128] BF16; k [B 32, H_kv 8, d 128] BF16 | 1.70 MFLOP | 1.63 MiB | 1.00 FLOP/B |
| RoPE | per element | rotate by position | Turns each pair of dimensions by an angle set by the token's position. cos, sin hold 64 values each, looked up by position. | Input: q [B 32, H_q 96, d 128] BF16; k [B 32, H_kv 8, d 128] BF16; cos, sin [B 32, d 128] BF16 \| Output: q [B 32, H_q 96, d 128] BF16; k [B 32, H_kv 8, d 128] BF16 | 1.28 MFLOP | 1.63 MiB | 0.75 FLOP/B |
| Scores | GEMM · cache | q · Kᵀ ÷ √128 | Compares q with the key of every token in the cache. Each KV head serves 12 query heads. | Input: q [B 32, H_q 96, d 128] BF16 \| Cache: K [B 32, L 128K, H_kv 8, d 128] FP8 \| Output: S [B 32, H_q 96, L 128K] on chip | 103 GFLOP | 4.00 GiB | 24.0 FLOP/B |
| Softmax | per element | exp, ÷ row sum | Turns each row of scores into weights that add up to 1. | Input: S [B 32, H_q 96, L 128K] on chip \| Output: P [B 32, H_q 96, L 128K] on chip | 2.01 GFLOP | 0, on chip | — |
| Weighted sum | GEMM · cache | P · V | Adds up the values of the tokens it may see, each times its weight. | Input: P [B 32, H_q 96, L 128K] on chip \| Cache: V [B 32, L 128K, H_kv 8, d 128] FP8 \| Output: o [B 32, H_q 96, d 128] BF16 | 103 GFLOP | 4.00 GiB | 24.0 FLOP/B |
| Output projection | GEMM · weights | concat(o) · W_o | Joins the 96 heads into 12,288 numbers and maps them back to 6,144. | Input: o [B 32, H_q·d 12,288] BF16 \| Weight: W_o [H_q·d 12,288, D 6,144] BF16 \| Output: out [B 32, D 6,144] BF16 | 4.83 GFLOP | 145 MiB | 31.8 FLOP/B |
| Write K, V | cache write | quantize, append | Stores this token's k and v in FP8, so every later token can read them. | Input: k [B 32, H_kv 8, d 128] BF16; v [B 32, H_kv 8, d 128] BF16 \| Output: K [B 32, H_kv 8, d 128] FP8; V [B 32, H_kv 8, d 128] FP8 | 65.5 KFLOP | 192 KiB | 0.33 FLOP/B |

*Axes*

| Symbol | Meaning |
|---|---|
| B | requests |
| N | tokens in the chunk |
| L | prefill: earlier tokens in the cache; decode: tokens in each request's cache, the new one included |
| H_q | query heads |
| H_kv | KV heads |
| d | head dim |
| D | hidden size |

*Cache*

- Prefill: KV cache, kept for the whole request; rows: K, V
- Decode: KV cache, kept for the whole request; rows: K, V
- 8 KV heads × 128 × (K + V) = 2 KiB per token and layer in FP8; 124 KiB per token over 62 layers
