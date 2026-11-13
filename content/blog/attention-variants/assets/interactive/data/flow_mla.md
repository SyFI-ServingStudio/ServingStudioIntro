# Gated MLA in one layer of Kimi K3

All the data behind this figure in "Beyond GQA: Characterizing Modern Attention Variants with ServingStudio".

*Prefill: data flow*

- Main row: x (N × 7,168) → Down-projection → RMSNorm → Query up-projection → Scores → Causal mask → Softmax → Weighted sum → Output gate → Output projection → out (to the MoE FFN)
- Parallel branch: from x (x) → Gate projection → joins Output gate (g)
- Parallel branch: from Query up-projection → Expand K, V → joins Scores (K)
- V goes from Expand K, V straight to Weighted sum
- Write: RMSNorm (c_kv) and Down-projection (k_s, not normalized) → Write c_kv, k_s → Latent cache, kept for the whole request
- Expand K, V reads c_kv, k_s from the cache: reads c_kv, k_s: L earlier + N in chunk

*Prefill at Chunk N = 8,192, Prefix L = 128K: every operation*

| Operation | Kind | Formula | What it does | Operands [axis size] dtype | FLOPs | Bytes | Intensity |
|---|---|---|---|---|---|---|---|
| Gate projection | GEMM · weights | g = x · W_G | Projects each token's hidden state into 12,288 gate values g, one for each number of the attention output. | Input: x [N 8K, D 7,168] BF16 \| Weight: W_G [D 7,168, H·d_v 12,288] BF16 \| Output: g [N 8K, H·d_v 12,288] BF16 | 1.44 TFLOP | 472 MiB | 2,916 FLOP/B |
| Down-projection | GEMM · weights | x · [W_DQ, W_DKV, W_KS] | Compresses each token's hidden state into c_q for the query and the latent c_kv for the key and the value. It also gives k_s, a 64-number key part that all heads share. | Input: x [N 8K, D 7,168] BF16 \| Weight: W_DQ [D 7,168, r_q 1,536] BF16; W_DKV [D 7,168, r_kv 512] BF16; W_KS [D 7,168, d_s 64] BF16 \| Output: c_q [N 8K, r_q 1,536] BF16; c_kv [N 8K, r_kv 512] BF16; k_s [N 8K, d_s 64] BF16 | 248 GFLOP | 174 MiB | 1,360 FLOP/B |
| RMSNorm | per element | c_q, c_kv ÷ rms × γ | Rescales c_q and c_kv, each with its own learned weights. k_s is not normalized. | Input: c_q [N 8K, r_q 1,536] BF16; c_kv [N 8K, r_kv 512] BF16 \| Weight: γ_q [r_q 1,536] BF16; γ_kv [r_kv 512] BF16 \| Output: c_q [N 8K, r_q 1,536] BF16; c_kv [N 8K, r_kv 512] BF16 | 67.1 MFLOP | 64.0 MiB | 1.00 FLOP/B |
| Query up-projection | GEMM · weights | c_q · W_UQ | Expands c_q into a 192-number query for each of the 96 heads: q_u (128) meets the key rebuilt from c_kv, and q_s (64) meets k_s. Kimi K3 rotates neither part by position. | Input: c_q [N 8K, r_q 1,536] BF16 \| Weight: W_UQ [r_q 1,536, H·(d_u+d_s) 18,432] BF16 \| Output: q_u [N 8K, H 96, d_u 128] BF16; q_s [N 8K, H 96, d_s 64] BF16 | 464 GFLOP | 366 MiB | 1,209 FLOP/B |
| Expand K, V | GEMM · weights | K = [c_kv · W_UK, k_s],  V = c_kv · W_UV | Rebuilds every head's key and value from the latent, for the L earlier tokens and the N tokens in the chunk. Each key ends with the shared k_s. The engine expands the earlier tokens a slice at a time, and drops each slice after use. | Weight: W_UK [r_kv 512, H·d_u 12,288] BF16; W_UV [r_kv 512, H·d_v 12,288] BF16 \| Cache: c_kv, k_s [L+N 136K, r_kv+d_s 576] FP8 \| Output: K [L+N 136K, H 96, d_u+d_s 192] BF16; V [L+N 136K, H 96, d_v 128] BF16 | 3.50 TFLOP | 8.07 GiB | 405 FLOP/B |
| Scores | GEMM · cache | Q · Kᵀ ÷ √192 | Compares every query in the chunk with the expanded keys of the L earlier tokens and of the chunk's own tokens, head by head. | Input: q_u, q_s [N 8K, H 96, d_u+d_s 192] BF16; K [L+N 136K, H 96, d_u+d_s 192] BF16 \| Output: S [N 8K, H 96, L+N 136K] on chip | 40.8 TFLOP | 5.06 GiB | 7,509 FLOP/B |
| Causal mask | per element | score = −∞ where key > query | Hides the tokens that come later in the chunk, so each token sees only itself and earlier ones. | Input: S [N 8K, H 96, L+N 136K] on chip \| Output: S [N 8K, H 96, L+N 136K] on chip | 0 | 0, on chip | — |
| Softmax | per element | exp, ÷ row sum | Turns each row of scores into weights that add up to 1. | Input: S [N 8K, H 96, L+N 136K] on chip \| Output: P [N 8K, H 96, L+N 136K] on chip | 532 GFLOP | 0, on chip | — |
| Weighted sum | GEMM · cache | weights · V | Adds up the expanded values of the tokens each query may see, each times its weight. The engine merges the partial sums of the earlier-token slices and of the chunk. | Input: P [N 8K, H 96, L+N 136K] on chip; V [L+N 136K, H 96, d_v 128] BF16 \| Output: o [N 8K, H 96, d_v 128] BF16 | 27.2 TFLOP | 3.38 GiB | 7,509 FLOP/B |
| Output gate | per element | o × sigmoid(g) | Squashes each of the 12,288 gate values g into (0, 1) and scales the matching number of o by it. | Input: o [N 8K, H·d_v 12,288] BF16; g [N 8K, H·d_v 12,288] BF16 \| Output: o [N 8K, H·d_v 12,288] BF16 | 503 MFLOP | 576 MiB | 0.83 FLOP/B |
| Output projection | GEMM · weights | concat(o) · W_O | Joins the 96 gated heads into 12,288 numbers and maps them back to 7,168. | Input: o [N 8K, H·d_v 12,288] BF16 \| Weight: W_O [H·d_v 12,288, D 7,168] BF16 \| Output: out [N 8K, D 7,168] BF16 | 1.44 TFLOP | 472 MiB | 2,916 FLOP/B |
| Write c_kv, k_s | cache write | quantize, append | Stores the chunk's latents and shared keys in FP8, so later chunks and tokens can read them. | Input: c_kv [N 8K, r_kv 512] BF16; k_s [N 8K, d_s 64] BF16 \| Output: c_kv, k_s [N 8K, r_kv+d_s 576] FP8 | 4.72 MFLOP | 13.5 MiB | 0.33 FLOP/B |

*Decode: data flow*

- Main row: x (B × 7,168) → Down-projection → RMSNorm → Query up-projection → Absorb W_UK → Scores → Softmax → Weighted sum → Absorb W_UV → Output gate → Output projection → out (to the MoE FFN)
- Parallel branch: from x (x) → Gate projection → joins Output gate (g)
- q_s goes from Query up-projection straight to Scores
- Write: RMSNorm (c_kv) and Down-projection (k_s, not normalized) → Write c_kv, k_s → Latent cache, kept for the whole request
- Scores reads c_kv, k_s from the cache: reads c_kv, k_s: all L tokens
- Weighted sum reads c_kv, k_s from the cache: same c_kv, as the values

*Decode at Batch B = 32, Context L = 128K: every operation*

| Operation | Kind | Formula | What it does | Operands [axis size] dtype | FLOPs | Bytes | Intensity |
|---|---|---|---|---|---|---|---|
| Gate projection | GEMM · weights | g = x · W_G | Projects each token's hidden state into 12,288 gate values g, one for each number of the attention output. | Input: x [B 32, D 7,168] BF16 \| Weight: W_G [D 7,168, H·d_v 12,288] BF16 \| Output: g [B 32, H·d_v 12,288] BF16 | 5.64 GFLOP | 169 MiB | 31.8 FLOP/B |
| Down-projection | GEMM · weights | x · [W_DQ, W_DKV, W_KS] | Compresses each token's hidden state into c_q for the query and the latent c_kv for the key and the value. It also gives k_s, a 64-number key part that all heads share. | Input: x [B 32, D 7,168] BF16 \| Weight: W_DQ [D 7,168, r_q 1,536] BF16; W_DKV [D 7,168, r_kv 512] BF16; W_KS [D 7,168, d_s 64] BF16 \| Output: c_q [B 32, r_q 1,536] BF16; c_kv [B 32, r_kv 512] BF16; k_s [B 32, d_s 64] BF16 | 969 MFLOP | 29.4 MiB | 31.4 FLOP/B |
| RMSNorm | per element | c_q, c_kv ÷ rms × γ | Rescales c_q and c_kv, each with its own learned weights. k_s is not normalized. | Input: c_q [B 32, r_q 1,536] BF16; c_kv [B 32, r_kv 512] BF16 \| Weight: γ_q [r_q 1,536] BF16; γ_kv [r_kv 512] BF16 \| Output: c_q [B 32, r_q 1,536] BF16; c_kv [B 32, r_kv 512] BF16 | 262 KFLOP | 260 KiB | 0.98 FLOP/B |
| Query up-projection | GEMM · weights | c_q · W_UQ | Expands c_q into a 192-number query for each of the 96 heads: q_u (128) meets the key rebuilt from c_kv, and q_s (64) meets k_s. Kimi K3 rotates neither part by position. | Input: c_q [B 32, r_q 1,536] BF16 \| Weight: W_UQ [r_q 1,536, H·(d_u+d_s) 18,432] BF16 \| Output: q_u [B 32, H 96, d_u 128] BF16; q_s [B 32, H 96, d_s 64] BF16 | 1.81 GFLOP | 55.2 MiB | 31.3 FLOP/B |
| Absorb W_UK | GEMM · weights | q̃ = q_u · W_UKᵀ, head by head | Moves each head's key matrix onto its query. q̃ · c_kv gives the same score as q_u · (c_kv · W_UK), so no key is ever built. | Input: q_u [B 32, H 96, d_u 128] BF16 \| Weight: W_UK [H 96, d_u 128, r_kv 512] BF16 \| Output: q̃ [B 32, H 96, r_kv 512] BF16 | 403 MFLOP | 15.8 MiB | 24.4 FLOP/B |
| Scores | GEMM · cache | [q̃, q_s] · [c_kv, k_s]ᵀ ÷ √192 | Compares each head's q̃ with every earlier token's c_kv, and its q_s with that token's k_s. All 96 heads read the same cache row. | Input: q̃ [B 32, H 96, r_kv 512] BF16; q_s [B 32, H 96, d_s 64] BF16 \| Cache: c_kv, k_s [B 32, L 128K, r_kv+d_s 576] FP8 \| Output: S [B 32, H 96, L 128K] on chip | 464 GFLOP | 2.25 GiB | 192 FLOP/B |
| Softmax | per element | exp, ÷ row sum | Turns each row of scores into weights that add up to 1. | Input: S [B 32, H 96, L 128K] on chip \| Output: P [B 32, H 96, L 128K] on chip | 2.01 GFLOP | 0, on chip | — |
| Weighted sum | GEMM · cache | weights · c_kv | Adds up the earlier tokens' c_kv, each times its weight. Each head gets its own mix of latents. | Input: P [B 32, H 96, L 128K] on chip \| Cache: c_kv [B 32, L 128K, r_kv 512] reused \| Output: o_c [B 32, H 96, r_kv 512] BF16 | 412 GFLOP | 3.00 MiB | 131,072 FLOP/B |
| Absorb W_UV | GEMM · weights | o = o_c · W_UV, head by head | Applies each head's value matrix after the sum: once for the new token, instead of once for every earlier token. | Input: o_c [B 32, H 96, r_kv 512] BF16 \| Weight: W_UV [H 96, r_kv 512, d_v 128] BF16 \| Output: o [B 32, H 96, d_v 128] BF16 | 403 MFLOP | 15.8 MiB | 24.4 FLOP/B |
| Output gate | per element | o × sigmoid(g) | Squashes each of the 12,288 gate values g into (0, 1) and scales the matching number of o by it. | Input: o [B 32, H·d_v 12,288] BF16; g [B 32, H·d_v 12,288] BF16 \| Output: o [B 32, H·d_v 12,288] BF16 | 1.97 MFLOP | 2.25 MiB | 0.83 FLOP/B |
| Output projection | GEMM · weights | concat(o) · W_O | Joins the 96 gated heads into 12,288 numbers and maps them back to 7,168. | Input: o [B 32, H·d_v 12,288] BF16 \| Weight: W_O [H·d_v 12,288, D 7,168] BF16 \| Output: out [B 32, D 7,168] BF16 | 5.64 GFLOP | 169 MiB | 31.8 FLOP/B |
| Write c_kv, k_s | cache write | quantize, append | Stores this token's c_kv and k_s in FP8, so every later token can read them. | Input: c_kv [B 32, r_kv 512] BF16; k_s [B 32, d_s 64] BF16 \| Output: c_kv, k_s [B 32, r_kv+d_s 576] FP8 | 18.4 KFLOP | 54.0 KiB | 0.33 FLOP/B |

*Axes*

| Symbol | Meaning |
|---|---|
| B | requests |
| N | tokens in the chunk |
| L | prefill: earlier tokens in the cache; decode: tokens in each request's cache, the new one included |
| H | heads |
| D | hidden size |
| r_q | query latent |
| r_kv | KV latent c_kv |
| d_u | query part that meets the rebuilt key |
| d_s | shared key part k_s / q_s |
| d_v | value head dim |

*Cache*

- Prefill: Latent cache, kept for the whole request; rows: c_kv, k_s
- Decode: Latent cache, kept for the whole request; rows: c_kv, k_s
- c_kv (512 numbers) and k_s (64), read by all heads: 576 B per token and layer in FP8; 13.5 KiB over 24 layers
