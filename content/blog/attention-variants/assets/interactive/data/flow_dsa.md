# Gated DSA in one layer of Kimi K3.1 Pro

All the data behind this figure in "Beyond GQA: Characterizing Modern Attention Variants with ServingStudio".

*Prefill: data flow*

- Main row: x (N × 7,168) → Down-projection → Norms → Query up-projection → Absorb W_UK → Sparse scores → Softmax → Weighted sum → Absorb W_UV → Output gate → Output projection → out (to the MoE FFN)
- Parallel branch: from x (x) → Gate projection → joins Output gate (g)
- Parallel branch: from Query up-projection (q_I) → Index scores → Top-k → joins Sparse scores (2,048 selected)
- w goes from Down-projection straight to Index scores
- q_s goes from Query up-projection straight to Sparse scores
- Write: Norms (k_I, c_kv) and Down-projection (k_s, not normalized) → Write caches → Two caches, kept for the whole request
- Index scores reads k_I from the cache: reads k_I: L earlier + N in chunk
- Sparse scores reads c_kv, k_s from the cache: reads only the 2,048 selected rows
- Weighted sum reads c_kv, k_s from the cache: same rows, as the values

*Prefill at Chunk N = 8,192, Prefix L = 128K: every operation*

| Operation | Kind | Formula | What it does | Operands [axis size] dtype | FLOPs | Bytes | Intensity |
|---|---|---|---|---|---|---|---|
| Gate projection | GEMM · weights | g = x · W_G | Projects each token's hidden state into 12,288 gate values g, one for each number of the attention output. | Input: x [N 8K, D 7,168] BF16 \| Weight: W_G [D 7,168, H·d_v 12,288] BF16 \| Output: g [N 8K, H·d_v 12,288] BF16 | 1.44 TFLOP | 472 MiB | 2,916 FLOP/B |
| Down-projection | GEMM · weights | x · [W_DQ, W_DKV, W_KS, W_IK, W_IW] | Gives the MLA inputs c_q, c_kv and k_s, as in Kimi K3. The same GEMM also gives the indexer's key k_I and one weight per index head, w. | Input: x [N 8K, D 7,168] BF16 \| Weight: W_DQ [D 7,168, r_q 1,536] BF16; W_DKV [D 7,168, r_kv 512] BF16; W_KS [D 7,168, d_s 64] BF16; W_IK [D 7,168, d_I 128] BF16; W_IW [D 7,168, H_I 32] BF16 \| Output: c_q [N 8K, r_q 1,536] BF16; c_kv [N 8K, r_kv 512] BF16; k_s [N 8K, d_s 64] BF16; k_I [N 8K, d_I 128] BF16; w [N 8K, H_I 32] BF16 | 267 GFLOP | 179 MiB | 1,425 FLOP/B |
| Norms | per element | RMSNorm(c_q), RMSNorm(c_kv), LayerNorm(k_I) | Rescales c_q, c_kv and k_I, each with its own learned weights. k_s is not normalized. | Input: c_q [N 8K, r_q 1,536] BF16; c_kv [N 8K, r_kv 512] BF16; k_I [N 8K, d_I 128] BF16 \| Weight: γ_q [r_q 1,536] BF16; γ_kv [r_kv 512] BF16; γ_I, β_I [2·d_I 256] BF16 \| Output: c_q [N 8K, r_q 1,536] BF16; c_kv [N 8K, r_kv 512] BF16; k_I [N 8K, d_I 128] BF16 | 72.4 MFLOP | 68.0 MiB | 1.01 FLOP/B |
| Query up-projection | GEMM · weights | c_q · [W_UQ, W_IQ] | Expands c_q into the attention query, q_u and q_s for each of the 96 heads, and into the indexer's query q_I for each of its 32 heads. Nothing is rotated by position. | Input: c_q [N 8K, r_q 1,536] BF16 \| Weight: W_UQ [r_q 1,536, H·(d_u+d_s) 18,432] BF16; W_IQ [r_q 1,536, H_I·d_I 4,096] BF16 \| Output: q_u [N 8K, H 96, d_u 128] BF16; q_s [N 8K, H 96, d_s 64] BF16; q_I [N 8K, H_I 32, d_I 128] BF16 | 567 GFLOP | 442 MiB | 1,223 FLOP/B |
| Index scores | GEMM · cache | Σₕ wₕ · ReLU(q_I,ₕ · k_I) | Scores every key each query in the chunk may see: the L earlier tokens and the chunk's tokens up to its own position. Each of the 32 index heads scores the key, and w, one weight per index head that the Down-proj. computes from x, sums them into one score. The query is cast to FP8, like the keys. | Input: q_I [N 8K, H_I 32, d_I 128] BF16; w [N 8K, H_I 32] BF16 \| Cache: k_I [L+N 136K, d_I 128] FP8 + scales \| Output: I [N 8K, L+N 136K] on chip | 9.07 TFLOP | 82.0 MiB | 105,457 FLOP/B |
| Top-k | per element | top 2,048 | For each query, keeps the positions of its 2,048 highest-scoring keys. A query that sees 2,048 keys or fewer keeps them all. | Input: I [N 8K, L+N 136K] on chip \| Output: idx [N 8K, k 2K] INT32 | 1.11 GFLOP | 64.0 MiB | 16.5 FLOP/B |
| Absorb W_UK | GEMM · weights | q̃ = q_u · W_UKᵀ, head by head | Moves each head's key matrix onto its query, for every query in the chunk, so attention runs over the cached latents as in decode. | Input: q_u [N 8K, H 96, d_u 128] BF16 \| Weight: W_UK [H 96, d_u 128, r_kv 512] BF16 \| Output: q̃ [N 8K, H 96, r_kv 512] BF16 | 103 GFLOP | 972 MiB | 101 FLOP/B |
| Sparse scores | GEMM · cache | [q̃, q_s] · [c_kv, k_s]ᵀ ÷ √192, selected rows only | Compares each query, head by head, with the c_kv and k_s of its own 2,048 selected tokens only. | Input: q̃ [N 8K, H 96, r_kv 512] BF16; q_s [N 8K, H 96, d_s 64] BF16; idx [N 8K, k 2K] INT32 \| Cache: c_kv [L+N 136K, r_kv 512] FP8 + scales; k_s [L+N 136K, d_s 64] BF16 \| Output: S [N 8K, H 96, k 2K] on chip | 1.86 TFLOP | 1015 MiB | 1,743 FLOP/B |
| Softmax | per element | exp, ÷ row sum | Turns each row of scores into weights that add up to 1. | Input: S [N 8K, H 96, k 2K] on chip \| Output: P [N 8K, H 96, k 2K] on chip | 8.05 GFLOP | 0, on chip | — |
| Weighted sum | GEMM · cache | weights · c_kv, selected rows only | Adds up the selected tokens' c_kv, each times its weight. Each head gets its own mix of latents. | Input: P [N 8K, H 96, k 2K] on chip \| Cache: c_kv [L+N 136K, r_kv 512] reused \| Output: o_c [N 8K, H 96, r_kv 512] BF16 | 1.65 TFLOP | 768 MiB | 2,048 FLOP/B |
| Absorb W_UV | GEMM · weights | o = o_c · W_UV, head by head | Applies each head's value matrix after the sum, once for each query. | Input: o_c [N 8K, H 96, r_kv 512] BF16 \| Weight: W_UV [H 96, r_kv 512, d_v 128] BF16 \| Output: o [N 8K, H 96, d_v 128] BF16 | 103 GFLOP | 972 MiB | 101 FLOP/B |
| Output gate | per element | o × sigmoid(g) | Squashes each of the 12,288 gate values g into (0, 1) and scales the matching number of o by it. | Input: o [N 8K, H·d_v 12,288] BF16; g [N 8K, H·d_v 12,288] BF16 \| Output: o [N 8K, H·d_v 12,288] BF16 | 503 MFLOP | 576 MiB | 0.83 FLOP/B |
| Output projection | GEMM · weights | concat(o) · W_O | Joins the 96 gated heads into 12,288 numbers and maps them back to 7,168. | Input: o [N 8K, H·d_v 12,288] BF16 \| Weight: W_O [H·d_v 12,288, D 7,168] BF16 \| Output: out [N 8K, D 7,168] BF16 | 1.44 TFLOP | 472 MiB | 2,916 FLOP/B |
| Write caches | cache write | quantize, append to both caches | Stores the chunk's k_I in FP8 with one scale per token in the index-key cache, and its c_kv and k_s in the latent cache. | Input: k_I [N 8K, d_I 128] BF16; c_kv [N 8K, r_kv 512] BF16; k_s [N 8K, d_s 64] BF16 \| Output: k_I [N 8K, d_I 128] FP8 + scales; c_kv [N 8K, r_kv 512] FP8 + scales; k_s [N 8K, d_s 64] BF16 | 5.24 MFLOP | 17.2 MiB | 0.29 FLOP/B |

*Decode: data flow*

- Main row: x (B × 7,168) → Down-projection → Norms → Query up-projection → Absorb W_UK → Sparse scores → Softmax → Weighted sum → Absorb W_UV → Output gate → Output projection → out (to the MoE FFN)
- Parallel branch: from x (x) → Gate projection → joins Output gate (g)
- Parallel branch: from Query up-projection (q_I) → Index scores → Top-k → joins Sparse scores (2,048 selected)
- w goes from Down-projection straight to Index scores
- q_s goes from Query up-projection straight to Sparse scores
- Write: Norms (k_I, c_kv) and Down-projection (k_s, not normalized) → Write caches → Two caches, kept for the whole request
- Index scores reads k_I from the cache: reads every k_I: all L tokens
- Sparse scores reads c_kv, k_s from the cache: reads only the 2,048 selected rows
- Weighted sum reads c_kv, k_s from the cache: same rows, as the values

*Decode at Batch B = 32, Context L = 128K: every operation*

| Operation | Kind | Formula | What it does | Operands [axis size] dtype | FLOPs | Bytes | Intensity |
|---|---|---|---|---|---|---|---|
| Gate projection | GEMM · weights | g = x · W_G | Projects each token's hidden state into 12,288 gate values g, one for each number of the attention output. | Input: x [B 32, D 7,168] BF16 \| Weight: W_G [D 7,168, H·d_v 12,288] BF16 \| Output: g [B 32, H·d_v 12,288] BF16 | 5.64 GFLOP | 169 MiB | 31.8 FLOP/B |
| Down-projection | GEMM · weights | x · [W_DQ, W_DKV, W_KS, W_IK, W_IW] | Gives the MLA inputs c_q, c_kv and k_s, as in Kimi K3. The same GEMM also gives the indexer's key k_I and one weight per index head, w. | Input: x [B 32, D 7,168] BF16 \| Weight: W_DQ [D 7,168, r_q 1,536] BF16; W_DKV [D 7,168, r_kv 512] BF16; W_KS [D 7,168, d_s 64] BF16; W_IK [D 7,168, d_I 128] BF16; W_IW [D 7,168, H_I 32] BF16 \| Output: c_q [B 32, r_q 1,536] BF16; c_kv [B 32, r_kv 512] BF16; k_s [B 32, d_s 64] BF16; k_I [B 32, d_I 128] BF16; w [B 32, H_I 32] BF16 | 1.04 GFLOP | 31.6 MiB | 31.4 FLOP/B |
| Norms | per element | RMSNorm(c_q), RMSNorm(c_kv), LayerNorm(k_I) | Rescales c_q, c_kv and k_I, each with its own learned weights. k_s is not normalized. | Input: c_q [B 32, r_q 1,536] BF16; c_kv [B 32, r_kv 512] BF16; k_I [B 32, d_I 128] BF16 \| Weight: γ_q [r_q 1,536] BF16; γ_kv [r_kv 512] BF16; γ_I, β_I [2·d_I 256] BF16 \| Output: c_q [B 32, r_q 1,536] BF16; c_kv [B 32, r_kv 512] BF16; k_I [B 32, d_I 128] BF16 | 283 KFLOP | 277 KiB | 1.00 FLOP/B |
| Query up-projection | GEMM · weights | c_q · [W_UQ, W_IQ] | Expands c_q into the attention query, q_u and q_s for each of the 96 heads, and into the indexer's query q_I for each of its 32 heads. Nothing is rotated by position. | Input: c_q [B 32, r_q 1,536] BF16 \| Weight: W_UQ [r_q 1,536, H·(d_u+d_s) 18,432] BF16; W_IQ [r_q 1,536, H_I·d_I 4,096] BF16 \| Output: q_u [B 32, H 96, d_u 128] BF16; q_s [B 32, H 96, d_s 64] BF16; q_I [B 32, H_I 32, d_I 128] BF16 | 2.21 GFLOP | 67.5 MiB | 31.3 FLOP/B |
| Index scores | GEMM · cache | Σₕ wₕ · ReLU(q_I,ₕ · k_I) | Scores every earlier token with the 32 small index heads in FP8. w, one weight per index head that the Down-proj. computes from x, sums the heads' scores into one number per token. | Input: q_I [B 32, H_I 32, d_I 128] BF16; w [B 32, H_I 32] BF16 \| Cache: k_I [B 32, L 128K, d_I 128] FP8 + scales \| Output: I [B 32, L 128K] on chip | 34.4 GFLOP | 528 MiB | 62.0 FLOP/B |
| Top-k | per element | top 2,048 | Keeps the positions of the 2,048 highest-scoring earlier tokens. With 2,048 or fewer earlier tokens, it keeps them all. | Input: I [B 32, L 128K] on chip \| Output: idx [B 32, k 2K] INT32 | 4.19 MFLOP | 256 KiB | 16.0 FLOP/B |
| Absorb W_UK | GEMM · weights | q̃ = q_u · W_UKᵀ, head by head | Moves each head's key matrix onto its query. q̃ · c_kv gives the same score as q_u · (c_kv · W_UK), so no key is ever built. | Input: q_u [B 32, H 96, d_u 128] BF16 \| Weight: W_UK [H 96, d_u 128, r_kv 512] BF16 \| Output: q̃ [B 32, H 96, r_kv 512] BF16 | 403 MFLOP | 15.8 MiB | 24.4 FLOP/B |
| Sparse scores | GEMM · cache | [q̃, q_s] · [c_kv, k_s]ᵀ ÷ √192, selected rows only | Compares each head's q̃ and q_s with the c_kv and k_s of the 2,048 selected tokens only. All 96 heads read the same rows. | Input: q̃ [B 32, H 96, r_kv 512] BF16; q_s [B 32, H 96, d_s 64] BF16; idx [B 32, k 2K] INT32 \| Cache: c_kv [B 32, k 2K, r_kv 512] FP8 + scales; k_s [B 32, k 2K, d_s 64] BF16 \| Output: S [B 32, H 96, k 2K] on chip | 7.25 GFLOP | 44.6 MiB | 155 FLOP/B |
| Softmax | per element | exp, ÷ row sum | Turns each row of scores into weights that add up to 1. | Input: S [B 32, H 96, k 2K] on chip \| Output: P [B 32, H 96, k 2K] on chip | 31.5 MFLOP | 0, on chip | — |
| Weighted sum | GEMM · cache | weights · c_kv, selected rows only | Adds up the selected tokens' c_kv, each times its weight. Each head gets its own mix of latents. | Input: P [B 32, H 96, k 2K] on chip \| Cache: c_kv [B 32, k 2K, r_kv 512] reused \| Output: o_c [B 32, H 96, r_kv 512] BF16 | 6.44 GFLOP | 3.00 MiB | 2,048 FLOP/B |
| Absorb W_UV | GEMM · weights | o = o_c · W_UV, head by head | Applies each head's value matrix after the sum, once for each query. | Input: o_c [B 32, H 96, r_kv 512] BF16 \| Weight: W_UV [H 96, r_kv 512, d_v 128] BF16 \| Output: o [B 32, H 96, d_v 128] BF16 | 403 MFLOP | 15.8 MiB | 24.4 FLOP/B |
| Output gate | per element | o × sigmoid(g) | Squashes each of the 12,288 gate values g into (0, 1) and scales the matching number of o by it. | Input: o [B 32, H·d_v 12,288] BF16; g [B 32, H·d_v 12,288] BF16 \| Output: o [B 32, H·d_v 12,288] BF16 | 1.97 MFLOP | 2.25 MiB | 0.83 FLOP/B |
| Output projection | GEMM · weights | concat(o) · W_O | Joins the 96 gated heads into 12,288 numbers and maps them back to 7,168. | Input: o [B 32, H·d_v 12,288] BF16 \| Weight: W_O [H·d_v 12,288, D 7,168] BF16 \| Output: out [B 32, D 7,168] BF16 | 5.64 GFLOP | 169 MiB | 31.8 FLOP/B |
| Write caches | cache write | quantize, append to both caches | Stores this token's k_I in FP8 with one scale in the index-key cache, and its c_kv and k_s in the latent cache. | Input: k_I [B 32, d_I 128] BF16; c_kv [B 32, r_kv 512] BF16; k_s [B 32, d_s 64] BF16 \| Output: k_I [B 32, d_I 128] FP8 + scales; c_kv [B 32, r_kv 512] FP8 + scales; k_s [B 32, d_s 64] BF16 | 20.5 KFLOP | 68.6 KiB | 0.29 FLOP/B |

*Axes*

| Symbol | Meaning |
|---|---|
| B | requests |
| N | tokens in the chunk |
| L | prefill: earlier tokens in the cache; decode: tokens in each request's cache, the new one included |
| k | selected rows: min(L, 2,048) per query |
| H | heads |
| H_I | index heads |
| d_I | index head dim |
| D | hidden size |
| r_q | query latent |
| r_kv | KV latent c_kv |
| d_u | query part that meets the rebuilt key |
| d_s | shared key part k_s / q_s |
| d_v | value head dim |

*Cache*

- Prefill: Two caches, kept for the whole request; rows: k_I, c_kv, k_s
- Decode: Two caches, kept for the whole request; rows: k_I, c_kv, k_s
- k_I: 128 FP8 + 4 B scale = 132 B; c_kv 512 FP8 + 16 B scales + k_s 64 BF16 = 656 B; 788 B per token and layer, 18.5 KiB over 24 layers
