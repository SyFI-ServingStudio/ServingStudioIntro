# MSA in one layer of MiniMax-M3.1 Flash

All the data behind this figure in "Beyond GQA: Characterizing Modern Attention Variants with ServingStudio".

Costs are for one whole layer on one GPU, as in the other figures.

*Prefill: data flow*

- Main row: x (N × 6,144) → Projection → Norms, RoPE → Sparse attention → Output projection → out (to the MoE FFN)
- Parallel branch: from Norms, RoPE (q_I) → Index scores → Top-k → joins Sparse attention (16 blocks per group)
- Write: Norms, RoPE (k, k_I) and Projection (v (no norm, no RoPE)) → Write caches → Two caches, kept for the whole request
- Index scores reads k_I from the cache: reads k_I: L earlier + N in chunk
- Sparse attention reads K, V from the cache: reads 16 blocks per group

*Prefill at Chunk N = 8,192, Prefix L = 128K: every operation*

| Operation | Kind | Formula | What it does | Operands [axis size] dtype | FLOPs | Bytes | Intensity |
|---|---|---|---|---|---|---|---|
| Projection | GEMM · weights | x · [W_q, W_k, W_v, W_Iq, W_Ik] | Turns each token's hidden state into a query, a key and a value, plus the indexer's query, one head per KV group, and its key, one head shared by all groups. One GEMM makes all five. | Input: x [N 8K, D 6,144] MXFP8 \| Weight: W_qkv,I [D 6,144, (H_q+2H_kv+H_I+1)·d 9,856] MXFP8 \| Output: q [N 8K, H_q 64, d 128] BF16; k, v [N 8K, 2·H_kv 8, d 128] BF16; q_I [N 8K, H_I 4, d_I 128] BF16; k_I [N 8K, d_I 128] BF16 | 992 GFLOP | 263 MiB | 3,597 FLOP/B |
| Norms, RoPE | per element | per head: ÷ rms × γ, rotate 64 channels | Rescales each head of q, k, q_I and k_I and turns 64 of its channels by the token's position. The same kernel writes k, v and k_I into the caches and casts q_I to FP8. | Input: q [N 8K, H_q 64, d 128] BF16; k [N 8K, H_kv 4, d 128] BF16; q_I [N 8K, H_I 4, d_I 128] BF16; k_I [N 8K, d_I 128] BF16 \| Weight: γ ×4 [4·d 512] BF16 \| Output: q [N 8K, H_q 64, d 128] BF16; k [N 8K, H_kv 4, d 128] on chip; q_I [N 8K, H_I 4, d_I 128] FP8; k_I [N 8K, d_I 128] on chip | 421 MFLOP | 278 MiB | 1.44 FLOP/B |
| Index scores | GEMM · cache | s_g(t) = q_I,g · k_I(t); block score = max over its 128 tokens | Scores every key each query may see, once for each KV group's index query, then gives every 128-token block the highest score of any token in it. | Input: q_I [N 8K, H_I 4, d_I 128] FP8 \| Cache: k_I [L+N 136K, d_I 128] FP8 \| Output: block scores [N 8K, H_I 4, (L+N)/128 1,088] on chip | 1.14 TFLOP | 21.0 MiB | 51,694 FLOP/B |
| Top-k | per element | top 16 blocks per group; the local block always | For each query and KV group, keeps the 16 best blocks, the block holding the query itself always among them. | Input: block scores [N 8K, H_I 4, (L+N)/128 1,088] on chip \| Output: idx [N 8K, H_I 4, 16 16] INT32 | 34.6 MFLOP | 2.00 MiB | 16.5 FLOP/B |
| Sparse attention | GEMM · cache | softmax(q · Kᵀ ÷ √128) · V over each group's 16 blocks | Each query head attends to the 2,048 tokens of the 16 blocks its KV group selected for that query. Scores and weights stay inside the kernel. | Input: q [N 8K, H_q 64, d 128] BF16; idx [N 8K, H_I 4, 16 16] INT32 \| Cache: K [L+N 136K, H_kv 4, d 128] FP8; V [L+N 136K, H_kv 4, d 128] FP8 \| Output: o [N 8K, H_q·d 8,192] BF16 | 550 GFLOP | 394 MiB | 1,331 FLOP/B |
| Output projection | GEMM · weights | concat(o) · W_o | Joins the 64 heads into 8,192 numbers and maps them back to 6,144. | Input: o [N 8K, H_q·d 8,192] BF16 \| Weight: W_o [H_q·d 8,192, D 6,144] MXFP8 \| Output: out [N 8K, D 6,144] BF16 | 825 GFLOP | 274 MiB | 2,875 FLOP/B |
| Write caches | cache write | quantize, append k, v and k_I | Stores the chunk's k and v in the K/V cache and its k_I in the index cache, in FP8. It is the end of the Norms, RoPE kernel. | Input: k [N 8K, H_kv 4, d 128] on chip; v [N 8K, H_kv 4, d 128] BF16; k_I [N 8K, d_I 128] on chip \| Output: k, v [N 8K, 2·H_kv 8, d 128] FP8; k_I [N 8K, d_I 128] FP8 | 9.44 MFLOP | 17.0 MiB | 0.53 FLOP/B |

*Decode: data flow*

- Main row: x (B × 6,144) → Projection → Norms, RoPE → Sparse attention → Output projection → out (to the MoE FFN)
- Parallel branch: from Norms, RoPE (q_I) → Index scores → Top-k → joins Sparse attention (16 blocks per group)
- Write: Norms, RoPE (k, k_I) and Projection (v (no norm, no RoPE)) → Write caches → Two caches, kept for the whole request
- Index scores reads k_I from the cache: reads every k_I: all L tokens
- Sparse attention reads K, V from the cache: reads 16 blocks per group

*Decode at Batch B = 32, Context L = 128K: every operation*

| Operation | Kind | Formula | What it does | Operands [axis size] dtype | FLOPs | Bytes | Intensity |
|---|---|---|---|---|---|---|---|
| Projection | GEMM · weights | x · [W_q, W_k, W_v, W_Iq, W_Ik] | Turns each token's hidden state into a query, a key and a value, plus the indexer's query, one head per KV group, and its key, one head shared by all groups. One GEMM makes all five. | Input: x [B 32, D 6,144] MXFP8 \| Weight: W_qkv,I [D 6,144, (H_q+2H_kv+H_I+1)·d 9,856] MXFP8 \| Output: q [B 32, H_q 64, d 128] BF16; k, v [B 32, 2·H_kv 8, d 128] BF16; q_I [B 32, H_I 4, d_I 128] BF16; k_I [B 32, d_I 128] BF16 | 3.88 GFLOP | 60.3 MiB | 61.2 FLOP/B |
| Norms, RoPE | per element | per head: ÷ rms × γ, rotate 64 channels | Rescales each head of q, k, q_I and k_I and turns 64 of its channels by the token's position. The same kernel writes k, v and k_I into the caches and casts q_I to FP8. | Input: q [B 32, H_q 64, d 128] BF16; k [B 32, H_kv 4, d 128] BF16; q_I [B 32, H_I 4, d_I 128] BF16; k_I [B 32, d_I 128] BF16 \| Weight: γ ×4 [4·d 512] BF16 \| Output: q [B 32, H_q 64, d 128] BF16; k [B 32, H_kv 4, d 128] on chip; q_I [B 32, H_I 4, d_I 128] FP8; k_I [B 32, d_I 128] on chip | 1.64 MFLOP | 1.09 MiB | 1.44 FLOP/B |
| Index scores | GEMM · cache | s_g(t) = q_I,g · k_I(t); block score = max over its 128 tokens | Scores every earlier token, once for each KV group's index query, then gives every 128-token block the highest score of any token in it. | Input: q_I [B 32, H_I 4, d_I 128] FP8 \| Cache: k_I [B 32, L 128K, d_I 128] FP8 \| Output: block scores [B 32, H_I 4, L/128 1K] on chip | 4.31 GFLOP | 512 MiB | 8.03 FLOP/B |
| Top-k | per element | top 16 blocks per group; the local block always | For each KV group, keeps the 16 best blocks, the block holding the newest tokens always among them. With 16 blocks or fewer, it keeps them all. | Input: block scores [B 32, H_I 4, L/128 1K] on chip \| Output: idx [B 32, H_I 4, 16 16] INT32 | 131 KFLOP | 8.00 KiB | 16.0 FLOP/B |
| Sparse attention | GEMM · cache | softmax(q · Kᵀ ÷ √128) · V over each group's 16 blocks | Each query head attends to the 2,048 tokens of the 16 blocks its KV group selected. The 4 groups read different blocks. | Input: q [B 32, H_q 64, d 128] BF16; idx [B 32, H_I 4, 16 16] INT32 \| Cache: K [B 32, H_kv 4, k 2K, d 128] FP8; V [B 32, H_kv 4, k 2K, d 128] FP8 \| Output: o [B 32, H_q·d 8,192] BF16 | 2.15 GFLOP | 65.0 MiB | 31.5 FLOP/B |
| Output projection | GEMM · weights | concat(o) · W_o | Joins the 64 heads into 8,192 numbers and maps them back to 6,144. | Input: o [B 32, H_q·d 8,192] BF16 \| Weight: W_o [H_q·d 8,192, D 6,144] MXFP8 \| Output: out [B 32, D 6,144] BF16 | 3.22 GFLOP | 50.4 MiB | 61.0 FLOP/B |
| Write caches | cache write | quantize, append k, v and k_I | Stores this token's k and v in the K/V cache and its k_I in the index cache, in FP8. It is the end of the Norms, RoPE kernel. | Input: k [B 32, H_kv 4, d 128] on chip; v [B 32, H_kv 4, d 128] BF16; k_I [B 32, d_I 128] on chip \| Output: k, v [B 32, 2·H_kv 8, d 128] FP8; k_I [B 32, d_I 128] FP8 | 36.9 KFLOP | 68.0 KiB | 0.53 FLOP/B |

*Axes*

| Symbol | Meaning |
|---|---|
| B | requests |
| N | tokens in the chunk |
| L | prefill: earlier tokens in the cache; decode: tokens in each request's cache, the new one included |
| k | tokens a KV group keeps: its 16 blocks of 128, min(L, 2,048) |
| H_q | query heads |
| H_kv | KV heads, one per group of 16 query heads |
| H_I | index query heads, one per KV group |
| d | head dim |
| d_I | index head dim |
| D | hidden size |

*Cache*

- Prefill: Two caches, kept for the whole request; rows: k_I, K, V
- Decode: Two caches, kept for the whole request; rows: k_I, K, V
- k_I: 128 FP8 = 128 B; K, V: 4 heads × 128 × 2 FP8 = 1,024 B; 1,152 B per token and layer, 67.5 KiB over 60 layers
