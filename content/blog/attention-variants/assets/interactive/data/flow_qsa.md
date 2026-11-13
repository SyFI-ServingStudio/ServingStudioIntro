# QSA in one layer of Qwen3.8 Flash Next

All the data behind this figure in "Beyond GQA: Characterizing Modern Attention Variants with ServingStudio".

Costs are for one whole layer on one GPU, as in the other figures.

*Prefill: data flow*

- Main row: x (N × 2,560) → Projection → Norms, RoPE → Sparse attention → Output projection → out (to the MoE FFN)
- Parallel branch: from Projection (q_I, k_raw) → Index prep → Index scores → Top-k → joins Sparse attention (2,048 selected)
- gate, v goes from Projection straight to Sparse attention
- Write: Index prep (k̄, k_raw) → Write index → Two caches, kept for the whole request
- Index scores reads k̄ from the cache: reads every k̄: (L+N)/4
- Sparse attention reads K, V from the cache: appends k, v; reads 2,048 selected

*Prefill at Chunk N = 8,192, Prefix L = 128K: every operation*

| Operation | Kind | Formula | What it does | Operands [axis size] dtype | FLOPs | Bytes | Intensity |
|---|---|---|---|---|---|---|---|
| Projection | GEMM · weights | x · [W_q, W_gate, W_k, W_v, W_Iq, W_Ik] | Turns each token's hidden state into a query for each of the 24 heads, an output gate of the same size, and a key and a value for each of the 2 KV heads. The same GEMM gives the indexer 4 small index queries and one raw index key. | Input: x [N 8K, D 2,560] BF16 \| Weight: W_qkv [D 2,560, 2·H_q·d+2·H_kv·d 13,312] BF16; W_Iqk [D 2,560, (H_I+1)·d_I 640] BF16 \| Output: q [N 8K, H_q 24, d 256] BF16; gate [N 8K, H_q·d 6,144] BF16; k, v [N 8K, 2·H_kv 4, d 256] BF16; q_I [N 8K, H_I 4, d_I 128] BF16; k_raw [N 8K, d_I 128] BF16 | 585 GFLOP | 326 MiB | 1,711 FLOP/B |
| Norms, RoPE | per element | q, k per head: ÷ rms × γ, rotate 64 channels | Rescales each head of q and k and turns 64 of its 256 channels by the token's position. The same kernel splits the gate off q. | Input: q [N 8K, H_q 24, d 256] BF16; k [N 8K, H_kv 2, d 256] BF16 \| Weight: γ_q, γ_k [2·d 512] BF16 \| Output: q [N 8K, H_q 24, d 256] BF16; k [N 8K, H_kv 2, d 256] BF16 | 259 MFLOP | 208 MiB | 1.19 FLOP/B |
| Index prep | per element | q_I: ÷ rms × γ, rotate; k̄ = norm(mean of 4 raw keys), rotated | Normalizes and rotates the index queries and casts them to FP8. Averages each complete block of 4 raw keys into one pooled key, then normalizes and rotates it. | Input: q_I [N 8K, H_I 4, d_I 128] BF16; k_raw [N 8K, d_I 128] BF16 \| Weight: γ_Iq, γ_Ik [2·d_I 256] BF16 \| State: ring [4 tokens 4, d_I 128] BF16 \| Output: q_I [N 8K, H_I 4, d_I 128] FP8; k̄ [N/4 2K, d_I 128] on chip; k_raw [≤ 3 3, d_I 128] on chip | 25.6 MFLOP | 14.0 MiB | 1.74 FLOP/B |
| Index scores | GEMM · cache | Σ_h ReLU(q_I,h · k̄_g), every complete block g | Scores every complete block of 4 tokens each query may see against its pooled key. One number per block comes out, not one per head. | Input: q_I [N 8K, H_I 4, d_I 128] FP8 \| Cache: k̄ [(L+N)/4 34K, d_I 128] FP8 \| Output: I [N 8K, (L+N)/4 34K] on chip | 283 GFLOP | 8.25 MiB | 32,768 FLOP/B |
| Top-k | per element | top 512 blocks → 2,048 token positions | For each query, keeps its 512 best blocks and expands them into 2,048 token positions, plus the tokens of its last, incomplete block. | Input: I [N 8K, (L+N)/4 34K] on chip \| Output: idx [N 8K, k 2K] INT32 | 277 MFLOP | 64.0 MiB | 4.12 FLOP/B |
| Sparse attention | GEMM · cache | append k, v;  o = softmax(q · Kᵀ ÷ √256) · V over 2,048 tokens;  o × sigmoid(gate) | Appends the chunk's k and v to the K/V cache. Then each query attends to its own 2,048 selected tokens, and the output gate scales the result. | Input: q [N 8K, H_q 24, d 256] BF16; gate [N 8K, H_q·d 6,144] BF16; k, v [N 8K, 2·H_kv 4, d 256] BF16; idx [N 8K, k 2K] INT32 \| Cache: K [L+N 136K, H_kv 2, d 256] FP8; V [L+N 136K, H_kv 2, d 256] FP8 \| Output: k, v [N 8K, 2·H_kv 4, d 256] FP8; o [N 8K, H_q·d 6,144] BF16 | 413 GFLOP | 512 MiB | 768 FLOP/B |
| Output projection | GEMM · weights | concat(o) · W_o | Joins the 24 gated heads into 6,144 numbers and maps them back to 2,560. | Input: o [N 8K, H_q·d 6,144] BF16 \| Weight: W_o [H_q·d 6,144, D 2,560] BF16 \| Output: out [N 8K, D 2,560] BF16 | 258 GFLOP | 166 MiB | 1,480 FLOP/B |
| Write index | cache write | pooled key in FP8; raw key into the ring | Stores the chunk's pooled keys in FP8 in the index cache, and keeps the raw keys of its last, incomplete block in the ring. It is the end of the Index prep kernel. | Input: k̄ [N/4 2K, d_I 128] on chip; k_raw [≤ 3 3, d_I 128] on chip \| Output: k̄ [N/4 2K, d_I 128] FP8; ring [≤ 3 3, d_I 128] BF16 | 262 KFLOP | 257 KiB | 1.00 FLOP/B |

*Decode: data flow*

- Main row: x (B × 2,560) → Projection → Norms, RoPE → Sparse attention → Output projection → out (to the MoE FFN)
- Parallel branch: from Projection (q_I, k_raw) → Index prep → Index scores → Top-k → joins Sparse attention (2,048 selected)
- gate, v goes from Projection straight to Sparse attention
- Write: Index prep (k̄, k_raw) → Write index → Two caches, kept for the whole request
- Index scores reads k̄ from the cache: reads every k̄: L/4
- Sparse attention reads K, V from the cache: appends k, v; reads 2,048 selected

*Decode at Batch B = 32, Context L = 128K: every operation*

| Operation | Kind | Formula | What it does | Operands [axis size] dtype | FLOPs | Bytes | Intensity |
|---|---|---|---|---|---|---|---|
| Projection | GEMM · weights | x · [W_q, W_gate, W_k, W_v, W_Iq, W_Ik] | Turns each token's hidden state into a query for each of the 24 heads, an output gate of the same size, and a key and a value for each of the 2 KV heads. The same GEMM gives the indexer 4 small index queries and one raw index key. | Input: x [B 32, D 2,560] BF16 \| Weight: W_qkv [D 2,560, 2·H_q·d+2·H_kv·d 13,312] BF16; W_Iqk [D 2,560, (H_I+1)·d_I 640] BF16 \| Output: q [B 32, H_q 24, d 256] BF16; gate [B 32, H_q·d 6,144] BF16; k, v [B 32, 2·H_kv 4, d 256] BF16; q_I [B 32, H_I 4, d_I 128] BF16; k_raw [B 32, d_I 128] BF16 | 2.29 GFLOP | 69.1 MiB | 31.5 FLOP/B |
| Norms, RoPE | per element | q, k per head: ÷ rms × γ, rotate 64 channels | Rescales each head of q and k and turns 64 of its 256 channels by the token's position. The same kernel splits the gate off q. | Input: q [B 32, H_q 24, d 256] BF16; k [B 32, H_kv 2, d 256] BF16 \| Weight: γ_q, γ_k [2·d 512] BF16 \| Output: q [B 32, H_q 24, d 256] BF16; k [B 32, H_kv 2, d 256] BF16 | 1.01 MFLOP | 833 KiB | 1.19 FLOP/B |
| Index prep | per element | q_I: ÷ rms × γ, rotate; k̄ = norm(mean of 4 raw keys), rotated | Normalizes and rotates the index queries and casts them to FP8. Every 4th token completes a block: its 4 raw keys, 3 of them from the ring, are averaged into one pooled key, then normalized and rotated. | Input: q_I [B 32, H_I 4, d_I 128] BF16; k_raw [B 32, d_I 128] BF16 \| Weight: γ_Iq, γ_Ik [2·d_I 256] BF16 \| State: ring [B 32, 4 tokens 4, d_I 128] BF16 \| Output: q_I [B 32, H_I 4, d_I 128] FP8; k̄ [B/4 8, d_I 128] on chip; k_raw [B 32, d_I 128] on chip | 99.8 KFLOP | 88.5 KiB | 1.10 FLOP/B |
| Index scores | GEMM · cache | Σ_h ReLU(q_I,h · k̄_g), every complete block g | Scores every complete block of 4 earlier tokens against its pooled key. One number per block comes out, not one per head. | Input: q_I [B 32, H_I 4, d_I 128] FP8 \| Cache: k̄ [B 32, L/4 32K, d_I 128] FP8 \| Output: I [B 32, L/4 32K] on chip | 1.07 GFLOP | 128 MiB | 8.00 FLOP/B |
| Top-k | per element | top 512 blocks → 2,048 token positions | Keeps the 512 best blocks and expands them into 2,048 token positions, plus the tokens of the last, incomplete block. With 512 blocks or fewer, it keeps them all. | Input: I [B 32, L/4 32K] on chip \| Output: idx [B 32, k 2K] INT32 | 1.05 MFLOP | 256 KiB | 4.00 FLOP/B |
| Sparse attention | GEMM · cache | append k, v;  o = softmax(q · Kᵀ ÷ √256) · V over 2,048 tokens;  o × sigmoid(gate) | Appends this token's k and v to the K/V cache. Then the query attends to the 2,048 selected tokens only, and the output gate scales the result. | Input: q [B 32, H_q 24, d 256] BF16; gate [B 32, H_q·d 6,144] BF16; k, v [B 32, 2·H_kv 4, d 256] BF16; idx [B 32, k 2K] INT32 \| Cache: K [B 32, k 2K, H_kv 2, d 256] FP8; V [B 32, k 2K, H_kv 2, d 256] FP8 \| Output: k, v [B 32, 2·H_kv 4, d 256] FP8; o [B 32, H_q·d 6,144] BF16 | 1.61 GFLOP | 65.5 MiB | 23.5 FLOP/B |
| Output projection | GEMM · weights | concat(o) · W_o | Joins the 24 gated heads into 6,144 numbers and maps them back to 2,560. | Input: o [B 32, H_q·d 6,144] BF16 \| Weight: W_o [H_q·d 6,144, D 2,560] BF16 \| Output: out [B 32, D 2,560] BF16 | 1.01 GFLOP | 30.5 MiB | 31.4 FLOP/B |
| Write index | cache write | pooled key in FP8; raw key into the ring | Puts this token's raw key into the ring. Every 4th token, it stores the new pooled key in FP8 in the index cache. It is the end of the Index prep kernel. | Input: k̄ [B/4 8, d_I 128] on chip; k_raw [B 32, d_I 128] on chip \| Output: k̄ [B/4 8, d_I 128] FP8; ring [B 32, d_I 128] BF16 | 1.02 KFLOP | 9.00 KiB | 0.11 FLOP/B |

*Axes*

| Symbol | Meaning |
|---|---|
| B | requests |
| N | tokens in the chunk |
| L | prefill: earlier tokens in the cache; decode: tokens in each request's cache, the new one included |
| k | selected tokens: 512 blocks of 4, min(L, 2,048) |
| H_q | query heads |
| H_kv | KV heads |
| H_I | index query heads |
| d | head dim |
| d_I | index head dim |
| D | hidden size |

*Cache*

- Prefill: Two caches, kept for the whole request; rows: k̄, K, V
- Decode: Two caches, kept for the whole request; rows: k̄, K, V
- k̄: one 128 B key (FP8) per 4 tokens; K, V: 2 heads × 256 × 2 FP8 = 1,024 B per token and layer. Plus a ring of the last raw keys, 1 KiB per request and layer. 12 QSA layers.
