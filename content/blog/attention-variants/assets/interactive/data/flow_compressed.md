# Compressed and shared caches in layer 2 of DeepSeek V4.1 Flash

All the data behind this figure in "Beyond GQA: Characterizing Modern Attention Variants with ServingStudio".

Costs are for one whole layer on one GPU, as in the other figures.

*Prefill: data flow*

- Main row: x (N × 5,120) → Projection → Norms → Query up-projection → q RoPE → Attention over the window and the selected entries → Output projection → out (to the MoE FFN)
- Parallel branch: from x (x) → Side-stream projections → Compress → Index key projection
- Parallel branch: from q RoPE (q_I) → Index scores → Top-k → joins Attention over the window and the selected entries (512 selected)
- w goes from Side-stream projections straight to Index scores
- kv goes from Norms straight to q RoPE
- Write: Index key projection (c, k_c) and q RoPE (kv) → Write caches → Caches of layer 2; layers 3–7 read its compressed rows
- Index scores reads index keys from the cache: reads all index keys
- Attention over the window and the selected entries reads compressed from the cache
- Attention over the window and the selected entries reads window from the cache: reads 512 entries + window per query

*Prefill at Chunk N = 8,192, Prefix L = 128K: every operation*

| Operation | Kind | Formula | What it does | Operands [axis size] dtype | FLOPs | Bytes | Intensity |
|---|---|---|---|---|---|---|---|
| Projection | GEMM · weights | x · [W_qa, W_kv] | Compresses each token's hidden state into c_q for the query and makes one 512-number kv, the key and the value of every head. | Input: x [N 8K, D 5,120] BF16 \| Weight: W_qa [D 5,120, r_q 1,280] FP8 + scales; W_kv [D 5,120, d 512] FP8 + scales \| Output: c_q [N 8K, r_q 1,280] BF16; kv [N 8K, d 512] BF16 | 150 GFLOP | 117 MiB | 1,228 FLOP/B |
| Norms | per element | RMSNorm(c_q), RMSNorm(kv) | Rescales c_q and kv, each with its own learned weights, in one kernel. Then the paths split: c_q to the query and the indexer, kv to the window. | Input: c_q [N 8K, r_q 1,280] BF16; kv [N 8K, d 512] BF16 \| Weight: γ_q [r_q 1,280] BF16; γ_kv [d 512] BF16 \| Output: c_q [N 8K, r_q 1,280] BF16; kv [N 8K, d 512] BF16 | 58.7 MFLOP | 56.0 MiB | 1.00 FLOP/B |
| Query up-projection | GEMM · weights | c_q · [W_qb, W_IQ] | Expands c_q into a 512-number query for each of the 64 heads, and into the indexer's query q_I for each of its 32 heads. vLLM runs the two as separate GEMMs. | Input: c_q [N 8K, r_q 1,280] BF16 \| Weight: W_qb [r_q 1,280, H·d 32,768] FP8 + scales; W_IQ [r_q 1,280, H_I·d_I 4,096] FP8 + scales \| Output: q [N 8K, H 64, d 512] BF16; q_I [N 8K, H_I 32, d_I 128] BF16 | 773 GFLOP | 641 MiB | 1,150 FLOP/B |
| q RoPE | per element | rotate the last 64 channels of each head of q and q_I | Turns the last 64 channels of each head's query by an angle set by the token's position, for the attention query q and the index query q_I. vLLM rotates q in the kernel that writes kv into the window (so kv comes here), or inside the attention kernel; it rotates q_I and casts it to FP4 in one more kernel. q is not normalized again: c_q was normalized before W_qb. | Input: q [N 8K, H 64, d 512] BF16; q_I [N 8K, H_I 32, d_I 128] BF16; cos, sin [N 8K, d_rot 64] BF16 \| Output: q [N 8K, H 64, d 512] BF16; q_I [N 8K, H_I 32, d_I 128] FP4 + scales | 151 MFLOP | 1.08 GiB | 0.13 FLOP/B |
| Side-stream projections | GEMM · weights | v, s = x · [W_ckv, W_gate],  w = x · W_IW | Two small GEMMs over x that vLLM runs on side streams beside the Projection. The compressor's (fused_wkv_wgate) makes a 512-number value v and 512 gate scores s for every token. The indexer's makes one weight per index head, w. | Input: x [N 8K, D 5,120] BF16 \| Weight: W_ckv [D 5,120, d 512] BF16; W_gate [D 5,120, d 512] BF16; W_IW [D 5,120, H_I 32] BF16 \| Output: v, s [N 8K, 2·d 1,024] FP32; w [N 8K, H_I 32] BF16 | 88.6 GFLOP | 123 MiB | 688 FLOP/B |
| Compress | per element | c = RMSNorm(Σ_t softmax(s_t) ⊙ v_t) over 2 tokens | Merges each pair of tokens into one 512-number entry, weighted channel by channel by the softmax of their gate scores, then normalizes it. One kernel (fused_save_compress_norm) also saves the chunk's last unpaired token in a small FP32 buffer. | Input: v, s [N 8K, 2·d 1,024] FP32 \| Weight: γ_c [d 512] BF16 \| State: state [8 rows 8, 2·d 1,024] FP32 \| Output: c [N/m 4K, d 512] BF16; state [8 rows 8, 2·d 1,024] FP32 | 8.39 MFLOP | 36.1 MiB | 0.22 FLOP/B |
| Index key projection | GEMM · weights | k_c = c · W_k | Turns each compressed entry into a 128-number index key for the indexer. c passes on to the write unchanged. | Input: c [N/m 4K, d 512] BF16 \| Weight: W_k [d 512, d_I 128] BF16 \| Output: k_c [N/m 4K, d_I 128] BF16 | 537 MFLOP | 5.13 MiB | 99.9 FLOP/B |
| Index scores | GEMM · cache | Σ_h w_h · ReLU(q_I,h · k_I) | Scores every compressed entry each query may see. Each of the 32 index heads scores the entry, and w sums the heads' scores into one number per entry. | Input: q_I [N 8K, H_I 32, d_I 128] FP4 + scales; w [N 8K, H_I 32] BF16 \| Cache: k_I [(L+N)/m 68K, d_I 128] FP4 + scales \| Output: I [N 8K, (L+N)/m 68K] on chip | 4.54 TFLOP | 22.0 MiB | 196,468 FLOP/B |
| Top-k | per element | top 512 | For each query, keeps the positions of its 512 highest-scoring entries. A query that sees 512 entries or fewer keeps them all. | Input: I [N 8K, (L+N)/m 68K] on chip \| Output: idx [N 8K, k 512] INT32 | 554 MFLOP | 16.0 MiB | 33.0 FLOP/B |
| Attention over the window and the selected entries | GEMM · cache | softmax(q · Kᵀ ÷ √512) · K,  K = 128 window keys + 512 selected entries;  then rotate o back | Each query attends to the last 128 tokens and to its 512 selected compressed entries. Every entry serves as both key and value, for all 64 heads. The values carry the keys' rotation, so the kernel turns the last 64 channels of each head's output back (inverse RoPE) and casts it to FP8. | Input: q [N 8K, H 64, d 512] BF16; idx [N 8K, k 512] INT32 \| Cache: kv_w [W−1+N 8,319, d 512] MXFP8; c [(L+N)/m 68K, d 512] NVFP4 \| Output: o [N 8K, H 64, d 512] FP8 | 687 GFLOP | 807 MiB | 812 FLOP/B |
| Output projection | GEMM · weights | o · W_oa (8 groups) · W_ob | Maps each group of 8 heads to 1,024 numbers, then the 8,192 results back to 5,120. | Input: o [N 8K, H·d 32,768] FP8 \| Weight: W_oa [G 8, H·d/G 4,096, r_o 1,024] FP8 + scales; W_ob [G·r_o 8,192, D 5,120] FP8 + scales \| Output: out [N 8K, D 5,120] BF16 | 1.24 TFLOP | 408 MiB | 2,891 FLOP/B |
| Write caches | cache write | rotate, quantize, append: kv to the window;  every 2 tokens, c and RMSNorm(k_c) | Rotates the chunk's kv and keeps its last 128 tokens in the window. Rotates the chunk's compressed entries, normalizes and rotates their index keys, and appends both to their caches. vLLM uses three kernels: the window insert, rope_quant_insert for c, and indexer_k_norm_rope_store for k_c. | Input: kv [N 128, d 512] BF16; c [N/m 4K, d 512] BF16; k_c [N/m 4K, d_I 128] BF16 \| Output: kv_w [N 128, d 512] MXFP8; c [N/m 4K, d 512] NVFP4; k_I [N/m 4K, d_I 128] FP4 + scales | 6.38 MFLOP | 6.58 MiB | 0.92 FLOP/B |

*Decode: data flow*

- Main row: x (B × 5,120) → Projection → Norms → Query up-projection → q RoPE → Attention over the window and the selected entries → Output projection → out (to the MoE FFN)
- Parallel branch: from x (x) → Side-stream projections → Compress → Index key projection
- Parallel branch: from q RoPE (q_I) → Index scores → Top-k → joins Attention over the window and the selected entries (512 selected)
- w goes from Side-stream projections straight to Index scores
- kv goes from Norms straight to q RoPE
- Write: Index key projection (c, k_c) and q RoPE (kv) → Write caches → Caches of layer 2; layers 3–7 read its compressed rows
- Index scores reads index keys from the cache: reads all index keys
- Attention over the window and the selected entries reads compressed from the cache
- Attention over the window and the selected entries reads window from the cache: reads 512 entries + 128 in window

*Decode at Batch B = 32, Context L = 128K: every operation*

| Operation | Kind | Formula | What it does | Operands [axis size] dtype | FLOPs | Bytes | Intensity |
|---|---|---|---|---|---|---|---|
| Projection | GEMM · weights | x · [W_qa, W_kv] | Compresses each token's hidden state into c_q for the query and makes one 512-number kv, the key and the value of every head. | Input: x [B 32, D 5,120] BF16 \| Weight: W_qa [D 5,120, r_q 1,280] FP8 + scales; W_kv [D 5,120, d 512] FP8 + scales \| Output: c_q [B 32, r_q 1,280] BF16; kv [B 32, d 512] BF16 | 587 MFLOP | 9.18 MiB | 61.0 FLOP/B |
| Norms | per element | RMSNorm(c_q), RMSNorm(kv) | Rescales c_q and kv, each with its own learned weights, in one kernel. Then the paths split: c_q to the query and the indexer, kv to the window. | Input: c_q [B 32, r_q 1,280] BF16; kv [B 32, d 512] BF16 \| Weight: γ_q [r_q 1,280] BF16; γ_kv [d 512] BF16 \| Output: c_q [B 32, r_q 1,280] BF16; kv [B 32, d 512] BF16 | 229 KFLOP | 228 KiB | 0.98 FLOP/B |
| Query up-projection | GEMM · weights | c_q · [W_qb, W_IQ] | Expands c_q into a 512-number query for each of the 64 heads, and into the indexer's query q_I for each of its 32 heads. vLLM runs the two as separate GEMMs. | Input: c_q [B 32, r_q 1,280] BF16 \| Weight: W_qb [r_q 1,280, H·d 32,768] FP8 + scales; W_IQ [r_q 1,280, H_I·d_I 4,096] FP8 + scales \| Output: q [B 32, H 64, d 512] BF16; q_I [B 32, H_I 32, d_I 128] BF16 | 3.02 GFLOP | 47.4 MiB | 60.8 FLOP/B |
| q RoPE | per element | rotate the last 64 channels of each head of q and q_I | Turns the last 64 channels of each head's query by an angle set by the token's position, for the attention query q and the index query q_I. vLLM rotates q in the kernel that writes kv into the window (so kv comes here), or inside the attention kernel; it rotates q_I and casts it to FP4 in one more kernel. q is not normalized again: c_q was normalized before W_qb. | Input: q [B 32, H 64, d 512] BF16; q_I [B 32, H_I 32, d_I 128] BF16; cos, sin [B 32, d_rot 64] BF16 \| Output: q [B 32, H 64, d 512] BF16; q_I [B 32, H_I 32, d_I 128] FP4 + scales | 590 KFLOP | 4.32 MiB | 0.13 FLOP/B |
| Side-stream projections | GEMM · weights | v, s = x · [W_ckv, W_gate],  w = x · W_IW | Two small GEMMs over x that vLLM runs on side streams beside the Projection. The compressor's (fused_wkv_wgate) makes a 512-number value v and 512 gate scores s for every token. The indexer's makes one weight per index head, w. | Input: x [B 32, D 5,120] BF16 \| Weight: W_ckv [D 5,120, d 512] BF16; W_gate [D 5,120, d 512] BF16; W_IW [D 5,120, H_I 32] BF16 \| Output: v, s [B 32, 2·d 1,024] FP32; w [B 32, H_I 32] BF16 | 346 MFLOP | 10.8 MiB | 30.7 FLOP/B |
| Compress | per element | c = RMSNorm(Σ_t softmax(s_t) ⊙ v_t) over 2 tokens | Every second token completes a pair, which merges into one entry, weighted by the gate scores, and is normalized. The first token of a pair waits in a small FP32 buffer. One kernel (fused_save_compress_norm) does all three. | Input: v, s [B 32, 2·d 1,024] FP32 \| Weight: γ_c [d 512] BF16 \| State: state [B 32, 8 rows 8, 2·d 1,024] FP32 \| Output: c [B/m 16, d 512] BF16; state [B 32, 8 rows 8, 2·d 1,024] FP32 | 32.8 KFLOP | 2.14 MiB | 0.01 FLOP/B |
| Index key projection | GEMM · weights | k_c = c · W_k | Turns each compressed entry into a 128-number index key for the indexer. c passes on to the write unchanged. | Input: c [B/m 16, d 512] BF16 \| Weight: W_k [d 512, d_I 128] BF16 \| Output: k_c [B/m 16, d_I 128] BF16 | 2.10 MFLOP | 148 KiB | 13.8 FLOP/B |
| Index scores | GEMM · cache | Σ_h w_h · ReLU(q_I,h · k_I) | Scores every compressed entry of the request. Each of the 32 index heads scores the entry, and w sums the heads' scores into one number per entry. | Input: q_I [B 32, H_I 32, d_I 128] FP4 + scales; w [B 32, H_I 32] BF16 \| Cache: k_I [B 32, L/m 64K, d_I 128] FP4 + scales \| Output: I [B 32, L/m 64K] on chip | 17.2 GFLOP | 136 MiB | 120 FLOP/B |
| Top-k | per element | top 512 | Keeps the positions of the 512 highest-scoring entries. Layers 3–7 have no indexer and reuse this selection. | Input: I [B 32, L/m 64K] on chip \| Output: idx [B 32, k 512] INT32 | 2.10 MFLOP | 64.0 KiB | 32.0 FLOP/B |
| Attention over the window and the selected entries | GEMM · cache | softmax(q · Kᵀ ÷ √512) · K,  K = 128 window keys + 512 selected entries;  then rotate o back | The new token attends to the last 128 tokens, itself included, and to the 512 selected compressed entries. Every entry serves as both key and value, for all 64 heads. The values carry the keys' rotation, so the kernel turns the last 64 channels of each head's output back (inverse RoPE) and casts it to FP8. | Input: q [B 32, H 64, d 512] BF16; idx [B 32, k 512] INT32 \| Cache: kv_w [B 32, W 128, d 512] MXFP8; c [B 32, k 512, d 512] NVFP4 \| Output: o [B 32, H 64, d 512] FP8 | 2.68 GFLOP | 9.63 MiB | 266 FLOP/B |
| Output projection | GEMM · weights | o · W_oa (8 groups) · W_ob | Maps each group of 8 heads to 1,024 numbers, then the 8,192 results back to 5,120. | Input: o [B 32, H·d 32,768] FP8 \| Weight: W_oa [G 8, H·d/G 4,096, r_o 1,024] FP8 + scales; W_ob [G·r_o 8,192, D 5,120] FP8 + scales \| Output: out [B 32, D 5,120] BF16 | 4.83 GFLOP | 73.4 MiB | 62.8 FLOP/B |
| Write caches | cache write | rotate, quantize, append: kv to the window;  every 2 tokens, c and RMSNorm(k_c) | Rotates this token's kv and stores it in the window, dropping the token 128 positions back. Every second token, it also stores one compressed entry and its normalized index key. | Input: kv [B 32, d 512] BF16; c [B/m 16, d 512] BF16; k_c [B/m 16, d_I 128] BF16 \| Output: kv_w [B 32, d 512] MXFP8; c [B/m 16, d 512] NVFP4; k_I [B/m 16, d_I 128] FP4 + scales | 47.1 KFLOP | 74.1 KiB | 0.62 FLOP/B |

*Axes*

| Symbol | Meaning |
|---|---|
| B | requests |
| N | tokens in the chunk |
| L | prefill: earlier tokens in the context; decode: tokens in each request's context, the new one included |
| m | tokens per compressed entry: 2 in layer 2 |
| W | window: 128 tokens, itself included |
| k | selected entries: min(L/m, 512) per query |
| H | heads |
| d | head dim; key = value |
| r_q | query latent |
| H_I | index heads |
| d_I | index head dim |
| D | hidden size |

*Cache*

- Prefill: Caches of layer 2; layers 3–7 read its compressed rows; rows: index keys, compressed, window
- Decode: Caches of layer 2; layers 3–7 read its compressed rows; rows: index keys, compressed, window
- Per 2 tokens: one 288 B entry (NVFP4) and one 68 B index key (FP4), kept for the whole request. Window: 128 × 528 B (MXFP8)
