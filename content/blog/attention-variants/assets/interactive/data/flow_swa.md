# Sliding window in one window layer of MiMo V2.6 Flash

All the data behind this figure in "Beyond GQA: Characterizing Modern Attention Variants with ServingStudio".

Costs are for one whole layer on one GPU, as in the other figures.

*Prefill: data flow*

- Main row: x (N × 4,096) → Projection → RoPE → Scores → Window mask → Softmax → Weighted sum → Output projection → out (to the MoE FFN)
- Write: RoPE (k, v) → Write K, V → Window cache: only the last 128 tokens
- Scores reads K from the cache: reads K: last 127 + N in chunk
- Weighted sum reads V from the cache: reads V: last 127 + N in chunk

*Prefill at Chunk N = 8,192, Prefix L = 128K: every operation*

| Operation | Kind | Formula | What it does | Operands [axis size] dtype | FLOPs | Bytes | Intensity |
|---|---|---|---|---|---|---|---|
| Projection | GEMM · weights | x · W_qkv | Turns each token's hidden state into a query, a key and a value. Q and K heads have 192 numbers, V heads 128. | Input: x [N 8K, D 4,096] FP8 + scales \| Weight: W_qkv [D 4,096, H_q·d_qk+H_kv·(d_qk+d_v) 14,848] FP8 + scales \| Output: q [N 8K, H_q 64, d_qk 192] BF16; k [N 8K, H_kv 8, d_qk 192] BF16; v [N 8K, H_kv 8, d_v 128] BF16 | 996 GFLOP | 322 MiB | 2,949 FLOP/B |
| RoPE | per element | rotate 64 of 192 channels | Turns the first 64 channels of each q and k head by an angle set by the token's position. The same kernel scales v by 0.707. | Input: q [N 8K, H_q 64, d_qk 192] BF16; k [N 8K, H_kv 8, d_qk 192] BF16; v [N 8K, H_kv 8, d_v 128] BF16; cos, sin [N 8K, d_rot 64] BF16 \| Output: q [N 8K, H_q 64, d_qk 192] BF16; k [N 8K, H_kv 8, d_qk 192] BF16; v [N 8K, H_kv 8, d_v 128] BF16 | 122 MFLOP | 465 MiB | 0.25 FLOP/B |
| Scores | GEMM · cache | q · Kᵀ ÷ √192 | Compares every query in the chunk with the keys of the 127 tokens before it, at most. Each KV head serves 8 query heads. | Input: q [N 8K, H_q 64, d_qk 192] BF16 \| Cache: K [W−1+N 8,319, H_kv 8, d_qk 192] FP8 \| Output: S [N 8K, H_q 64, W 128] on chip | 25.8 GFLOP | 204 MiB | 120 FLOP/B |
| Window mask | per element | score = −∞ where key > query or key ≤ query − 128 | Hides the later tokens of the chunk and every token more than 127 positions back. | Input: S [N 8K, H_q 64, W 128] on chip \| Output: S [N 8K, H_q 64, W 128] on chip | 0 | 0, on chip | — |
| Softmax | per element | exp, ÷ (row sum + e^sink) | Turns each row of scores into weights. A learned sink per head joins the sum, so the weights may add up to less than 1. | Input: S [N 8K, H_q 64, W 128] on chip \| Weight: sink [H_q 64] FP32 \| Output: P [N 8K, H_q 64, W 128] on chip | 336 MFLOP | 256 B | 1,310,720 FLOP/B |
| Weighted sum | GEMM · cache | weights · V | Adds up the values of the tokens in the window, each times its weight. | Input: P [N 8K, H_q 64, W 128] on chip \| Cache: V [W−1+N 8,319, H_kv 8, d_v 128] FP8 \| Output: o [N 8K, H_q 64, d_v 128] BF16 | 17.2 GFLOP | 136 MiB | 120 FLOP/B |
| Output projection | GEMM · weights | concat(o) · W_o | Joins the 64 heads into 8,192 numbers and maps them back to 4,096. | Input: o [N 8K, H_q·d_v 8,192] BF16 \| Weight: W_o [H_q·d_v 8,192, D 4,096] BF16 \| Output: out [N 8K, D 4,096] BF16 | 550 GFLOP | 256 MiB | 2,048 FLOP/B |
| Write K, V | cache write | write into a 128-token window | Stores the chunk's keys and values. Once the chunk is done, only its last 128 tokens stay; the older slots are freed. | Input: k [N 8K, H_kv 8, d_qk 192] BF16; v [N 8K, H_kv 8, d_v 128] BF16 \| Output: K [N 8K, H_kv 8, d_qk 192] FP8; V [N 8K, H_kv 8, d_v 128] FP8 | 21.0 MFLOP | 60.0 MiB | 0.33 FLOP/B |

*Decode: data flow*

- Main row: x (B × 4,096) → Projection → RoPE → Scores → Softmax → Weighted sum → Output projection → out (to the MoE FFN)
- Write: RoPE (k, v) → Write K, V → Window cache: only the last 128 tokens
- Scores reads K from the cache: reads K: last 127 + this token
- Weighted sum reads V from the cache: reads V: last 127 + this token

*Decode at Batch B = 32, Context L = 128K: every operation*

| Operation | Kind | Formula | What it does | Operands [axis size] dtype | FLOPs | Bytes | Intensity |
|---|---|---|---|---|---|---|---|
| Projection | GEMM · weights | x · W_qkv | Turns each token's hidden state into a query, a key and a value. Q and K heads have 192 numbers, V heads 128. | Input: x [B 32, D 4,096] FP8 + scales \| Weight: W_qkv [D 4,096, H_q·d_qk+H_kv·(d_qk+d_v) 14,848] FP8 + scales \| Output: q [B 32, H_q 64, d_qk 192] BF16; k [B 32, H_kv 8, d_qk 192] BF16; v [B 32, H_kv 8, d_v 128] BF16 | 3.89 GFLOP | 59.0 MiB | 62.9 FLOP/B |
| RoPE | per element | rotate 64 of 192 channels | Turns the first 64 channels of each q and k head by an angle set by the token's position. The same kernel scales v by 0.707. | Input: q [B 32, H_q 64, d_qk 192] BF16; k [B 32, H_kv 8, d_qk 192] BF16; v [B 32, H_kv 8, d_v 128] BF16; cos, sin [B 32, d_rot 64] BF16 \| Output: q [B 32, H_q 64, d_qk 192] BF16; k [B 32, H_kv 8, d_qk 192] BF16; v [B 32, H_kv 8, d_v 128] BF16 | 475 KFLOP | 1.82 MiB | 0.25 FLOP/B |
| Scores | GEMM · cache | q · Kᵀ ÷ √192 | Compares q with the keys of the last 127 tokens and its own, never more. Each KV head serves 8 query heads. | Input: q [B 32, H_q 64, d_qk 192] BF16 \| Cache: K [B 32, W 128, H_kv 8, d_qk 192] FP8 \| Output: S [B 32, H_q 64, W 128] on chip | 101 MFLOP | 6.75 MiB | 14.2 FLOP/B |
| Softmax | per element | exp, ÷ (row sum + e^sink) | Turns each row of scores into weights. A learned sink per head joins the sum, so the weights may add up to less than 1. | Input: S [B 32, H_q 64, W 128] on chip \| Weight: sink [H_q 64] FP32 \| Output: P [B 32, H_q 64, W 128] on chip | 1.31 MFLOP | 256 B | 5,120 FLOP/B |
| Weighted sum | GEMM · cache | weights · V | Adds up the values of the tokens in the window, each times its weight. | Input: P [B 32, H_q 64, W 128] on chip \| Cache: V [B 32, W 128, H_kv 8, d_v 128] FP8 \| Output: o [B 32, H_q 64, d_v 128] BF16 | 67.1 MFLOP | 4.50 MiB | 14.2 FLOP/B |
| Output projection | GEMM · weights | concat(o) · W_o | Joins the 64 heads into 8,192 numbers and maps them back to 4,096. | Input: o [B 32, H_q·d_v 8,192] BF16 \| Weight: W_o [H_q·d_v 8,192, D 4,096] BF16 \| Output: out [B 32, D 4,096] BF16 | 2.15 GFLOP | 64.8 MiB | 31.6 FLOP/B |
| Write K, V | cache write | write into a 128-token window | Stores this token's k and v, and drops the token that falls out of the window, 128 positions back. | Input: k [B 32, H_kv 8, d_qk 192] BF16; v [B 32, H_kv 8, d_v 128] BF16 \| Output: K [B 32, H_kv 8, d_qk 192] FP8; V [B 32, H_kv 8, d_v 128] FP8 | 81.9 KFLOP | 240 KiB | 0.33 FLOP/B |

*Axes*

| Symbol | Meaning |
|---|---|
| B | requests |
| N | tokens in the chunk |
| L | prefill: earlier tokens in the context; decode: tokens in each request's context, the new one included |
| W | window: 128 keys a query sees, itself included |
| H_q | query heads |
| H_kv | KV heads |
| d_qk | query and key head dim |
| d_v | value head dim |
| D | hidden size |

*Cache*

- Prefill: Window cache: only the last 128 tokens; rows: K, V
- Decode: Window cache: only the last 128 tokens; rows: K, V
- 8 KV heads × (192 + 128) = 2.5 KiB per token and layer in FP8; 320 KiB per layer at most, 12 MiB over 39 window layers
