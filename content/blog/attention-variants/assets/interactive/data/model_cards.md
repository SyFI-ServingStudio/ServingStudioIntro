# The models and their attention layers

All the data behind this figure in "Beyond GQA: Characterizing Modern Attention Variants with ServingStudio".

*Models*

| Model | Role | Facts | Attention layers |
|---|---|---|---|
| Qwen3-Coder-480B-A35B | Baseline | 480B total; 35B active; 262K context | Grouped-query attention (GQA) ×62 (Full attention) |
| MiMo V2.6 Flash | — | 309B total; 15B active; 1M context | Sliding-window attention (SWA) ×39 (Sliding window); Grouped-query attention (GQA) ×9 (Full attention) |
| Inkling | — | 975B total; 41B active; 1M context | Sliding-window attention (SWA) ×55 (Sliding window); Grouped-query attention (GQA) ×11 (Full attention) |
| GLM-5.3 | — | 753B total; ~40B active; 1M context | DeepSeek Sparse Attention (DSA) ×78 (Sparse attention) |
| MiniMax-M3.1 Flash | — | ~428B total; ~23B active; 1M context | MiniMax Sparse Attention (MSA) ×60 (Sparse attention) |
| DeepSeek V4 Pro | — | 1.6T total; 49B active; 1M context | Compressed Sparse Attention (CSA) ×30 (Sparse attention); Heavily Compressed Attention (HCA) ×31 (Full attention) |
| DeepSeek V4.1 Flash | — | 552B backbone; 8B active in prefill; 16B active in decode; 1M context | Sliding-window attention (SWA) ×40 (Sliding window); Compressed Sparse Attention (CSA2) ×38 (Sparse attention) |
| Kimi K3 | — | 2.8T total; 104B active; 1M context | Kimi Delta Attention (KDA) ×69 (Linear attention); Gated MLA ×24 (Full attention) |
| Kimi K3.1 Pro | — | 2.8T total; 104B active; 1M context | Kimi Delta Attention (KDA) ×69 (Linear attention); Gated DSA ×24 (Sparse attention) |
| GLM-5.3-Flash | — | 320B total; 18B active; 1M context | Kimi Delta Attention (KDA) ×34 (Linear attention); DeepSeek Sparse Attention (DSA) ×11 (Sparse attention) |
| Qwen3.8 Flash Next | — | 125B total; 6B active; 51B n-gram embedding; 262K context, 1M with YaRN | Gated DeltaNet (GDN) ×36 (Linear attention); Qwen Sparse Attention (QSA) ×12 (Sparse attention) |
| Qwen3.8 27B | — | 27B dense; 262K context, 1M with YaRN | Gated DeltaNet (GDN) ×48 (Linear attention); Gated attention (GQA) ×16 (Full attention) |

Storage parts behind the memory and decode-read figures, one copy per request (DP attention: each GPU runs all heads, a request sits on one GPU). Kept bytes = fixed + bytes per token × min(context, window). A decode step reads, per part, (readBytes, or bytes per token × tokens per entry) × the entries it reads, where entries = min(context, window) ÷ tokens per entry, capped at top-k for a sparse layer; a fixed state is read and written back.

*Storage parts per model*

| Model | Part | Kind | Bytes per token | Window | Fixed | Tokens per entry | Top-k read | Read bytes per entry |
|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | GQA K/V, 62 layers | Whole-context cache | 126,976 B | — | — | — | — | — |
| MiMo V2.6 Flash | Global K/V, 9 layers | Whole-context cache | 11,520 B | — | — | — | — | — |
| MiMo V2.6 Flash | 128-token windows, 39 layers | Window | 99,840 B | 128 | — | — | — | — |
| Inkling | Global K/V, 11 layers | Whole-context cache | 22,528 B | — | — | — | — | — |
| Inkling | 512-token windows, 55 layers | Window | 225,280 B | 512 | — | — | — | — |
| Inkling | Short-conv state (BF16), 66 layers | Fixed state | — | — | 8.08 MiB | — | — | — |
| GLM-5.3 | MLA cache, 78 DSA layers | Whole-context cache | 44,928 B | — | — | — | 2,048 | — |
| GLM-5.3 | Indexer keys, 21 indexing layers | Index | 2,772 B | — | — | — | — | — |
| MiniMax-M3.1 Flash | MSA K/V, 60 layers | Whole-context cache | 61,440 B | — | — | — | 2,048 | — |
| MiniMax-M3.1 Flash | Index keys, 60 layers | Index | 7,680 B | — | — | — | — | — |
| DeepSeek V4 Pro | CSA compressed cache, 30 layers (4 tokens an entry) | Whole-context cache | 4,380 B | — | — | 4 | 1,024 | — |
| DeepSeek V4 Pro | HCA compressed cache, 31 layers (128 tokens an entry) | Whole-context cache | 141.438 B | — | — | 128 | — | — |
| DeepSeek V4 Pro | Indexer keys, 30 CSA layers | Index | 990 B | — | — | 4 | — | — |
| DeepSeek V4 Pro | 128-token windows, 61 layers | Window | 35,624 B | 128 | — | — | — | — |
| DeepSeek V4 Pro | Compressor state (FP32) | Fixed state | — | — | 17.8 MiB | — | — | — |
| DeepSeek V4.1 Flash | Compressed cache, layers 2, 8, 14 (2 tokens an entry) | Whole-context cache | 432 B | — | — | 2 | 512 | 5,184 B |
| DeepSeek V4.1 Flash | Compressed cache, layer 20 | Whole-context cache | 288 B | — | — | — | 512 | 5,760 B |
| DeepSeek V4.1 Flash | Indexer keys (FP4), layers 2, 8, 14 | Index | 102 B | — | — | 2 | — | — |
| DeepSeek V4.1 Flash | Indexer keys (FP4), layer 20 | Index | 68 B | — | — | — | — | 340 B |
| DeepSeek V4.1 Flash | 128-token windows, 40 layers | Window | 21,120 B | 128 | — | — | — | — |
| DeepSeek V4.1 Flash | Compressor ring | Fixed state | — | — | 96.0 KiB | — | — | — |
| Kimi K3 | MLA cache, 24 Gated MLA layers | Whole-context cache | 13,824 B | — | — | — | — | — |
| Kimi K3 | KDA state (FP32), 69 layers | Fixed state | — | — | 429 MiB | — | — | — |
| Kimi K3.1 Pro | MLA cache (fp8_ds_mla), 24 Gated DSA layers | Whole-context cache | 15,744 B | — | — | — | 2,048 | — |
| Kimi K3.1 Pro | Indexer keys, 24 layers | Index | 3,168 B | — | — | — | — | — |
| Kimi K3.1 Pro | KDA state (FP32), 69 layers | Fixed state | — | — | 429 MiB | — | — | — |
| GLM-5.3-Flash | MLA cache, 11 DSA layers | Whole-context cache | 5,632 B | — | — | — | 2,048 | — |
| GLM-5.3-Flash | Pooled index (1 key / 4 tokens) | Index | 363 B | — | — | — | — | — |
| GLM-5.3-Flash | KDA state, 34 layers | Fixed state | — | — | 141 MiB | — | — | — |
| GLM-5.3-Flash | Pooled-index tail (last partial block) | Fixed state | — | — | 22.0 KiB | — | — | — |
| Qwen3.8 Flash Next | QSA K/V, 12 layers | Whole-context cache | 12,288 B | — | — | — | 2,048 | — |
| Qwen3.8 Flash Next | Block index (1 key / 4 tokens) | Index | 384 B | — | — | — | — | — |
| Qwen3.8 Flash Next | QSA ring | Fixed state | — | — | 12.0 KiB | — | — | — |
| Qwen3.8 Flash Next | GDN state, 36 layers | Fixed state | — | — | 110 MiB | — | — | — |
| Qwen3.8 Flash Next | PLE conv state | Fixed state | — | — | 180 KiB | — | — | — |
| Qwen3.8 27B | GQA K/V, 16 layers | Whole-context cache | 32,768 B | — | — | — | — | — |
| Qwen3.8 27B | GDN state, 48 layers | Fixed state | — | — | 147 MiB | — | — | — |
