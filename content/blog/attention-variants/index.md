# Beyond GQA: Characterizing Modern Attention Variants with ServingStudio

![Eleven attention designs in today's open models, and the best one on each measure against a GQA baseline at 128K tokens](assets/interactive/teaser.png)

*Top: the eleven attention designs in today's open models. Bottom: the best new model on each measure, against the GQA baseline at 128K tokens.*

## Introduction

A few years ago, most open LLMs used the same attention design: grouped-query attention (GQA), in which multiple Query vectors attend to the same Key-Value vector to save memory. The optimization of attention did not stop there. Recent models use much
more diverse attention mechanisms, and these mechanisms drastically change the HBM consumption, the memory loading, and the TFLOPs spent on the attention. 

In this post, we look at eleven recent models: DeepSeek V4.1 Flash, DeepSeek V4 Pro, GLM-5.3-Flash, Qwen3.8 Flash
Next, MiMo V2.6 Flash, Inkling, GLM-5.3, MiniMax-M3.1 Flash, Kimi K3, Kimi K3.1 Pro and Qwen3.8 27B. We compare them with
Qwen3-Coder-480B-A35B, a GQA model of similar size to most of them.

At a 128K-token context, the new models need far less than the baseline:

- **Memory.** A request keeps up to 139× less attention state (DeepSeek V4.1 Flash).
- **Decode.** Each step loads up to 250× fewer bytes, and its attention runs up to 44× faster (DeepSeek V4.1 Flash).
- **Prefill.** A prompt needs up to 183× fewer attention FLOPs (Qwen3.8 Flash Next), and its attention runs up to
  46× faster (DeepSeek V4.1 Flash).

These changes matter beyond the numbers themselves. Much of today's performance analysis of LLM inference still
rests on older GQA models: when a step is bound by attention and when by the FFN, how many requests fit in memory,
and which kernels are worth tuning. The new attention designs change these answers.

<div class="key-message">
<p class="key-message-label">Our main takeaway</p>
<p class="key-message-head">The attention landscape has changed, and inference research needs to catch up.</p>
<p>New attention designs change how much memory a request needs, how many bytes each step moves, and how much it
computes. As a result, the bottlenecks and optimization opportunities of GQA models do not always carry over. Future
inference research should study these new models directly, instead of reusing conclusions drawn from GQA models.</p>
</div>

## Architectures

All models except Qwen3.8 27B are mixture-of-experts (MoE) transformers. Their attention layers differ. The baseline uses GQA in every layer. GLM-5.3 replaces GQA with DSA in every layer, and MiniMax-M3.1
Flash replaces it with MSA. The other nine models mix two or more kinds of attention layers. The table below lists
their size and attention choices.

**Figure data: The models and their attention layers**

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

All of this figure's data (every setting and operation): [data/model_cards.md](assets/interactive/data/model_cards.md)

All analysis below assumes the model is deployed on B200 GPUs with data-parallel (DP) attention: each GPU runs every attention head of a
layer, and each request stays on one GPU. We borrow kernels from vLLM or SGLang and measure them with ServingStudio.

## Attention Variants

We first go through how each attention design works. They fall into two branches. The first keeps an entry for
every earlier token in the cache. On this branch, GQA and MLA make each entry smaller, while sliding windows, DSA, CSA
and HCA cut how much of the cache is kept or read. The second branch, linear attention, 
folds the whole context into a state of fixed size.

### Grouped-query attention (GQA)

**Figure data: GQA in one layer of Qwen3-Coder-480B-A35B**

*Prefill: data flow*

- Main row: x (N × 6,144) → Projection → QK-norm → RoPE → Scores → Causal mask → Softmax → Weighted sum → Output projection → out (to the MoE FFN)
- Write: RoPE (k) and Projection (v (no norm, no RoPE)) → Write K, V → KV cache, kept for the whole request
- Scores reads K from the cache: reads K: L earlier + N in chunk
- Weighted sum reads V from the cache: reads V: L earlier + N in chunk

*Decode: data flow*

- Main row: x (B × 6,144) → Projection → QK-norm → RoPE → Scores → Softmax → Weighted sum → Output projection → out (to the MoE FFN)
- Write: RoPE (k) and Projection (v (no norm, no RoPE)) → Write K, V → KV cache, kept for the whole request
- Scores reads K from the cache: reads K: all L tokens
- Weighted sum reads V from the cache: reads V: all L tokens

*Cache*

- Prefill: KV cache, kept for the whole request; rows: K, V
- Decode: KV cache, kept for the whole request; rows: K, V
- 8 KV heads × 128 × (K + V) = 2 KiB per token and layer in FP8; 124 KiB per token over 62 layers

All of this figure's data (every setting and operation): [data/flow_gqa.md](assets/interactive/data/flow_gqa.md)

In standard multi-head attention, every query head has its own key head and value head. GQA lets a group of query
heads share one key-value head, so each token keeps fewer keys and values. Our baseline, Qwen3-Coder-480B-A35B, has
96 query heads and 8 key-value heads in each layer, so 12 query heads share each key-value head.

The token's hidden state is first projected into queries, keys and values, and
QK-norm and RoPE normalize the queries and keys and encode their positions. The keys and values are stored in the KV
cache, where they stay for the entire lifecycle of the request. To compute its output, a new token compares its
queries with the keys of all earlier tokens, turns these scores into weights with a softmax, and takes the weighted
sum of the earlier tokens' values. The two blue arrows show these two reads from the cache: computing the scores
reads every key, and the weighted sum reads every value.

### Multi-head latent attention (MLA)

**Figure data: Gated MLA in one layer of Kimi K3**

*Prefill: data flow*

- Main row: x (N × 7,168) → Down-projection → RMSNorm → Query up-projection → Scores → Causal mask → Softmax → Weighted sum → Output gate → Output projection → out (to the MoE FFN)
- Parallel branch: from x (x) → Gate projection → joins Output gate (g)
- Parallel branch: from Query up-projection → Expand K, V → joins Scores (K)
- V goes from Expand K, V straight to Weighted sum
- Write: RMSNorm (c_kv) and Down-projection (k_s, not normalized) → Write c_kv, k_s → Latent cache, kept for the whole request
- Expand K, V reads c_kv, k_s from the cache: reads c_kv, k_s: L earlier + N in chunk

*Decode: data flow*

- Main row: x (B × 7,168) → Down-projection → RMSNorm → Query up-projection → Absorb W_UK → Scores → Softmax → Weighted sum → Absorb W_UV → Output gate → Output projection → out (to the MoE FFN)
- Parallel branch: from x (x) → Gate projection → joins Output gate (g)
- q_s goes from Query up-projection straight to Scores
- Write: RMSNorm (c_kv) and Down-projection (k_s, not normalized) → Write c_kv, k_s → Latent cache, kept for the whole request
- Scores reads c_kv, k_s from the cache: reads c_kv, k_s: all L tokens
- Weighted sum reads c_kv, k_s from the cache: same c_kv, as the values

*Cache*

- Prefill: Latent cache, kept for the whole request; rows: c_kv, k_s
- Decode: Latent cache, kept for the whole request; rows: c_kv, k_s
- c_kv (512 numbers) and k_s (64), read by all heads: 576 B per token and layer in FP8; 13.5 KiB over 24 layers

All of this figure's data (every setting and operation): [data/flow_mla.md](assets/interactive/data/flow_mla.md)

MLA makes each entry smaller in a different way. Instead of storing keys and values per head, an MLA layer caches one
compressed vector per token, the latent $c_{kv}$, which all query heads share. Each head can rebuild its own key and value
from $c_{kv}$ with two learned matrices, $W_{UK}$ and $W_{UV}$.

In prefill, the engine rebuilds the keys and values of the earlier tokens from their cached latents with $W_{UK}$ and
$W_{UV}$ (Expand K, V), and then computes attention as in GQA. The rebuild is paid once and shared
by the thousands of tokens in the chunk.

In decode, only one new token arrives at each step, so rebuilding every
earlier key and value just for it would be wasteful. Instead, the engine folds $W_{UK}$ into the new token's query
(Absorb $W_{UK}$) and applies $W_{UV}$ after the weighted sum (Absorb $W_{UV}$). Attention then works on the cached latents
directly, so both blue arrows start from the same cache row.

Absorbing $W_{UK}$ does not work if RoPE rotates the keys, because the rotation depends on each token's position. The
original MLA in DeepSeek V2 therefore caches one more small key part per token, $k_s$, which all heads share, and
applies RoPE only to $k_s$.

Kimi K3, whose layer the figure shows, changes MLA in two small ways. It drops RoPE, so $k_s$ is a plain key that
carries no position. It also adds an output gate. A separate projection of the input
gives a gate $g$, and before the output projection each number of the attention output is scaled by $sigmoid(g)$.

### Sliding-window attention

**Figure data: Sliding window in one window layer of MiMo V2.6 Flash**

Costs are for one whole layer on one GPU, as in the other figures.

*Prefill: data flow*

- Main row: x (N × 4,096) → Projection → RoPE → Scores → Window mask → Softmax → Weighted sum → Output projection → out (to the MoE FFN)
- Write: RoPE (k, v) → Write K, V → Window cache: only the last 128 tokens
- Scores reads K from the cache: reads K: last 127 + N in chunk
- Weighted sum reads V from the cache: reads V: last 127 + N in chunk

*Decode: data flow*

- Main row: x (B × 4,096) → Projection → RoPE → Scores → Softmax → Weighted sum → Output projection → out (to the MoE FFN)
- Write: RoPE (k, v) → Write K, V → Window cache: only the last 128 tokens
- Scores reads K from the cache: reads K: last 127 + this token
- Weighted sum reads V from the cache: reads V: last 127 + this token

*Cache*

- Prefill: Window cache: only the last 128 tokens; rows: K, V
- Decode: Window cache: only the last 128 tokens; rows: K, V
- 8 KV heads × (192 + 128) = 2.5 KiB per token and layer in FP8; 320 KiB per layer at most, 12 MiB over 39 window layers

All of this figure's data (every setting and operation): [data/flow_swa.md](assets/interactive/data/flow_swa.md)

A sliding-window layer attends only to the last $W$ tokens. Older tokens are never read again, so the cache keeps
only $W$ entries and stops growing once the context passes the window. The window can sit on top of either GQA or
MLA. MiMo V2.6 Flash uses GQA-based windows in most of its layers, and DeepSeek V4.1 Flash uses MLA-based windows in
all of its layers. Both use $W$ = 128.

In prefill, each token in the chunk compares its query only with the last $W$ tokens up to itself, so the scores form
a narrow band along the diagonal. In decode, each step adds the new token's entry and drops the oldest one.

Inkling also mixes window layers, with $W$ = 512, and GQA layers. Instead of RoPE, it encodes positions with a
learned bias that it adds to each score, based on how far apart the two tokens are. Its attention kernel must
therefore compute this bias as well.

### DeepSeek Sparse Attention (DSA)

**Figure data: Gated DSA in one layer of Kimi K3.1 Pro**

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

*Cache*

- Prefill: Two caches, kept for the whole request; rows: k_I, c_kv, k_s
- Decode: Two caches, kept for the whole request; rows: k_I, c_kv, k_s
- k_I: 128 FP8 + 4 B scale = 132 B; c_kv 512 FP8 + 16 B scales + k_s 64 BF16 = 656 B; 788 B per token and layer, 18.5 KiB over 24 layers

All of this figure's data (every setting and operation): [data/flow_dsa.md](assets/interactive/data/flow_dsa.md)

Not all earlier tokens matter equally. After the softmax, most of the weight usually falls on a few of them, and
reading the rest barely changes the output. DSA, the sparse attention that DeepSeek introduced in V3.2, adds a small
indexer in front of MLA to find these tokens. The indexer keeps its own short key, $k_I$, for every token. For each new
token, it scores every earlier token with a cheap dot product and keeps the 2,048 tokens with the highest scores.
The layer then runs ordinary MLA over only these selected tokens.

The indexer still scores the whole context, but its key is small, so each score is cheap. Sparse attention therefore
cuts what each step reads and computes, but not what a request stores: any token may be selected later, so every
token stays in the cache.

In prefill, different tokens select different earlier tokens. Expanding the whole context would rebuild keys and
values that most tokens never read, so vLLM keeps long Kimi K3.1 Pro prompts on the absorbed path in prefill too.

GLM-5.3 and GLM-5.3-Flash also use DSA, each with one change to the indexer. In GLM-5.3, only some layers run an
indexer, and the others reuse the most recent selection. GLM-5.3-Flash keeps one index key for every 4 tokens and
selects tokens in groups of 4.

### MiniMax/Qwen Sparse Attention (MSA/QSA)

**Figure data: MSA in one layer of MiniMax-M3.1 Flash**

Costs are for one whole layer on one GPU, as in the other figures.

*Prefill: data flow*

- Main row: x (N × 6,144) → Projection → Norms, RoPE → Sparse attention → Output projection → out (to the MoE FFN)
- Parallel branch: from Norms, RoPE (q_I) → Index scores → Top-k → joins Sparse attention (16 blocks per group)
- Write: Norms, RoPE (k, k_I) and Projection (v (no norm, no RoPE)) → Write caches → Two caches, kept for the whole request
- Index scores reads k_I from the cache: reads k_I: L earlier + N in chunk
- Sparse attention reads K, V from the cache: reads 16 blocks per group

*Decode: data flow*

- Main row: x (B × 6,144) → Projection → Norms, RoPE → Sparse attention → Output projection → out (to the MoE FFN)
- Parallel branch: from Norms, RoPE (q_I) → Index scores → Top-k → joins Sparse attention (16 blocks per group)
- Write: Norms, RoPE (k, k_I) and Projection (v (no norm, no RoPE)) → Write caches → Two caches, kept for the whole request
- Index scores reads k_I from the cache: reads every k_I: all L tokens
- Sparse attention reads K, V from the cache: reads 16 blocks per group

*Cache*

- Prefill: Two caches, kept for the whole request; rows: k_I, K, V
- Decode: Two caches, kept for the whole request; rows: k_I, K, V
- k_I: 128 FP8 = 128 B; K, V: 4 heads × 128 × 2 FP8 = 1,024 B; 1,152 B per token and layer, 67.5 KiB over 60 layers

All of this figure's data (every setting and operation): [data/flow_msa.md](assets/interactive/data/flow_msa.md)

MSA in MiniMax-M3.1 Flash builds sparse attention on GQA and selects whole blocks of 128 tokens instead of single
tokens. Its indexer keeps one short key per token and scores every earlier token, as in DSA. A block's score is the
highest score of any token in it. Each group of query heads picks its own 16 blocks,
and the block holding the newest tokens is always among them.

**Figure data: QSA in one layer of Qwen3.8 Flash Next**

Costs are for one whole layer on one GPU, as in the other figures.

*Prefill: data flow*

- Main row: x (N × 2,560) → Projection → Norms, RoPE → Sparse attention → Output projection → out (to the MoE FFN)
- Parallel branch: from Projection (q_I, k_raw) → Index prep → Index scores → Top-k → joins Sparse attention (2,048 selected)
- gate, v goes from Projection straight to Sparse attention
- Write: Index prep (k̄, k_raw) → Write index → Two caches, kept for the whole request
- Index scores reads k̄ from the cache: reads every k̄: (L+N)/4
- Sparse attention reads K, V from the cache: appends k, v; reads 2,048 selected

*Decode: data flow*

- Main row: x (B × 2,560) → Projection → Norms, RoPE → Sparse attention → Output projection → out (to the MoE FFN)
- Parallel branch: from Projection (q_I, k_raw) → Index prep → Index scores → Top-k → joins Sparse attention (2,048 selected)
- gate, v goes from Projection straight to Sparse attention
- Write: Index prep (k̄, k_raw) → Write index → Two caches, kept for the whole request
- Index scores reads k̄ from the cache: reads every k̄: L/4
- Sparse attention reads K, V from the cache: appends k, v; reads 2,048 selected

*Cache*

- Prefill: Two caches, kept for the whole request; rows: k̄, K, V
- Decode: Two caches, kept for the whole request; rows: k̄, K, V
- k̄: one 128 B key (FP8) per 4 tokens; K, V: 2 heads × 256 × 2 FP8 = 1,024 B per token and layer. Plus a ring of the last raw keys, 1 KiB per request and layer. 12 QSA layers.

All of this figure's data (every setting and operation): [data/flow_qsa.md](assets/interactive/data/flow_qsa.md)

QSA in Qwen3.8 Flash Next also builds on GQA, but its blocks hold only 4 tokens. It also shrinks the index: the
indexer keeps one key for every block of 4 tokens instead of one per token, so scanning the whole context reads much
less.

### Compressed sparse attention (CSA)

**Figure data: Compressed and shared caches in layer 2 of DeepSeek V4.1 Flash**

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

*Cache*

- Prefill: Caches of layer 2; layers 3–7 read its compressed rows; rows: index keys, compressed, window
- Decode: Caches of layer 2; layers 3–7 read its compressed rows; rows: index keys, compressed, window
- Per 2 tokens: one 288 B entry (NVFP4) and one 68 B index key (FP4), kept for the whole request. Window: 128 × 528 B (MXFP8)

All of this figure's data (every setting and operation): [data/flow_compressed.md](assets/interactive/data/flow_compressed.md)

DeepSeek V4 introduced CSA, which combines the ideas above. The most recent 128 tokens are kept in a window, as in
SWA. Older tokens are compressed: a learned compressor merges a few consecutive tokens into one entry, 4 tokens in
DeepSeek V4 Pro. An indexer, as in DSA, then selects some of these entries, 1,024 in DeepSeek V4 Pro, and attention
reads only the window and the selected entries. Unlike MLA, the cached vector itself serves as the key and the value
of every head, with no $W_{UK}$ or $W_{UV}$.

DeepSeek V4 Pro pairs CSA with heavily compressed attention (HCA) and alternates the two across its layers. An HCA
layer merges every 128 tokens into one entry, so its compressed cache is small enough to read in full. It therefore
needs no indexer: attention reads the window and every compressed entry.

DeepSeek V4.1 Flash uses a revised version, which we call CSA2. It compresses less: layers in the first half merge
every 2 tokens, and layers in the second half do not compress at all. To save memory, it stores the
entries in FP4 and lets several layers share one cache and one selection. Only 4 of the 40 layers write a compressed
cache, and the other layers read the cache of the nearest writer before them. The indexer runs in 8 layers: the 4
writers, and 4 more in the second half that select again from layer 20's cache. Every other layer reuses the
selection of the nearest indexer layer before it. The figure follows layer 2, the first writer. Layers 3 to 7 read
its cache and reuse its selection.

Layer 20 writes the cache that all later layers read, so layers 21 to 39 write no compressed cache. In prefill,
these layers therefore only need to run over the last 128 tokens of the prompt, enough to fill their SWA windows. DeepSeek calls this SWA bounded replay, and vLLM (PR #58132) and SGLang implement it.

### Gated DeltaNet / Kimi Delta Attention (GDN/KDA)

**Figure data: Kimi Delta Attention in one linear layer of GLM-5.3-Flash**

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

*Decode: data flow*

- Main row: x (B × 4,096) → Projection → Short conv → Delta-rule update and read-out → Gated norm → Output projection → out (to the MoE FFN)
- Parallel branch: from Projection (f_a, g_a) → Gates → joins Per-block preparation (f, b), Delta-rule update and read-out (f), Gated norm (g)
- b goes from Projection straight to Delta-rule update and read-out
- Write: Short conv (inputs) → Write conv → State, fixed in size
- Write: Delta-rule update and read-out (S) → Write S → State, fixed in size
- Short conv reads conv from the cache: reads last 3
- Delta-rule update and read-out reads S from the cache: reads S

*Cache*

- Prefill: State, fixed in size; conv, 144 KiB: the prefix's last 3 pad the chunk; its last 3 replace them; S: 64 × 128 × 128 FP32 = 4 MiB; read once, written once
- Decode: State, fixed in size; conv: last 3 inputs, 3 × 24,576 BF16 = 144 KiB; S: 64 heads × 128 × 128, FP32 = 4 MiB
- 4.1 MiB per layer, whatever L is; 141 MiB per request over 34 KDA layers

All of this figure's data (every setting and operation): [data/flow_linear.md](assets/interactive/data/flow_linear.md)

A linear-attention layer keeps no entry per token. Instead, each head keeps one matrix $S$ of fixed size, with one row
per key channel and one column per value channel. A token writes into $S$ by adding the product $k v^\top$ of its key
and value, and reads its output as $o = S^\top q$. Because every token adds to the same matrix, the state never grows.
But it also never forgets, so over a long context the stored values pile up and blur into each other.

Gated DeltaNet (GDN), used by Qwen3.8 Flash Next and Qwen3.8 27B, fixes this in two ways, and each gives it part of
its name. The first is the delta rule. Before writing, the token reads what $S$ already stores for its key, $S^\top k$,
and writes only the difference from its own value $v$. With a write strength $\beta$ between 0 and 1, the update is
$S \leftarrow S + \beta k (v - S^\top k)^\top$. A key that is written again therefore replaces its old value instead
of adding to it. The second is the gate. Before each write, $S$ is multiplied by a decay $\alpha$ between 0 and 1, so
content that is not written again fades over time. In GDN, $\alpha$ is one number per head, so the whole state of a
head fades at one rate.

Kimi Delta Attention (KDA), used by GLM-5.3-Flash, Kimi K3 and Kimi K3.1 Pro, keeps the delta rule but makes the gate
finer. Its $\alpha$ holds one decay per key channel, so each row of $S$ fades at its own rate: some rows keep old
content for long, while others forget quickly. The figure shows a KDA layer. Its Gates step turns each token into the input of
the decay $\alpha$ and into an output gate $g$. The attention kernel then computes $\alpha$ and applies the delta rule,
and $g$ scales the attention output before the output projection, as in Kimi K3's MLA layer.

In decode, each step reads and writes the same state at any context length. In prefill, running the update token by
token would leave the GPU idle. The kernel therefore splits the prompt into chunks of 64 tokens, computes each chunk
with matrix multiplies, and passes $S$ from chunk to chunk. A state of fixed size cannot keep every detail of a long
context, so these models keep a few full-attention, DSA or QSA layers to look back exactly.

> **Takeaway.** Each variant in one line:
>
> - **GQA:** several query heads share one key-value head.
> - **MLA:** each token's keys and values are compressed into one latent vector.
> - **SWA:** the layer keeps only the last $W$ tokens.
> - **DSA:** an indexer selects the top-$k$ tokens, and MLA runs over only those.
> - **MSA:** an indexer selects the top-$k$ blocks of tokens, and GQA runs over only those.
> - **QSA:** like MSA, with much smaller blocks, so the selection is finer.
> - **CSA:** the keys of several tokens merge into one entry, and DSA runs over these entries.
> - **HCA:** many more tokens merge into one entry, so attention reads all entries without an indexer.
> - **CSA2:** a few layers write the compressed keys, and the other layers share them.
> - **GDN:** every token updates a fixed-size state whose old content keeps fading.
> - **KDA:** GDN in which each channel of the state fades at its own rate.

## Attention Memory per Request

Each request keeps its attention state in GPU memory, and this memory limits how many requests a GPU can serve at once. With DP
attention, all of a request's attention state sits on one GPU. We count every KV cache in
FP8, one byte per number, unless the model itself uses FP4, as DeepSeek V4.1 Flash does. Kimi K3.1 Pro is the one exception: the sparse MLA kernel that vLLM
runs for its 96 heads keeps $k_s$ in BF16. Linear-attention states and DeepSeek
V4 Pro's compressor states are kept in FP32. You can explore every layout
on the [Models page](https://servingstudio.cs.washington.edu/models.html?preset=DeepSeek-V4.1-Flash%2Fdeepseek_v41_vllm).

The first figure below splits one request's memory into its parts, at the context set by the slider. The second
lists the total for each model at several contexts, colored from blue for little memory to orange-red for much.
Qwen3-Coder-480B-A35B does not support contexts beyond 256K tokens, so its 1M cell is empty.

**Figure data: Attention memory of one request**

*Total per model by context length*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 496 MiB | 992 MiB | 1.94 GiB | 3.88 GiB | 7.75 GiB | 15.5 GiB | 31.0 GiB | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 57.2 MiB | 102 MiB | 192 MiB | 372 MiB | 732 MiB | 1.42 GiB | 2.82 GiB | 5.64 GiB | 11.3 GiB |
| Inkling | 206 MiB | 294 MiB | 470 MiB | 822 MiB | 1.49 GiB | 2.87 GiB | 5.62 GiB | 11.1 GiB | 22.1 GiB |
| GLM-5.3 | 186 MiB | 373 MiB | 745 MiB | 1.46 GiB | 2.91 GiB | 5.82 GiB | 11.6 GiB | 23.3 GiB | 46.6 GiB |
| MiniMax-M3.1 Flash | 270 MiB | 540 MiB | 1.05 GiB | 2.11 GiB | 4.22 GiB | 8.44 GiB | 16.9 GiB | 33.8 GiB | 67.5 GiB |
| DeepSeek V4 Pro | 43.7 MiB | 65.3 MiB | 108 MiB | 194 MiB | 367 MiB | 711 MiB | 1.37 GiB | 2.71 GiB | 5.40 GiB |
| DeepSeek V4.1 Flash | 6.15 MiB | 9.63 MiB | 16.6 MiB | 30.5 MiB | 58.3 MiB | 114 MiB | 225 MiB | 448 MiB | 893 MiB |
| Kimi K3 | 483 MiB | 537 MiB | 645 MiB | 861 MiB | 1.26 GiB | 2.11 GiB | 3.79 GiB | 7.17 GiB | 13.9 GiB |
| Kimi K3.1 Pro | 502 MiB | 576 MiB | 724 MiB | 1020 MiB | 1.57 GiB | 2.73 GiB | 5.04 GiB | 9.65 GiB | 18.9 GiB |
| GLM-5.3-Flash | 164 MiB | 188 MiB | 234 MiB | 328 MiB | 515 MiB | 890 MiB | 1.60 GiB | 3.06 GiB | 5.99 GiB |
| Qwen3.8 Flash Next | 160 MiB | 209 MiB | 308 MiB | 506 MiB | 902 MiB | 1.65 GiB | 3.20 GiB | 6.30 GiB | 12.5 GiB |
| Qwen3.8 27B | 275 MiB | 403 MiB | 659 MiB | 1.14 GiB | 2.14 GiB | 4.14 GiB | 8.14 GiB | 16.1 GiB | 32.1 GiB |

*Takeaways the figure shows as the slider moves*

- 4K–16K: At short contexts, the fixed linear-attention states dominate. At 4K tokens, Kimi K3 and Kimi K3.1 Pro keep 483 MiB and 502 MiB per request, about as much as the baseline's 496 MiB.
- 32K–64K: As the context grows, the per-token caches catch up with the fixed states, and the bytes per token start to decide.
- 128K–256K: At 128K tokens, DeepSeek V4.1 Flash keeps 139× less than the baseline's 15.5 GiB. Even MiniMax-M3.1 Flash, which keeps the most of the new models, keeps 1.8× less.
- 512K–1M: Past 256K tokens, only the new models go on. At 1M tokens, they keep from 893 MiB in DeepSeek V4.1 Flash to 67.5 GiB in MiniMax-M3.1 Flash.

All of this figure's data (every setting and operation): [data/kv_storage.md](assets/interactive/data/kv_storage.md)

**Figure data: Attention memory of one request as the context grows**

*Per model by context length*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 496 MiB | 992 MiB | 1.94 GiB | 3.88 GiB | 7.75 GiB | 15.5 GiB | 31.0 GiB | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 57.2 MiB | 102 MiB | 192 MiB | 372 MiB | 732 MiB | 1.42 GiB | 2.82 GiB | 5.64 GiB | 11.3 GiB |
| Inkling | 206 MiB | 294 MiB | 470 MiB | 822 MiB | 1.49 GiB | 2.87 GiB | 5.62 GiB | 11.1 GiB | 22.1 GiB |
| GLM-5.3 | 186 MiB | 373 MiB | 745 MiB | 1.46 GiB | 2.91 GiB | 5.82 GiB | 11.6 GiB | 23.3 GiB | 46.6 GiB |
| MiniMax-M3.1 Flash | 270 MiB | 540 MiB | 1.05 GiB | 2.11 GiB | 4.22 GiB | 8.44 GiB | 16.9 GiB | 33.8 GiB | 67.5 GiB |
| DeepSeek V4 Pro | 43.7 MiB | 65.3 MiB | 108 MiB | 194 MiB | 367 MiB | 711 MiB | 1.37 GiB | 2.71 GiB | 5.40 GiB |
| DeepSeek V4.1 Flash | 6.15 MiB | 9.63 MiB | 16.6 MiB | 30.5 MiB | 58.3 MiB | 114 MiB | 225 MiB | 448 MiB | 893 MiB |
| Kimi K3 | 483 MiB | 537 MiB | 645 MiB | 861 MiB | 1.26 GiB | 2.11 GiB | 3.79 GiB | 7.17 GiB | 13.9 GiB |
| Kimi K3.1 Pro | 502 MiB | 576 MiB | 724 MiB | 1020 MiB | 1.57 GiB | 2.73 GiB | 5.04 GiB | 9.65 GiB | 18.9 GiB |
| GLM-5.3-Flash | 164 MiB | 188 MiB | 234 MiB | 328 MiB | 515 MiB | 890 MiB | 1.60 GiB | 3.06 GiB | 5.99 GiB |
| Qwen3.8 Flash Next | 160 MiB | 209 MiB | 308 MiB | 506 MiB | 902 MiB | 1.65 GiB | 3.20 GiB | 6.30 GiB | 12.5 GiB |
| Qwen3.8 27B | 275 MiB | 403 MiB | 659 MiB | 1.14 GiB | 2.14 GiB | 4.14 GiB | 8.14 GiB | 16.1 GiB | 32.1 GiB |

All of this figure's data (every setting and operation): [data/kv_growth.md](assets/interactive/data/kv_growth.md)

## Decode Attention Performance

Next, we look at decode: how many bytes one decode step reads, and how long it takes.

### Bytes read per decode step

We first count the bytes that one decode step reads for one request. The hatched bars in the first figure below show what each
request keeps. While a full-attention or window layer reads all of its cache at every step, a sparse layer reads only
its index and the entries it selects, a fraction of the KV cache storage. At 128K tokens, GLM-5.3 reads only 7.3% of what it keeps
at each step, and MiniMax-M3.1 Flash reads 12.5%.

**Figure data: Bytes one request reads per decode step**

*Total per model by context length*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 496 MiB | 992 MiB | 1.94 GiB | 3.88 GiB | 7.75 GiB | 15.5 GiB | 31.0 GiB | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 57.2 MiB | 102 MiB | 192 MiB | 372 MiB | 732 MiB | 1.42 GiB | 2.82 GiB | 5.64 GiB | 11.3 GiB |
| Inkling | 214 MiB | 302 MiB | 478 MiB | 830 MiB | 1.50 GiB | 2.87 GiB | 5.62 GiB | 11.1 GiB | 22.1 GiB |
| GLM-5.3 | 98.6 MiB | 109 MiB | 131 MiB | 174 MiB | 261 MiB | 434 MiB | 781 MiB | 1.44 GiB | 2.79 GiB |
| MiniMax-M3.1 Flash | 150 MiB | 180 MiB | 240 MiB | 360 MiB | 600 MiB | 1.05 GiB | 1.99 GiB | 3.87 GiB | 7.62 GiB |
| DeepSeek V4 Pro | 61.6 MiB | 66.0 MiB | 74.8 MiB | 92.5 MiB | 128 MiB | 199 MiB | 340 MiB | 623 MiB | 1.16 GiB |
| DeepSeek V4.1 Flash | 9.84 MiB | 11.6 MiB | 15.0 MiB | 21.9 MiB | 35.7 MiB | 63.4 MiB | 119 MiB | 229 MiB | 450 MiB |
| Kimi K3 | 911 MiB | 965 MiB | 1.05 GiB | 1.26 GiB | 1.68 GiB | 2.52 GiB | 4.21 GiB | 7.59 GiB | 14.3 GiB |
| Kimi K3.1 Pro | 900 MiB | 913 MiB | 937 MiB | 987 MiB | 1.06 GiB | 1.25 GiB | 1.64 GiB | 2.41 GiB | 3.96 GiB |
| GLM-5.3-Flash | 294 MiB | 295 MiB | 298 MiB | 304 MiB | 315 MiB | 338 MiB | 383 MiB | 474 MiB | 656 MiB |
| Qwen3.8 Flash Next | 246 MiB | 248 MiB | 251 MiB | 257 MiB | 269 MiB | 293 MiB | 341 MiB | 437 MiB | 629 MiB |
| Qwen3.8 27B | 422 MiB | 550 MiB | 806 MiB | 1.29 GiB | 2.29 GiB | 4.29 GiB | 8.29 GiB | 16.3 GiB | 32.3 GiB |

*Takeaways the figure shows as the slider moves*

- 4K–16K: At short contexts, the linear-attention state is most of what a step reads, because each step reads it and writes it back at any context length. In GLM-5.3-Flash, it is over 90%.
- 32K–128K: A sparse layer keeps every token but reads only its index and the selected entries. At 128K tokens, GLM-5.3 keeps 5.8 GiB per request, but a step reads only 434 MiB, 14× less, and 80% of that is the index.
- 256K–1M: The index read keeps growing with the context. At 1M tokens, it is more than half of a step in every sparse model, and 98% in DeepSeek V4.1 Flash and MiniMax-M3.1 Flash. Kimi K3's dense MLA reads its whole cache instead, 3.6× more than Kimi K3.1 Pro.

All of this figure's data (every setting and operation): [data/decode_bytes.md](assets/interactive/data/decode_bytes.md)

**Figure data: Bytes one request reads per decode step as the context grows**

*Per model by context length*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 496 MiB | 992 MiB | 1.94 GiB | 3.88 GiB | 7.75 GiB | 15.5 GiB | 31.0 GiB | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 57.2 MiB | 102 MiB | 192 MiB | 372 MiB | 732 MiB | 1.42 GiB | 2.82 GiB | 5.64 GiB | 11.3 GiB |
| Inkling | 214 MiB | 302 MiB | 478 MiB | 830 MiB | 1.50 GiB | 2.87 GiB | 5.62 GiB | 11.1 GiB | 22.1 GiB |
| GLM-5.3 | 98.6 MiB | 109 MiB | 131 MiB | 174 MiB | 261 MiB | 434 MiB | 781 MiB | 1.44 GiB | 2.79 GiB |
| MiniMax-M3.1 Flash | 150 MiB | 180 MiB | 240 MiB | 360 MiB | 600 MiB | 1.05 GiB | 1.99 GiB | 3.87 GiB | 7.62 GiB |
| DeepSeek V4 Pro | 61.6 MiB | 66.0 MiB | 74.8 MiB | 92.5 MiB | 128 MiB | 199 MiB | 340 MiB | 623 MiB | 1.16 GiB |
| DeepSeek V4.1 Flash | 9.84 MiB | 11.6 MiB | 15.0 MiB | 21.9 MiB | 35.7 MiB | 63.4 MiB | 119 MiB | 229 MiB | 450 MiB |
| Kimi K3 | 911 MiB | 965 MiB | 1.05 GiB | 1.26 GiB | 1.68 GiB | 2.52 GiB | 4.21 GiB | 7.59 GiB | 14.3 GiB |
| Kimi K3.1 Pro | 900 MiB | 913 MiB | 937 MiB | 987 MiB | 1.06 GiB | 1.25 GiB | 1.64 GiB | 2.41 GiB | 3.96 GiB |
| GLM-5.3-Flash | 294 MiB | 295 MiB | 298 MiB | 304 MiB | 315 MiB | 338 MiB | 383 MiB | 474 MiB | 656 MiB |
| Qwen3.8 Flash Next | 246 MiB | 248 MiB | 251 MiB | 257 MiB | 269 MiB | 293 MiB | 341 MiB | 437 MiB | 629 MiB |
| Qwen3.8 27B | 422 MiB | 550 MiB | 806 MiB | 1.29 GiB | 2.29 GiB | 4.29 GiB | 8.29 GiB | 16.3 GiB | 32.3 GiB |

All of this figure's data (every setting and operation): [data/decode_bytes_grid.md](assets/interactive/data/decode_bytes_grid.md)

### Attention time per decode step

We then measure the attention time of one decode step for a batch of 32 requests on one B200, varying the context
length from 4K to 1M tokens. Each bar adds up the time of the attention kernels in every layer of the model, leaving
out the FFN modules. Its colors show how much of that time each kind of attention takes: global, window, sparse or
linear attention, and the indexer of the sparse layers. Gray segments add the rest of each attention module's work:
the QKV and output projections, norms, RoPE and similar small kernels.

At long contexts, the caches of 32 requests no longer fit in one GPU's memory. No server would run such a batch,
but we still measure its kernel time.

**Figure data: Measured attention time of one decode step**

ms per decode step, whole model · B200, DP attention (all heads on each GPU). Hover a bar for the backend.

*Total per model by context length (Batch 32, With other ops)*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 5.82 ms | 8.19 ms | 12.9 ms | 22.5 ms | 41.1 ms | 78.9 ms | 154 ms | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 2.81 ms | 3.13 ms | 3.89 ms | 5.42 ms | 8.46 ms | 14.5 ms | 26.6 ms | 50.6 ms | 96.3 ms |
| Inkling | 8.77 ms | 9.98 ms | 12.6 ms | 17.7 ms | 27.3 ms | 46.1 ms | 83.6 ms | 159 ms | 307 ms |
| GLM-5.3 | 8.30 ms | 8.38 ms | 8.68 ms | 8.96 ms | 9.49 ms | 10.5 ms | 12.4 ms | 16.2 ms | 23.7 ms |
| MiniMax-M3.1 Flash | 4.32 ms | 4.44 ms | 4.71 ms | 5.38 ms | 6.33 ms | 8.62 ms | 13.2 ms | 22.5 ms | 41.3 ms |
| DeepSeek V4 Pro | 6.57 ms | 6.57 ms | 6.60 ms | 6.92 ms | 7.24 ms | 7.87 ms | 9.02 ms | 11.2 ms | 15.8 ms |
| DeepSeek V4.1 Flash | 3.17 ms | 3.19 ms | 3.26 ms | 3.36 ms | 3.52 ms | 3.83 ms | 4.45 ms | 5.67 ms | 8.08 ms |
| Kimi K3 | 18.7 ms | 19.0 ms | 19.9 ms | 21.7 ms | 25.5 ms | 31.1 ms | 48.8 ms | 79.4 ms | 141 ms |
| Kimi K3.1 Pro | 20.8 ms | 20.9 ms | 21.2 ms | 21.5 ms | 22.1 ms | 23.3 ms | 25.4 ms | 29.8 ms | 38.4 ms |
| GLM-5.3-Flash | 6.01 ms | 6.01 ms | 6.04 ms | 6.07 ms | 6.18 ms | 6.35 ms | 6.61 ms | 7.10 ms | 8.16 ms |
| Qwen3.8 Flash Next | 4.51 ms | 4.52 ms | 4.52 ms | 4.56 ms | 4.63 ms | 4.78 ms | 5.09 ms | 5.56 ms | 6.54 ms |
| Qwen3.8 27B | 5.27 ms | 5.87 ms | 7.05 ms | 9.40 ms | 14.1 ms | 23.5 ms | 42.3 ms | 79.9 ms | 155 ms |

*Backend per model*

| Model | Backend |
|---|---|
| Qwen3-Coder-480B-A35B | FlashInfer trtllm-gen paged prefill/decode, FP8 Q/KV |
| MiMo V2.6 Flash | vLLM FlashAttention 4, FP8 KV, with flash-attention PR #2918 (unpatched FA4 FP8 split-KV decode is wrong) |
| Inkling | per cell the faster of vLLM Inkling fork tml_fa4 with a BF16 cache and SGLang FA4 with MXFP8 Q/K/V (block-scaled QK, in-kernel V dequant), both with the sheared relative bias; Triton fused_qkvr_prep (short conv + q/k norm + cache write) |
| GLM-5.3 | FlashInfer trtllm-gen FP8 sparse MLA, DeepGEMM FP8 indexer, vLLM top-k (cooperative <= 64 rows) |
| MiniMax-M3.1 Flash | vLLM MiniMax-M3 MSA with FP8 K/V and index keys; decode indexer: CuTe DSL index scoring + FlashInfer top-k; decode attention: Triton sparse GQA below 16 requests, CUTLASS fmha_sm100 from 16; prefill indexer: fmha_sm100 score-only (score + block max pool) + FlashInfer top-k; prefill attention: CuTe DSL KV-outer sparse attention + combine |
| DeepSeek V4 Pro | vLLM FlashMLA sparse MLA on the fp8_ds_mla cache, all 128 heads: BF16 prefill (flash_mla_sparse_fwd), FP8 decode under CUDA graphs (Sm100 Head128 split-KV); CSA (C4) layers attend the 128-token window + top-1024 compressed entries, HCA (C128) layers the window + every compressed entry, one call per layer; DeepGEMM FP8 indexer logits, vLLM top-k (cooperative <= 64 rows, persistent above) |
| DeepSeek V4.1 Flash | FlashMLA (compressed sparse MLA; CuTe DSL swap-AB decode for steps of <= 16 requests), DeepGEMM FP8 indexer, vLLM top-k; prefill with decoder bounded replay, as vLLM's V4.1 fork runs it by default |
| Kimi K3 | FlashInfer cute-dsl MLA decode (FP8 q/KV, all 96 heads, no padding); TRT-LLM-gen ragged prefill (vLLM TRTLLM_RAGGED, FP8 Q/K/V); FlashKDA prefill and fused CUDA KDA decode as Kimi K3.1 Pro |
| Kimi K3.1 Pro | vLLM FlashMLA sparse MLA on the fp8_ds_mla cache: FP8 decode (BF16 q padded to 128 heads); prefill upconverts the context to BF16 (BF16 sparse attention at 128 heads); DeepGEMM FP8 indexer, vLLM top-k (cooperative <= 64 rows); FlashKDA prefill, vLLM fused CUDA KDA decode (fused_kda_decode) |
| GLM-5.3-Flash | FlashInfer trtllm-gen FP8 sparse MLA, DeepGEMM FP8 indexer, vLLM top-k (cooperative <= 64 rows); FlashKDA prefill, FLA Triton KDA decode |
| Qwen3.8 Flash Next | vLLM Triton sparse GQA, vLLM top-k (cooperative <= 64 rows); FlashInfer / Triton GDN |
| Qwen3.8 27B | FlashInfer trtllm-gen paged prefill/decode, FP8 Q/KV (FA4 has no FP8-KV path at head dim 256); FlashInfer CUTLASS GDN chunked prefill, vLLM Triton GDN decode |

*Takeaways the figure shows as the slider moves*

- 4K–16K, batch 32: At short contexts, fixed work decides. Kimi K3 and Kimi K3.1 Pro read and write their large linear-attention states at every step, so up to 8K tokens they are slower than the baseline. Inkling is slower too, because its 55 window layers take about 3.5 ms per step at any context.
- 32K–128K, batch 32: The baseline's time grows with the bytes it reads, to 75.9 ms at 128K tokens. DeepSeek V4.1 Flash takes 1.7 ms there, 44× less. Inkling takes 42.3 ms: it reads less than Qwen3.8 27B but takes twice as long.
- 256K–1M, batch 32: At long contexts, the indexer takes most of the time in GLM-5.3 and MiniMax-M3.1 Flash, and the models with full-attention layers slow down the most. At 1M tokens, Qwen3.8 Flash Next takes 4.1 ms per step and Inkling 303 ms.
- 4K–32K, batch 1: With one request, a step reads too little to keep the GPU busy. Up to 32K tokens, every model takes 0.3 to 2.0 ms in attention kernels.
- 64K–256K, batch 1: The baseline's time grows with the context, to 5.4 ms at 256K tokens. The other new models stay at or below 2.3 ms, but Inkling reaches 3.5 ms, twice the time of Qwen3.8 27B, which reads more bytes per step.
- 512K–1M, batch 1: Even at 1M tokens, GLM-5.3-Flash and Qwen3.8 Flash Next take less than 0.7 ms per step.

All of this figure's data (every setting and operation): [data/decode_time.md](assets/interactive/data/decode_time.md)

**Figure data: Measured attention time of one decode step as the context grows**

*Per model by context length (Batch 32, With other ops)*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 5.82 ms | 8.19 ms | 12.9 ms | 22.5 ms | 41.1 ms | 78.9 ms | 154 ms | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 2.81 ms | 3.13 ms | 3.89 ms | 5.42 ms | 8.46 ms | 14.5 ms | 26.6 ms | 50.6 ms | 96.3 ms |
| Inkling | 8.77 ms | 9.98 ms | 12.6 ms | 17.7 ms | 27.3 ms | 46.1 ms | 83.6 ms | 159 ms | 307 ms |
| GLM-5.3 | 8.30 ms | 8.38 ms | 8.68 ms | 8.96 ms | 9.49 ms | 10.5 ms | 12.4 ms | 16.2 ms | 23.7 ms |
| MiniMax-M3.1 Flash | 4.32 ms | 4.44 ms | 4.71 ms | 5.38 ms | 6.33 ms | 8.62 ms | 13.2 ms | 22.5 ms | 41.3 ms |
| DeepSeek V4 Pro | 6.57 ms | 6.57 ms | 6.60 ms | 6.92 ms | 7.24 ms | 7.87 ms | 9.02 ms | 11.2 ms | 15.8 ms |
| DeepSeek V4.1 Flash | 3.17 ms | 3.19 ms | 3.26 ms | 3.36 ms | 3.52 ms | 3.83 ms | 4.45 ms | 5.67 ms | 8.08 ms |
| Kimi K3 | 18.7 ms | 19.0 ms | 19.9 ms | 21.7 ms | 25.5 ms | 31.1 ms | 48.8 ms | 79.4 ms | 141 ms |
| Kimi K3.1 Pro | 20.8 ms | 20.9 ms | 21.2 ms | 21.5 ms | 22.1 ms | 23.3 ms | 25.4 ms | 29.8 ms | 38.4 ms |
| GLM-5.3-Flash | 6.01 ms | 6.01 ms | 6.04 ms | 6.07 ms | 6.18 ms | 6.35 ms | 6.61 ms | 7.10 ms | 8.16 ms |
| Qwen3.8 Flash Next | 4.51 ms | 4.52 ms | 4.52 ms | 4.56 ms | 4.63 ms | 4.78 ms | 5.09 ms | 5.56 ms | 6.54 ms |
| Qwen3.8 27B | 5.27 ms | 5.87 ms | 7.05 ms | 9.40 ms | 14.1 ms | 23.5 ms | 42.3 ms | 79.9 ms | 155 ms |

All of this figure's data (every setting and operation): [data/decode_time_grid.md](assets/interactive/data/decode_time_grid.md)

Decode attention mostly moves data, so its time should follow the bytes that a step reads. The figure below shows how fast each model's attention kernels move these bytes, as their achieved memory bandwidth: the bytes that a decode step reads, divided by the time of its attention kernels. Only the
baseline and Qwen3.8 27B come close to the GPU's limit. At 128K tokens, they reach about 7 TB/s, near
the B200's peak of 8 TB/s. The new designs do not turn all of their byte savings into time. DeepSeek V4.1 Flash
reads 250 times fewer bytes than the baseline but is only 44 times faster, because its sparse attention takes about
1 ms per step at every context length. Inkling's time does not follow its bytes either, because its kernels reach only about 2.3 TB/s at 128K tokens.

**Figure data: Achieved memory bandwidth of decode attention**

*Per model by context length (Batch 32)*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 5.81 TB/s | 6.37 TB/s | 6.68 TB/s | 6.83 TB/s | 6.99 TB/s | 7.02 TB/s | 7.03 TB/s | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 1.93 TB/s | 2.61 TB/s | 3.11 TB/s | 3.47 TB/s | 3.70 TB/s | 3.83 TB/s | 3.92 TB/s | 3.97 TB/s | 4.10 TB/s |
| Inkling | 1.43 TB/s | 1.62 TB/s | 1.82 TB/s | 1.99 TB/s | 2.18 TB/s | 2.33 TB/s | 2.42 TB/s | 2.47 TB/s | 2.51 TB/s |
| GLM-5.3 | 2.53 TB/s | 2.65 TB/s | 2.61 TB/s | 2.97 TB/s | 3.50 TB/s | 4.19 TB/s | 4.87 TB/s | 5.38 TB/s | 5.73 TB/s |
| MiniMax-M3.1 Flash | 2.34 TB/s | 2.66 TB/s | 3.18 TB/s | 3.77 TB/s | 4.84 TB/s | 5.63 TB/s | 6.20 TB/s | 6.54 TB/s | 6.69 TB/s |
| DeepSeek V4 Pro | 1.50 TB/s | 1.61 TB/s | 1.79 TB/s | 1.45 TB/s | 1.74 TB/s | 2.16 TB/s | 2.69 TB/s | 3.25 TB/s | 3.63 TB/s |
| DeepSeek V4.1 Flash | 0.31 TB/s | 0.35 TB/s | 0.43 TB/s | 0.58 TB/s | 0.84 TB/s | 1.23 TB/s | 1.71 TB/s | 2.19 TB/s | 2.58 TB/s |
| Kimi K3 | 5.53 TB/s | 5.56 TB/s | 5.36 TB/s | 5.10 TB/s | 4.71 TB/s | 4.84 TB/s | 4.06 TB/s | 3.94 TB/s | 3.86 TB/s |
| Kimi K3.1 Pro | 4.39 TB/s | 4.38 TB/s | 4.31 TB/s | 4.33 TB/s | 4.42 TB/s | 4.59 TB/s | 4.89 TB/s | 5.22 TB/s | 5.55 TB/s |
| GLM-5.3-Flash | 4.01 TB/s | 4.04 TB/s | 4.03 TB/s | 4.06 TB/s | 4.03 TB/s | 4.06 TB/s | 4.21 TB/s | 4.48 TB/s | 4.78 TB/s |
| Qwen3.8 Flash Next | 4.01 TB/s | 4.03 TB/s | 4.06 TB/s | 4.09 TB/s | 4.15 TB/s | 4.22 TB/s | 4.34 TB/s | 4.71 TB/s | 5.16 TB/s |
| Qwen3.8 27B | 5.81 TB/s | 6.08 TB/s | 6.41 TB/s | 6.74 TB/s | 6.97 TB/s | 7.13 TB/s | 7.21 TB/s | 7.26 TB/s | 7.28 TB/s |

Color scale: 0 TB/s to 8 TB/s, the B200's peak.

All of this figure's data (every setting and operation): [data/decode_bandwidth_grid.md](assets/interactive/data/decode_bandwidth_grid.md)

The rest of the attention module also takes time, mostly in the QKV and output projections. This cost does not
grow with the context, because each step reads the same projection weights. For the baseline at 128K tokens, it is
only 4% of the step. The new models, however, have cut their attention kernels so far that the projections now
take a large share: 67% in GLM-5.3, 60% in Kimi K3.1 Pro and 55% in DeepSeek V4.1 Flash. In these three models,
even an infinitely fast attention kernel would shorten the step by less than half. With a single request, the
projections and other small kernels take more than half of the step in every new model.

## Prefill Attention Performance

Next, we look at prefill: how much attention work one prompt needs, and how long it takes. The new models need far less attention work than the baseline at long prompts, but several of them save much less time, because their kernels run well below the GPU's peak.

### Attention FLOPs per prompt

We first count the attention FLOPs to prefill one prompt, adding up every layer of the model. The colors of each bar show how much of the work each kind of attention does. A sparse layer attends to a fixed number of selected tokens, so its attention grows only linearly with the prompt. Its indexer, however, still scores every earlier token, so at long prompts the indexer does most of the work.

**Figure data: Attention FLOPs to prefill one prompt**

whole model, one copy · score, weighted sum, indexer scoring and linear-layer chunks; projections, softmax and top-k not counted

*Total per model by prompt length*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 25.6 TFLOP | 102 TFLOP | 409 TFLOP | 1.64 PFLOP | 6.54 PFLOP | 26.2 PFLOP | 105 PFLOP | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 3.92 TFLOP | 14.0 TFLOP | 52.8 TFLOP | 205 TFLOP | 805 TFLOP | 3.19 PFLOP | 12.7 PFLOP | 50.8 PFLOP | 203 PFLOP |
| Inkling | 6.57 TFLOP | 19.4 TFLOP | 63.3 TFLOP | 224 TFLOP | 834 TFLOP | 3.22 PFLOP | 12.6 PFLOP | 50.0 PFLOP | 199 PFLOP |
| GLM-5.3 | 69.8 TFLOP | 165 TFLOP | 365 TFLOP | 799 TFLOP | 1.80 PFLOP | 4.37 PFLOP | 11.7 PFLOP | 35.3 PFLOP | 118 PFLOP |
| MiniMax-M3.1 Flash | 12.9 TFLOP | 30.9 TFLOP | 70.1 TFLOP | 161 TFLOP | 392 TFLOP | 1.05 PFLOP | 3.16 PFLOP | 10.6 PFLOP | 38.0 PFLOP |
| DeepSeek V4 Pro | 26.3 TFLOP | 72.3 TFLOP | 174 TFLOP | 414 TFLOP | 1.05 PFLOP | 2.91 PFLOP | 9.03 PFLOP | 30.9 PFLOP | 113 PFLOP |
| DeepSeek V4.1 Flash | 6.40 TFLOP | 13.6 TFLOP | 29.0 TFLOP | 63.9 TFLOP | 150 TFLOP | 389 TFLOP | 1.13 PFLOP | 3.67 PFLOP | 13.0 PFLOP |
| Kimi K3 | 16.9 TFLOP | 58.5 TFLOP | 216 TFLOP | 828 TFLOP | 3.24 PFLOP | 12.8 PFLOP | 51.0 PFLOP | 203 PFLOP | 812 PFLOP |
| Kimi K3.1 Pro | 37.7 TFLOP | 89.2 TFLOP | 202 TFLOP | 468 TFLOP | 1.16 PFLOP | 3.17 PFLOP | 9.73 PFLOP | 33.0 PFLOP | 120 PFLOP |
| GLM-5.3-Flash | 10.7 TFLOP | 24.9 TFLOP | 54.3 TFLOP | 118 TFLOP | 263 TFLOP | 625 TFLOP | 1.64 PFLOP | 4.83 PFLOP | 15.9 PFLOP |
| Qwen3.8 Flash Next | 3.06 TFLOP | 6.79 TFLOP | 14.4 TFLOP | 30.3 TFLOP | 64.4 TFLOP | 143 TFLOP | 339 TFLOP | 889 TFLOP | 2.62 PFLOP |
| Qwen3.8 27B | 4.87 TFLOP | 16.3 TFLOP | 59.1 TFLOP | 224 TFLOP | 870 TFLOP | 3.43 PFLOP | 13.6 PFLOP | 54.2 PFLOP | 217 PFLOP |

*Takeaways the figure shows as the slider moves*

- 4K–16K: At short prompts, selecting a fixed number of tokens saves little. At 4K tokens, GLM-5.3, Kimi K3.1 Pro and DeepSeek V4 Pro even need more attention FLOPs than the baseline, and GLM-5.3 still does at 8K.
- 32K–128K: The baseline's FLOPs grow with the square of the prompt, while the sparse models grow about twice per doubling. At 128K tokens, Qwen3.8 Flash Next needs 183× less than the baseline.
- 256K–1M: The indexer still scores every earlier token, so its share keeps rising. At 1M tokens, it is 60% to 90% of the attention FLOPs in the sparse models.

All of this figure's data (every setting and operation): [data/prefill_flops.md](assets/interactive/data/prefill_flops.md)

**Figure data: Attention FLOPs to prefill one prompt as the prompt grows**

*Per model by prompt length*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 25.6 TFLOP | 102 TFLOP | 409 TFLOP | 1.64 PFLOP | 6.54 PFLOP | 26.2 PFLOP | 105 PFLOP | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 3.92 TFLOP | 14.0 TFLOP | 52.8 TFLOP | 205 TFLOP | 805 TFLOP | 3.19 PFLOP | 12.7 PFLOP | 50.8 PFLOP | 203 PFLOP |
| Inkling | 6.57 TFLOP | 19.4 TFLOP | 63.3 TFLOP | 224 TFLOP | 834 TFLOP | 3.22 PFLOP | 12.6 PFLOP | 50.0 PFLOP | 199 PFLOP |
| GLM-5.3 | 69.8 TFLOP | 165 TFLOP | 365 TFLOP | 799 TFLOP | 1.80 PFLOP | 4.37 PFLOP | 11.7 PFLOP | 35.3 PFLOP | 118 PFLOP |
| MiniMax-M3.1 Flash | 12.9 TFLOP | 30.9 TFLOP | 70.1 TFLOP | 161 TFLOP | 392 TFLOP | 1.05 PFLOP | 3.16 PFLOP | 10.6 PFLOP | 38.0 PFLOP |
| DeepSeek V4 Pro | 26.3 TFLOP | 72.3 TFLOP | 174 TFLOP | 414 TFLOP | 1.05 PFLOP | 2.91 PFLOP | 9.03 PFLOP | 30.9 PFLOP | 113 PFLOP |
| DeepSeek V4.1 Flash | 6.40 TFLOP | 13.6 TFLOP | 29.0 TFLOP | 63.9 TFLOP | 150 TFLOP | 389 TFLOP | 1.13 PFLOP | 3.67 PFLOP | 13.0 PFLOP |
| Kimi K3 | 16.9 TFLOP | 58.5 TFLOP | 216 TFLOP | 828 TFLOP | 3.24 PFLOP | 12.8 PFLOP | 51.0 PFLOP | 203 PFLOP | 812 PFLOP |
| Kimi K3.1 Pro | 37.7 TFLOP | 89.2 TFLOP | 202 TFLOP | 468 TFLOP | 1.16 PFLOP | 3.17 PFLOP | 9.73 PFLOP | 33.0 PFLOP | 120 PFLOP |
| GLM-5.3-Flash | 10.7 TFLOP | 24.9 TFLOP | 54.3 TFLOP | 118 TFLOP | 263 TFLOP | 625 TFLOP | 1.64 PFLOP | 4.83 PFLOP | 15.9 PFLOP |
| Qwen3.8 Flash Next | 3.06 TFLOP | 6.79 TFLOP | 14.4 TFLOP | 30.3 TFLOP | 64.4 TFLOP | 143 TFLOP | 339 TFLOP | 889 TFLOP | 2.62 PFLOP |
| Qwen3.8 27B | 4.87 TFLOP | 16.3 TFLOP | 59.1 TFLOP | 224 TFLOP | 870 TFLOP | 3.43 PFLOP | 13.6 PFLOP | 54.2 PFLOP | 217 PFLOP |

All of this figure's data (every setting and operation): [data/prefill_flops_grid.md](assets/interactive/data/prefill_flops_grid.md)

### Attention time per prompt

We then measure the attention time of the same prompts on one B200. Each prompt is prefilled in chunks of 8K tokens. Each bar adds up the time of all its chunks. As in decode, the colors show each kind of attention, and gray
segments add the rest of each attention module's work.

**Figure data: Measured attention time to prefill one prompt**

ms to prefill the prompt in 8K chunks, whole model · B200, DP attention (all heads on each GPU). Hover a bar for the backend.

*Total per model by prompt length (With other ops)*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 70.1 ms | 151 ms | 425 ms | 1,329 ms | 4,570 ms | 16,826 ms | 64,641 ms | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 29.6 ms | 62.2 ms | 138 ms | 330 ms | 873 ms | 2,604 ms | 8,594 ms | 30,573 ms | 115,938 ms |
| Inkling | 82.3 ms | 167 ms | 355 ms | 788 ms | 1,893 ms | 5,091 ms | 15,378 ms | 51,402 ms | 184,970 ms |
| GLM-5.3 | 148 ms | 347 ms | 704 ms | 1,448 ms | 3,064 ms | 6,736 ms | 15,716 ms | 40,148 ms | 116,001 ms |
| MiniMax-M3.1 Flash | 55.0 ms | 118 ms | 249 ms | 518 ms | 1,091 ms | 2,371 ms | 5,525 ms | 14,344 ms | 41,901 ms |
| DeepSeek V4 Pro | 126 ms | 271 ms | 544 ms | 1,095 ms | 2,247 ms | 4,966 ms | 12,167 ms | 34,113 ms | 109,076 ms |
| DeepSeek V4.1 Flash | 17.2 ms | 33.4 ms | 67.4 ms | 138 ms | 289 ms | 634 ms | 1,497 ms | 4,010 ms | 12,629 ms |
| Kimi K3 | 296 ms | 596 ms | 1,259 ms | 2,778 ms | 6,578 ms | 17,277 ms | 51,040 ms | 168,021 ms | 599,805 ms |
| Kimi K3.1 Pro | 327 ms | 703 ms | 1,418 ms | 2,880 ms | 5,941 ms | 12,565 ms | 27,690 ms | 65,380 ms | 171,781 ms |
| GLM-5.3-Flash | 86.1 ms | 173 ms | 348 ms | 699 ms | 1,414 ms | 2,896 ms | 6,091 ms | 13,409 ms | 32,604 ms |
| Qwen3.8 Flash Next | 54.6 ms | 113 ms | 233 ms | 474 ms | 967 ms | 1,992 ms | 4,236 ms | 9,210 ms | 20,542 ms |
| Qwen3.8 27B | 43.8 ms | 89.1 ms | 192 ms | 437 ms | 1,085 ms | 2,996 ms | 9,514 ms | 33,426 ms | 123,175 ms |

*Backend per model*

| Model | Backend |
|---|---|
| Qwen3-Coder-480B-A35B | FlashInfer trtllm-gen paged prefill/decode, FP8 Q/KV |
| MiMo V2.6 Flash | vLLM FlashAttention 4, FP8 KV, with flash-attention PR #2918 (unpatched FA4 FP8 split-KV decode is wrong) |
| Inkling | per cell the faster of vLLM Inkling fork tml_fa4 with a BF16 cache and SGLang FA4 with MXFP8 Q/K/V (block-scaled QK, in-kernel V dequant), both with the sheared relative bias; Triton fused_qkvr_prep (short conv + q/k norm + cache write) |
| GLM-5.3 | FlashInfer trtllm-gen FP8 sparse MLA, DeepGEMM FP8 indexer, vLLM top-k (cooperative <= 64 rows) |
| MiniMax-M3.1 Flash | vLLM MiniMax-M3 MSA with FP8 K/V and index keys; decode indexer: CuTe DSL index scoring + FlashInfer top-k; decode attention: Triton sparse GQA below 16 requests, CUTLASS fmha_sm100 from 16; prefill indexer: fmha_sm100 score-only (score + block max pool) + FlashInfer top-k; prefill attention: CuTe DSL KV-outer sparse attention + combine |
| DeepSeek V4 Pro | vLLM FlashMLA sparse MLA on the fp8_ds_mla cache, all 128 heads: BF16 prefill (flash_mla_sparse_fwd), FP8 decode under CUDA graphs (Sm100 Head128 split-KV); CSA (C4) layers attend the 128-token window + top-1024 compressed entries, HCA (C128) layers the window + every compressed entry, one call per layer; DeepGEMM FP8 indexer logits, vLLM top-k (cooperative <= 64 rows, persistent above) |
| DeepSeek V4.1 Flash | FlashMLA (compressed sparse MLA; CuTe DSL swap-AB decode for steps of <= 16 requests), DeepGEMM FP8 indexer, vLLM top-k; prefill with decoder bounded replay, as vLLM's V4.1 fork runs it by default |
| Kimi K3 | FlashInfer cute-dsl MLA decode (FP8 q/KV, all 96 heads, no padding); TRT-LLM-gen ragged prefill (vLLM TRTLLM_RAGGED, FP8 Q/K/V); FlashKDA prefill and fused CUDA KDA decode as Kimi K3.1 Pro |
| Kimi K3.1 Pro | vLLM FlashMLA sparse MLA on the fp8_ds_mla cache: FP8 decode (BF16 q padded to 128 heads); prefill upconverts the context to BF16 (BF16 sparse attention at 128 heads); DeepGEMM FP8 indexer, vLLM top-k (cooperative <= 64 rows); FlashKDA prefill, vLLM fused CUDA KDA decode (fused_kda_decode) |
| GLM-5.3-Flash | FlashInfer trtllm-gen FP8 sparse MLA, DeepGEMM FP8 indexer, vLLM top-k (cooperative <= 64 rows); FlashKDA prefill, FLA Triton KDA decode |
| Qwen3.8 Flash Next | vLLM Triton sparse GQA, vLLM top-k (cooperative <= 64 rows); FlashInfer / Triton GDN |
| Qwen3.8 27B | FlashInfer trtllm-gen paged prefill/decode, FP8 Q/KV (FA4 has no FP8-KV path at head dim 256); FlashInfer CUTLASS GDN chunked prefill, vLLM Triton GDN decode |

*Takeaways the figure shows as the slider moves*

- 4K–16K: At short prompts, sparse attention saves little. At 8K tokens, the baseline's dense attention is still faster than the attention of GLM-5.3, DeepSeek V4 Pro, Kimi K3 and Kimi K3.1 Pro.
- 32K–128K: The baseline's time grows with the square of the prompt, and from 32K tokens on it is slower than every new model. At 128K tokens, GLM-5.3 needs 1.4× the attention FLOPs of MiMo V2.6 Flash and takes 1.7× as long.
- 256K–1M: At long prompts, the indexer takes a growing share of the time. At 1M tokens, it is most of the time in every sparse model except Qwen3.8 Flash Next, which takes 13.9 s in total against 86.8 s for GLM-5.3.

All of this figure's data (every setting and operation): [data/prefill_time.md](assets/interactive/data/prefill_time.md)

**Figure data: Measured attention time to prefill one prompt as the prompt grows**

*Per model by prompt length (With other ops)*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 70.1 ms | 151 ms | 425 ms | 1.33 s | 4.57 s | 16.8 s | 64.6 s | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 29.6 ms | 62.2 ms | 138 ms | 330 ms | 873 ms | 2.60 s | 8.59 s | 30.6 s | 116 s |
| Inkling | 82.3 ms | 167 ms | 355 ms | 788 ms | 1.89 s | 5.09 s | 15.4 s | 51.4 s | 185 s |
| GLM-5.3 | 148 ms | 347 ms | 704 ms | 1.45 s | 3.06 s | 6.74 s | 15.7 s | 40.1 s | 116 s |
| MiniMax-M3.1 Flash | 55.0 ms | 118 ms | 249 ms | 518 ms | 1.09 s | 2.37 s | 5.53 s | 14.3 s | 41.9 s |
| DeepSeek V4 Pro | 126 ms | 271 ms | 544 ms | 1.09 s | 2.25 s | 4.97 s | 12.2 s | 34.1 s | 109 s |
| DeepSeek V4.1 Flash | 17.2 ms | 33.4 ms | 67.4 ms | 138 ms | 289 ms | 634 ms | 1.50 s | 4.01 s | 12.6 s |
| Kimi K3 | 296 ms | 596 ms | 1.26 s | 2.78 s | 6.58 s | 17.3 s | 51.0 s | 168 s | 600 s |
| Kimi K3.1 Pro | 327 ms | 703 ms | 1.42 s | 2.88 s | 5.94 s | 12.6 s | 27.7 s | 65.4 s | 172 s |
| GLM-5.3-Flash | 86.1 ms | 173 ms | 348 ms | 699 ms | 1.41 s | 2.90 s | 6.09 s | 13.4 s | 32.6 s |
| Qwen3.8 Flash Next | 54.6 ms | 113 ms | 233 ms | 474 ms | 967 ms | 1.99 s | 4.24 s | 9.21 s | 20.5 s |
| Qwen3.8 27B | 43.8 ms | 89.1 ms | 192 ms | 437 ms | 1.09 s | 3.00 s | 9.51 s | 33.4 s | 123 s |

All of this figure's data (every setting and operation): [data/prefill_time_grid.md](assets/interactive/data/prefill_time_grid.md)

To compare time with work, we use the achieved compute throughput: the attention FLOPs of a prompt divided by the time of its attention kernels. The figure below shows it for every model and prompt length.

**Figure data: Achieved compute throughput of prefill attention**

*Per model by prompt length*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 0.81 PFLOP/s | 1.37 PFLOP/s | 1.51 PFLOP/s | 1.60 PFLOP/s | 1.65 PFLOP/s | 1.68 PFLOP/s | 1.68 PFLOP/s | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 0.73 PFLOP/s | 1.02 PFLOP/s | 1.27 PFLOP/s | 1.50 PFLOP/s | 1.66 PFLOP/s | 1.75 PFLOP/s | 1.81 PFLOP/s | 1.85 PFLOP/s | 1.85 PFLOP/s |
| Inkling | 0.29 PFLOP/s | 0.39 PFLOP/s | 0.53 PFLOP/s | 0.70 PFLOP/s | 0.87 PFLOP/s | 1.00 PFLOP/s | 1.09 PFLOP/s | 1.14 PFLOP/s | 1.17 PFLOP/s |
| GLM-5.3 | 2.07 PFLOP/s | 1.39 PFLOP/s | 1.47 PFLOP/s | 1.49 PFLOP/s | 1.46 PFLOP/s | 1.42 PFLOP/s | 1.39 PFLOP/s | 1.38 PFLOP/s | 1.36 PFLOP/s |
| MiniMax-M3.1 Flash | 0.41 PFLOP/s | 0.45 PFLOP/s | 0.47 PFLOP/s | 0.50 PFLOP/s | 0.56 PFLOP/s | 0.66 PFLOP/s | 0.80 PFLOP/s | 0.94 PFLOP/s | 1.07 PFLOP/s |
| DeepSeek V4 Pro | 0.77 PFLOP/s | 0.83 PFLOP/s | 0.99 PFLOP/s | 1.16 PFLOP/s | 1.23 PFLOP/s | 1.26 PFLOP/s | 1.30 PFLOP/s | 1.30 PFLOP/s | 1.28 PFLOP/s |
| DeepSeek V4.1 Flash | 0.87 PFLOP/s | 0.91 PFLOP/s | 0.95 PFLOP/s | 1.00 PFLOP/s | 1.07 PFLOP/s | 1.15 PFLOP/s | 1.25 PFLOP/s | 1.30 PFLOP/s | 1.27 PFLOP/s |
| Kimi K3 | 0.28 PFLOP/s | 0.44 PFLOP/s | 0.66 PFLOP/s | 0.91 PFLOP/s | 1.16 PFLOP/s | 1.35 PFLOP/s | 1.46 PFLOP/s | 1.53 PFLOP/s | 1.57 PFLOP/s |
| Kimi K3.1 Pro | 0.44 PFLOP/s | 0.39 PFLOP/s | 0.43 PFLOP/s | 0.48 PFLOP/s | 0.54 PFLOP/s | 0.64 PFLOP/s | 0.78 PFLOP/s | 0.94 PFLOP/s | 1.08 PFLOP/s |
| GLM-5.3-Flash | 0.39 PFLOP/s | 0.44 PFLOP/s | 0.47 PFLOP/s | 0.51 PFLOP/s | 0.54 PFLOP/s | 0.61 PFLOP/s | 0.69 PFLOP/s | 0.81 PFLOP/s | 0.90 PFLOP/s |
| Qwen3.8 Flash Next | 0.11 PFLOP/s | 0.11 PFLOP/s | 0.11 PFLOP/s | 0.11 PFLOP/s | 0.12 PFLOP/s | 0.12 PFLOP/s | 0.13 PFLOP/s | 0.15 PFLOP/s | 0.19 PFLOP/s |
| Qwen3.8 27B | 0.54 PFLOP/s | 0.81 PFLOP/s | 1.10 PFLOP/s | 1.39 PFLOP/s | 1.63 PFLOP/s | 1.81 PFLOP/s | 1.86 PFLOP/s | 1.87 PFLOP/s | 1.89 PFLOP/s |

Color scale: 0 PFLOP/s to 4.5 PFLOP/s, the B200's FP8 peak.

All of this figure's data (every setting and operation): [data/prefill_throughput_grid.md](assets/interactive/data/prefill_throughput_grid.md)

At 128K tokens, the baseline, MiMo V2.6 Flash, Qwen3.8 27B, Kimi K3, GLM-5.3 and DeepSeek V4 Pro all reach 1.3 to 1.8 PFLOP/s on one B200, so their prefill time roughly follows their FLOPs. GLM-5.3 spends more than a third of its attention time on the indexer and the top-k selection. At 1M tokens, these two take most of the attention time in every model with an indexer, except Qwen3.8 Flash Next.

The other new models run their attention more slowly, so their time falls by much less than their FLOPs. Qwen3.8 Flash Next needs 183
times fewer FLOPs than the baseline but is only 13 times faster. Most of that time goes to its sparse attention, which reaches only about 90 TFLOP/s. Kimi K3.1 Pro's attention reaches only about 640 TFLOP/s, because vLLM runs its sparse
MLA prefill in BF16 and pads its 96 heads to 128. Inkling needs about as many attention FLOPs as MiMo V2.6 Flash, but
its attention takes 1.8 times as long and reaches about 1.0 PFLOP/s. Both models run FlashAttention 4, but Inkling's version also adds a learned position bias to every score, which may account for part of the gap.

## Conclusion and Get Started

Attention in open models is no longer one design. At 128K tokens, the eleven new models keep 1.8× to 139× less
attention memory than a GQA model of similar size, and they read and compute far less in decode and prefill. The
time saved is smaller than the work saved, because the new designs add work, such as indexers, and some of their
kernels run far below the GPU's peak. Inference analysis and optimization should start from these models.

The kernel times in this post come from ServingStudio Sim's measured kernel profiles.
ServingStudio Sim also models whole serving steps, beyond attention. On the
[Models page](https://servingstudio.cs.washington.edu/models.html), you can explore models such as Inkling,
DeepSeek V4 Flash, GLM-5.2 and Qwen3-235B. For each one, the page shows its attention layout and cache parts, the
cost tree that builds one step's time from measured kernels, and a live prediction for a batch of your own. The
simulator is on [GitHub](https://github.com/SyFI-ServingStudio/ServingStudio).
