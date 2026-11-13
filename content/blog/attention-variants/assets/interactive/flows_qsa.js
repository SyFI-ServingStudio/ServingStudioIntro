/* Flow figure for the block-sparse subsection (`<!-- interactive: flow_qsa -->`):
   one QSA layer of Qwen3.8 Flash Next. Layout contract: see flows.js.

   Config main/model/config/qwen3_8_flash_next_nvfp4.json: hidden 2,560;
   24 query heads and 2 KV heads of 256; partial_rotary_factor 0.25 (64
   channels, interleaved mRoPE); output_gate_type sigmoid; indexer_n_heads 4,
   indexer_kv_heads 1, indexer_head_dim 128, indexer_compress_ratio 4,
   indexer_budget 2,048 tokens (512 groups of 4). 12 of the 48 layers are
   QSA ("full_attention" in layer_types); the other 36 are GDN.

   What vLLM runs (vllm/models/qwen4_exp/nvidia/qsa.py Qwen4ExpQSAAttention
   .forward and indexer_qsa.py; ServingStudio Sim's
   simulator/src/worklet/vllm_qsa_attn_local.rs and
   op/attention/sparse_gqa_indexer.rs follow the same launches):
   - qkv_proj over x: q with its gate half (2 × 24 × 256), k, v (2 × 256 each),
     and the indexer's 4 index query heads and one raw index key of 128.
     vLLM merged the indexer projection into this GEMM in #59533
     (fuse_indexer_projection; on whenever the attention weights are
     unquantized, as here). ServingStudio Sim still times index_qk_proj as a
     GEMM of its own, a 2,560 → 640 GEMM beside a 2,560 → 13,312 one.
   - _project_qkv_gate: per-head RMSNorm and partial RoPE on q and k, and the
     gate split, in one kernel.
   - The indexer's q_I and k_raw come from that GEMM and do not go through
     q or k, so the indexer is a lane forked from the projection.
   - indexer prepare (qsa_pre_indexer): RMSNorm and RoPE of the index query,
     cast to FP8; this token's raw key into a ring of 4 (BF16, per request);
     every 4th token, the 4 raw keys are averaged, normalized, rotated and
     stored in FP8 as one pooled key. The figure splits it into Index prep
     and Write index (the stores), the latter drawn under it.
   - indexer logits: Σ_h ReLU(q_I,h · k̄) for every complete group;
     top-k keeps 512 groups; expand turns them into token indices, plus the
     tokens of the last, incomplete group, which are always kept.
   - _run_qsa then appends k, v to the FP8 K/V cache (do_kv_cache_update)
     and runs qsa_sparse_paged_attention over the selected tokens with the
     sigmoid output gate applied inside; the figure folds the append into
     the attention card, which therefore writes the K/V row itself.
   - o_proj.
   dtypes: the NVFP4 quantization_config targets only the MoE experts, so
   every attention weight is BF16 (the indexer's included); activations
   BF16; K/V cache FP8 (2 heads × 256 × 2 = 1,024 B per token and layer);
   pooled keys FP8, 128 B per 4 tokens; the raw-key ring 4 × 128 BF16 =
   1 KiB per request and layer.
   Operands (flows.js contract): decode L counts this token too;
   k = min(L, 2,048) selected tokens (the tail of under 4 tokens is left out,
   as in the prefill_flops figure); prefill reads count the L + N rows of the
   request once. Index scores are counted on chip. Costs cover one whole
   layer on one GPU. */

(() => {
  const { axis, tokens, operand, rows, rowCount, cacheRows, cappedPairs, isDecode } = window.ATTN_FLOW_KIT;
  const HIDDEN = 2560;
  const Q_HEADS = 24;
  const KV_HEADS = 2;
  const HEAD_DIM = 256;
  const ROPE = 64;
  const INDEX_HEADS = 4;
  const INDEX_DIM = 128;
  const RATIO = 4;
  const BUDGET = 2048;
  const RING = 4;
  const Q_WIDTH = Q_HEADS * HEAD_DIM; // 6,144
  const QKV_WIDTH = 2 * Q_WIDTH + 2 * KV_HEADS * HEAD_DIM; // 13,312
  const INDEX_WIDTH = (INDEX_HEADS + 1) * INDEX_DIM; // 640
  const hidden = axis("D", HIDDEN);
  const qHeads = axis("H_q", Q_HEADS);
  const kvHeads = axis("H_kv", KV_HEADS);
  const headDim = axis("d", HEAD_DIM);
  const qWidth = axis("H_q·d", Q_WIDTH);
  const indexHeads = axis("H_I", INDEX_HEADS);
  const indexDim = axis("d_I", INDEX_DIM);
  // pooled keys made this step: decode B/4 (one per request every 4th token); prefill N/4
  const groupRows = (scale) => (isDecode(scale) ? [tokens("B/4", scale.batch / RATIO)] : [tokens("N/4", scale.chunk / RATIO)]);
  // pooled keys of one request: decode L/4; prefill (L+N)/4
  const groupCache = (scale) =>
    isDecode(scale)
      ? [tokens("B", scale.batch), tokens("L/4", Math.floor(scale.context / RATIO))]
      : [tokens("(L+N)/4", Math.floor((scale.context + scale.chunk) / RATIO))];
  const groupAxis = (scale) =>
    isDecode(scale) ? tokens("L/4", Math.floor(scale.context / RATIO)) : tokens("(L+N)/4", Math.floor((scale.context + scale.chunk) / RATIO));
  const selected = (scale) => tokens("k", Math.min(BUDGET, isDecode(scale) ? scale.context : scale.context + 1));
  const sparseRows = (scale) => (isDecode(scale) ? [tokens("B", scale.batch), selected(scale), kvHeads] : [...cacheRows(scale), kvHeads]);
  const ring = (scale) => operand("ring", [...(isDecode(scale) ? [tokens("B", scale.batch)] : []), axis("4 tokens", RING), indexDim], "BF16");
  const groupPairs = (scale) => cappedPairs(scale, Infinity, RATIO);
  const attentionPairs = (scale) => cappedPairs(scale, BUDGET);

  window.ATTN_BLOG_FLOWS.qsa = {
    title: "QSA in one layer of Qwen3.8 Flash Next",
    note: "Costs are for one whole layer on one GPU, as in the other figures.",
    axes: [
      ["B", "requests"],
      ["N", "tokens in the chunk"],
      ["L", { decode: "tokens in each request's cache, the new one included", prefill: "earlier tokens in the cache" }],
      ["k", "selected tokens: 512 blocks of 4, min(L, 2,048)"],
      ["H_q", "query heads"],
      ["H_kv", "KV heads"],
      ["H_I", "index query heads"],
      ["d", "head dim"],
      ["d_I", "index head dim"],
      ["D", "hidden size"],
    ],
    input: { label: "x", note: { prefill: "N × 2,560", decode: "B × 2,560" } },
    output: { label: "out", note: "to the MoE FFN" },
    // q_I and k_raw leave the shared projection GEMM and skip q and k: the
    // indexer runs beside the main path and joins it at attention with the
    // selected tokens.
    lanes: {
      indexer: { side: "below", from: { step: "project", label: "q_I, k_raw" }, to: [{ step: "attend", label: "2,048 selected" }] },
    },
    // gate and v leave the projection and skip the norms (they get no norm,
    // no RoPE) on their own edge over the main row (flows.js `edges`)
    edges: [{ from: "project", to: "attend", label: "gate, v", level: "over", dy: 16 }],
    steps: [
      {
        key: "project",
        title: "Projection",
        role: "weights",
        formula: "x · [W_q, W_gate, W_k, W_v, W_Iq, W_Ik]",
        does: "Turns each token's hidden state into a query for each of the 24 heads, an output gate of the same size, and a key and a value for each of the 2 KV heads. The same GEMM gives the indexer 4 small index queries and one raw index key.",
        counted: "2 per weight per token, 2 × 2,560 × 13,952 = 71.4 MFLOP a token.",
        operands: (scale) => ({
          input: [operand("x", [...rows(scale), hidden], "BF16")],
          weight: [operand("W_qkv", [hidden, axis("2·H_q·d+2·H_kv·d", QKV_WIDTH)], "BF16"), operand("W_Iqk", [hidden, axis("(H_I+1)·d_I", INDEX_WIDTH)], "BF16")],
          output: [
            operand("q", [...rows(scale), qHeads, headDim], "BF16"),
            operand("gate", [...rows(scale), qWidth], "BF16"),
            operand("k, v", [...rows(scale), axis("2·H_kv", 2 * KV_HEADS), headDim], "BF16"),
            operand("q_I", [...rows(scale), indexHeads, indexDim], "BF16"),
            operand("k_raw", [...rows(scale), indexDim], "BF16"),
          ],
          flops: 2 * rowCount(scale) * HIDDEN * (QKV_WIDTH + INDEX_WIDTH),
          summed: ["D"],
        }),
      },
      {
        key: "norms",
        title: "Norms, RoPE",
        role: "elementwise",
        edgeIn: "q, k",
        formula: "q, k per head: ÷ rms × γ, rotate 64 channels",
        does: "Rescales each head of q and k and turns 64 of its 256 channels by the token's position. The same kernel splits the gate off q.",
        counted: "about 4 per number for the norms and 3 per rotated number.",
        operands: (scale) => ({
          input: [operand("q", [...rows(scale), qHeads, headDim], "BF16"), operand("k", [...rows(scale), kvHeads, headDim], "BF16")],
          weight: [operand("γ_q, γ_k", [axis("2·d", 2 * HEAD_DIM)], "BF16")],
          output: [operand("q", [...rows(scale), qHeads, headDim], "BF16"), operand("k", [...rows(scale), kvHeads, headDim], "BF16")],
          flops: rowCount(scale) * (Q_HEADS + KV_HEADS) * (4 * HEAD_DIM + 3 * ROPE),
          summed: ["d"],
        }),
      },
      {
        key: "prep",
        title: "Index prep",
        role: "elementwise",
        lane: "indexer",
        edgeIn: "q_I, k_raw",
        formula: "q_I: ÷ rms × γ, rotate; k̄ = norm(mean of 4 raw keys), rotated",
        does: {
          prefill: "Normalizes and rotates the index queries and casts them to FP8. Averages each complete block of 4 raw keys into one pooled key, then normalizes and rotates it.",
          decode: "Normalizes and rotates the index queries and casts them to FP8. Every 4th token completes a block: its 4 raw keys, 3 of them from the ring, are averaged into one pooled key, then normalized and rotated.",
        },
        counted: "about 4 per number for the norms, 3 per rotated number and 1 per raw key number for the mean. Decode makes a quarter of a pooled key per request and step on average (B/4).",
        operands: (scale) => ({
          input: [operand("q_I", [...rows(scale), indexHeads, indexDim], "BF16"), operand("k_raw", [...rows(scale), indexDim], "BF16")],
          weight: [operand("γ_Iq, γ_Ik", [axis("2·d_I", 2 * INDEX_DIM)], "BF16")],
          state: [ring(scale)],
          // k_raw goes on to the write card's ring, in the same kernel (on chip)
          output: [
            operand("q_I", [...rows(scale), indexHeads, indexDim], "FP8"),
            operand("k̄", [...groupRows(scale), indexDim], "BF16", { onChip: true }),
            operand("k_raw", [...(isDecode(scale) ? rows(scale) : [axis("≤ 3", RING - 1)]), indexDim], "BF16", { onChip: true }),
          ],
          flops: rowCount(scale) * (INDEX_HEADS * (4 * INDEX_DIM + 3 * ROPE) + INDEX_DIM) + (rowCount(scale) / RATIO) * (4 * INDEX_DIM + 3 * ROPE),
          summed: ["d_I"],
        }),
      },
      {
        key: "index",
        title: "Index scores",
        role: "core",
        lane: "indexer",
        edgeIn: "q_I",
        formula: "Σ_h ReLU(q_I,h · k̄_g), every complete block g",
        does: {
          prefill: "Scores every complete block of 4 tokens each query may see against its pooled key. One number per block comes out, not one per head.",
          decode: "Scores every complete block of 4 earlier tokens against its pooled key. One number per block comes out, not one per head.",
        },
        counted: "2 × 4 × 128 = 1,024 FLOP per query–block pair (a query at position j sees j/4 blocks). The scores stay on chip.",
        operands: (scale) => ({
          input: [operand("q_I", [...rows(scale), indexHeads, indexDim], "FP8")],
          cache: [operand("k̄", [...groupCache(scale), indexDim], "FP8")],
          output: [operand("I", [...rows(scale), groupAxis(scale)], "FP32", { onChip: true })],
          flops: 2 * INDEX_HEADS * INDEX_DIM * groupPairs(scale),
          summed: ["d_I", "H_I"],
        }),
      },
      {
        key: "topk",
        title: "Top-k",
        role: "elementwise",
        lane: "indexer",
        edgeIn: "scores",
        formula: "top 512 blocks → 2,048 token positions",
        does: {
          prefill: "For each query, keeps its 512 best blocks and expands them into 2,048 token positions, plus the tokens of its last, incomplete block.",
          decode: "Keeps the 512 best blocks and expands them into 2,048 token positions, plus the tokens of the last, incomplete block. With 512 blocks or fewer, it keeps them all.",
        },
        counted: "a selection, not arithmetic: about 1 comparison per block score, counted as 1 FLOP. Only the token positions go to HBM.",
        operands: (scale) => ({
          input: [operand("I", [...rows(scale), groupAxis(scale)], "FP32", { onChip: true })],
          output: [operand("idx", [...rows(scale), selected(scale)], "INT32")],
          flops: groupPairs(scale),
          summed: [],
        }),
      },
      {
        key: "attend",
        title: "Sparse attention",
        role: "core",
        edgeIn: "q, k",
        formula: "append k, v;  o = softmax(q · Kᵀ ÷ √256) · V over 2,048 tokens;  o × sigmoid(gate)",
        does: {
          prefill: "Appends the chunk's k and v to the K/V cache. Then each query attends to its own 2,048 selected tokens, and the output gate scales the result.",
          decode: "Appends this token's k and v to the K/V cache. Then the query attends to the 2,048 selected tokens only, and the output gate scales the result.",
        },
        counted: {
          prefill: "2 × 24 × (256 + 256) = 24.6 KFLOP per query–key pair (scores and weighted sum); the query at position j keeps min(j, 2,048) keys; plus about 1 per number for the FP8 append and 5 per number for the gate. Each query gathers its own tokens; read once each, the chunk touches at most the L + N rows of the request, counted as all of them.",
          decode: "2 × 24 × (256 + 256) = 24.6 KFLOP per query–key pair, B × k pairs; plus about 1 per number for the FP8 append and 5 per number for the gate.",
        },
        operands: (scale) => ({
          input: [
            operand("q", [...rows(scale), qHeads, headDim], "BF16"),
            operand("gate", [...rows(scale), qWidth], "BF16"),
            operand("k, v", [...rows(scale), axis("2·H_kv", 2 * KV_HEADS), headDim], "BF16"),
            operand("idx", [...rows(scale), selected(scale)], "INT32"),
          ],
          cache: [operand("K", [...sparseRows(scale), headDim], "FP8"), operand("V", [...sparseRows(scale), headDim], "FP8")],
          output: [operand("k, v", [...rows(scale), axis("2·H_kv", 2 * KV_HEADS), headDim], "FP8"), operand("o", [...rows(scale), qWidth], "BF16")],
          flops: 2 * Q_HEADS * 2 * HEAD_DIM * attentionPairs(scale) + rowCount(scale) * (2 * KV_HEADS * HEAD_DIM + 5 * Q_WIDTH),
          summed: ["d", "k"],
        }),
      },
      {
        key: "out",
        title: "Output proj.",
        fullTitle: "Output projection",
        role: "weights",
        edgeIn: "o",
        formula: "concat(o) · W_o",
        does: "Joins the 24 gated heads into 6,144 numbers and maps them back to 2,560.",
        counted: "2 per weight per token, 2 × 6,144 × 2,560 = 31.5 MFLOP a token.",
        operands: (scale) => ({
          input: [operand("o", [...rows(scale), qWidth], "BF16")],
          weight: [operand("W_o", [qWidth, hidden], "BF16")],
          output: [operand("out", [...rows(scale), hidden], "BF16")],
          flops: 2 * rowCount(scale) * Q_WIDTH * HIDDEN,
          summed: ["H_q·d"],
        }),
      },
    ],
    branch: {
      title: "Write index",
      role: "write",
      place: "under",
      formula: "pooled key in FP8; raw key into the ring",
      does: {
        prefill: "Stores the chunk's pooled keys in FP8 in the index cache, and keeps the raw keys of its last, incomplete block in the ring. It is the end of the Index prep kernel.",
        decode: "Puts this token's raw key into the ring. Every 4th token, it stores the new pooled key in FP8 in the index cache. It is the end of the Index prep kernel.",
      },
      counted: "about 1 per number to quantize the pooled key; the raw key is copied.",
      operands: (scale) => ({
        input: [operand("k̄", [...groupRows(scale), indexDim], "BF16", { onChip: true }), operand("k_raw", [...(isDecode(scale) ? rows(scale) : [axis("≤ 3", RING - 1)]), indexDim], "BF16", { onChip: true })],
        output: [operand("k̄", [...groupRows(scale), indexDim], "FP8"), operand("ring", [...(isDecode(scale) ? rows(scale) : [axis("≤ 3", RING - 1)]), indexDim], "BF16")],
        flops: (rowCount(scale) / RATIO) * INDEX_DIM,
        summed: [],
      }),
      from: [{ step: "prep", label: "k̄, k_raw" }],
    },
    cache: {
      title: "Two caches, kept for the whole request",
      rows: [
        {
          label: "k̄",
          cells: { prefill: ["t0–3", "t4–7", "t8–11", "…", "t(L−4)–(L−1)"], decode: ["t0–3", "t4–7", "t8–11", "…", "t(L−8)–(L−5)"] },
          newCell: { prefill: "N / 4", decode: "every 4th" },
        },
        {
          label: "K, V",
          cells: { prefill: ["t0", "t1", "t2", "…", "t(L−1)"], decode: ["t0", "t1", "t2", "…", "t(L−2)"] },
          newCell: { prefill: "chunk", decode: "new" },
        },
      ],
      labelWidth: 62,
      cellWidth: 96,
      groups: { old: { prefill: "L earlier tokens", decode: "L − 1 earlier tokens" }, new: { prefill: "N in this chunk", decode: "this token" } },
      note: "k̄: one 128 B key (FP8) per 4 tokens; K, V: 2 heads × 256 × 2 FP8 = 1,024 B per token and layer. Plus a ring of the last raw keys, 1 KiB per request and layer. 12 QSA layers.",
      writes: [{ step: "attend", row: 1, dx: 14, dy: 7, label: "" }],
      reads: [
        { step: "index", row: 0, label: { prefill: "reads every k̄: (L+N)/4", decode: "reads every k̄: L/4" } },
        { step: "attend", row: 1, dx: -14, dy: -7, label: "appends k, v; reads 2,048 selected" },
      ],
    },
  };
})();
