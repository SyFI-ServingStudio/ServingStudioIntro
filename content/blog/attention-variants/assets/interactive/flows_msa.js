/* Flow figure for the block-sparse subsection (`<!-- interactive: flow_msa -->`):
   one MSA layer of MiniMax-M3.1 Flash. Layout contract: see flows.js.

   Config (huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash config.json, sha
   ce7f4df8; blog/data/measured/runs/RUNS_msa.md): hidden 6,144; 64 query
   heads and 4 KV heads of 128; rotary_dim 64 (partial NeoX RoPE); 60 layers,
   all MSA. sparse_attention_config: index dim 128, 4 index query heads (one
   per KV group) and one index key head shared by all groups, top-16 blocks
   of 128 tokens, score "max", local block 1 (always kept).

   What vLLM runs (vllm/models/minimax_m3/nvidia/model.py,
   MiniMaxM3SparseAttention.forward):
   - qkv_proj: one MXFP8 GEMM over x for [q | k | v | index_q | index_k],
     6,144 → 9,856 (64 + 4 + 4 + 4 + 1 heads of 128). Its MXFP8 input comes
     from the layer's input norm, outside this figure.
   - fused_minimax_m3_qknorm_rope_kv_insert: per-head Gemma RMSNorm and
     partial RoPE on q, k, index_q and index_k, then k, v into the paged K/V
     cache and index_k into the index cache, all in one launch. The figure
     draws the norms and the cache write as two cards (Norms, RoPE; Write
     caches), both fed by that one kernel. index_q leaves it in FP8, the
     index cache's dtype.
   - self.indexer(index_q) (MiniMaxM3Indexer): scores index_q against every
     cached index key, takes each 128-token block's maximum, and keeps the top
     16 blocks per KV group, the local block always among them; it writes
     them to topk_indices_buffer. The figure splits it into Index scores
     (score and block max) and Top-k.
   - self.impl.forward(q): each KV group's 16 query heads attend over the K/V
     of that group's 16 blocks only (2,048 tokens).
   - o_proj: BF16 → MXFP8 quant, then an MXFP8 GEMM 8,192 → 6,144.
   Lanes (flows.js contract): the indexer needs only index_q (and the index
   cache), so it runs beside the main q, which waits for the selection; the
   write card sits under Norms, RoPE because that kernel does the inserts.
   dtypes: quant_method mxfp8 with weight_block_size [1, 32] covers the
   attention projections, so W_qkv and W_o are MXFP8 (FP8 numbers, one
   1-byte scale per 32); activations BF16. Caches in FP8 as the blog counts
   them (precision rule; vLLM's Blackwell recipe serves NVFP4): K/V 4 heads ×
   128 × 2 = 1,024 B per token and layer, index key 128 B.
   Operands (flows.js contract): decode L counts this token too;
   k = min(L, 2,048) tokens per KV group; prefill reads count the L + N rows
   of the request once. Index scores and block scores are counted on chip
   (vLLM writes a score buffer). Costs cover the whole layer on one GPU. */

(() => {
  const { axis, tokens, operand, rows, rowCount, cacheRows, causalPairs, cappedPairs, isDecode } = window.ATTN_FLOW_KIT;
  const HIDDEN = 6144;
  const Q_HEADS = 64;
  const KV_HEADS = 4;
  const HEAD_DIM = 128;
  const ROPE = 64;
  const INDEX_HEADS = 4;
  const INDEX_DIM = 128;
  const BLOCK = 128;
  const TOP_BLOCKS = 16;
  const TOP_TOKENS = TOP_BLOCKS * BLOCK; // 2,048
  const QKV_WIDTH = (Q_HEADS + 2 * KV_HEADS + INDEX_HEADS + 1) * HEAD_DIM; // 9,856
  const NORMED_HEADS = Q_HEADS + KV_HEADS + INDEX_HEADS + 1; // q, k, index_q, index_k
  const hidden = axis("D", HIDDEN);
  const qHeads = axis("H_q", Q_HEADS);
  const kvHeads = axis("H_kv", KV_HEADS);
  const headDim = axis("d", HEAD_DIM);
  const indexHeads = axis("H_I", INDEX_HEADS);
  const indexDim = axis("d_I", INDEX_DIM);
  const qWidth = axis("H_q·d", Q_HEADS * HEAD_DIM);
  // tokens one query's KV group keeps: decode min(L, 2,048)
  const selected = (scale) => tokens("k", Math.min(TOP_TOKENS, isDecode(scale) ? scale.context : scale.context + 1));
  // blocks one query scores: decode ⌈L/128⌉; prefill, those of the L + N tokens
  const blockAxis = (scale) =>
    isDecode(scale) ? tokens("L/128", Math.ceil(scale.context / BLOCK)) : tokens("(L+N)/128", Math.ceil((scale.context + scale.chunk) / BLOCK));
  // the K/V a sparse step reads: decode, each KV group its own k tokens; prefill, at most every row of the request once
  const sparseRows = (scale) => (isDecode(scale) ? [tokens("B", scale.batch), kvHeads, selected(scale)] : [...cacheRows(scale), kvHeads]);
  const attentionPairs = (scale) => cappedPairs(scale, TOP_TOKENS);
  const perRow = (name, heads, dtype, scale) => operand(name, [...rows(scale), heads, headDim], dtype);

  window.ATTN_BLOG_FLOWS.msa = {
    title: "MSA in one layer of MiniMax-M3.1 Flash",
    note: "Costs are for one whole layer on one GPU, as in the other figures.",
    axes: [
      ["B", "requests"],
      ["N", "tokens in the chunk"],
      ["L", { decode: "tokens in each request's cache, the new one included", prefill: "earlier tokens in the cache" }],
      ["k", "tokens a KV group keeps: its 16 blocks of 128, min(L, 2,048)"],
      ["H_q", "query heads"],
      ["H_kv", "KV heads, one per group of 16 query heads"],
      ["H_I", "index query heads, one per KV group"],
      ["d", "head dim"],
      ["d_I", "index head dim"],
      ["D", "hidden size"],
    ],
    input: { label: "x", note: { prefill: "N × 6,144", decode: "B × 6,144" } },
    output: { label: "out", note: "to the MoE FFN" },
    lanes: {
      indexer: { side: "below", from: { step: "norms", label: "q_I" }, to: [{ step: "attend", label: "16 blocks per group" }] },
    },
    steps: [
      {
        key: "project",
        title: "Projection",
        role: "weights",
        formula: "x · [W_q, W_k, W_v, W_Iq, W_Ik]",
        does: "Turns each token's hidden state into a query, a key and a value, plus the indexer's query, one head per KV group, and its key, one head shared by all groups. One GEMM makes all five.",
        counted: "2 per weight per token, 2 × 6,144 × 9,856 = 121 MFLOP a token.",
        operands: (scale) => ({
          input: [operand("x", [...rows(scale), hidden], "MXFP8")],
          weight: [operand("W_qkv,I", [hidden, axis("(H_q+2H_kv+H_I+1)·d", QKV_WIDTH)], "MXFP8")],
          output: [
            perRow("q", qHeads, "BF16", scale),
            perRow("k, v", axis("2·H_kv", 2 * KV_HEADS), "BF16", scale),
            operand("q_I", [...rows(scale), indexHeads, indexDim], "BF16"),
            operand("k_I", [...rows(scale), indexDim], "BF16"),
          ],
          flops: 2 * rowCount(scale) * HIDDEN * QKV_WIDTH,
          summed: ["D"],
        }),
      },
      {
        key: "norms",
        title: "Norms, RoPE",
        role: "elementwise",
        formula: "per head: ÷ rms × γ, rotate 64 channels",
        does: "Rescales each head of q, k, q_I and k_I and turns 64 of its channels by the token's position. The same kernel writes k, v and k_I into the caches and casts q_I to FP8.",
        counted: "about 4 per number for the norms and 3 per rotated number.",
        operands: (scale) => ({
          input: [
            perRow("q", qHeads, "BF16", scale),
            operand("k", [...rows(scale), kvHeads, headDim], "BF16"),
            operand("q_I", [...rows(scale), indexHeads, indexDim], "BF16"),
            operand("k_I", [...rows(scale), indexDim], "BF16"),
          ],
          weight: [operand("γ ×4", [axis("4·d", 4 * HEAD_DIM)], "BF16")],
          output: [
            perRow("q", qHeads, "BF16", scale),
            operand("k", [...rows(scale), kvHeads, headDim], "BF16", { onChip: true }),
            operand("q_I", [...rows(scale), indexHeads, indexDim], "FP8"),
            operand("k_I", [...rows(scale), indexDim], "BF16", { onChip: true }),
          ],
          flops: rowCount(scale) * NORMED_HEADS * (4 * HEAD_DIM + 3 * ROPE),
          summed: ["d"],
        }),
      },
      {
        key: "index",
        title: "Index scores",
        role: "core",
        lane: "indexer",
        formula: "s_g(t) = q_I,g · k_I(t); block score = max over its 128 tokens",
        does: {
          prefill: "Scores every key each query may see, once for each KV group's index query, then gives every 128-token block the highest score of any token in it.",
          decode: "Scores every earlier token, once for each KV group's index query, then gives every 128-token block the highest score of any token in it.",
        },
        counted: {
          prefill: "2 × 4 × 128 = 1,024 FLOP per query–key pair over the N × L + N(N+1)/2 causal pairs, plus 1 comparison per score for the block max. The scores stay on chip.",
          decode: "2 × 4 × 128 = 1,024 FLOP per query–key pair, B × L pairs, plus 1 comparison per score for the block max. The scores stay on chip.",
        },
        operands: (scale) => ({
          input: [operand("q_I", [...rows(scale), indexHeads, indexDim], "FP8")],
          cache: [operand("k_I", [...cacheRows(scale), indexDim], "FP8")],
          output: [operand("block scores", [...rows(scale), indexHeads, blockAxis(scale)], "FP32", { onChip: true })],
          flops: (2 * INDEX_DIM + 1) * INDEX_HEADS * causalPairs(scale),
          summed: ["d_I"],
        }),
      },
      {
        key: "topk",
        title: "Top-k",
        role: "elementwise",
        lane: "indexer",
        edgeIn: "scores",
        formula: "top 16 blocks per group; the local block always",
        does: {
          prefill: "For each query and KV group, keeps the 16 best blocks, the block holding the query itself always among them.",
          decode: "For each KV group, keeps the 16 best blocks, the block holding the newest tokens always among them. With 16 blocks or fewer, it keeps them all.",
        },
        counted: "a selection, not arithmetic: about 1 comparison per block score, counted as 1 FLOP. Only the block positions go to HBM.",
        operands: (scale) => ({
          input: [operand("block scores", [...rows(scale), indexHeads, blockAxis(scale)], "FP32", { onChip: true })],
          output: [operand("idx", [...rows(scale), indexHeads, axis("16", TOP_BLOCKS)], "INT32")],
          flops: INDEX_HEADS * cappedPairs(scale, Infinity, BLOCK),
          summed: [],
        }),
      },
      {
        key: "attend",
        title: "Sparse attention",
        role: "core",
        edgeIn: "q",
        formula: "softmax(q · Kᵀ ÷ √128) · V over each group's 16 blocks",
        does: {
          prefill: "Each query head attends to the 2,048 tokens of the 16 blocks its KV group selected for that query. Scores and weights stay inside the kernel.",
          decode: "Each query head attends to the 2,048 tokens of the 16 blocks its KV group selected. The 4 groups read different blocks.",
        },
        counted: {
          prefill: "2 × 64 × (128 + 128) = 32.8 KFLOP per query–key pair (scores and weighted sum); the query at position j keeps min(j, 2,048) keys (the masked part of its local block is not taken out). Each query gathers its own blocks; read once each, the chunk touches at most the L + N rows of the request, counted as all of them.",
          decode: "2 × 64 × (128 + 128) = 32.8 KFLOP per query–key pair, B × k pairs. Each group reads its own k tokens of its own KV head.",
        },
        operands: (scale) => ({
          input: [perRow("q", qHeads, "BF16", scale), operand("idx", [...rows(scale), indexHeads, axis("16", TOP_BLOCKS)], "INT32")],
          cache: [operand("K", [...sparseRows(scale), headDim], "FP8"), operand("V", [...sparseRows(scale), headDim], "FP8")],
          output: [operand("o", [...rows(scale), qWidth], "BF16")],
          flops: 2 * Q_HEADS * 2 * HEAD_DIM * attentionPairs(scale),
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
        does: "Joins the 64 heads into 8,192 numbers and maps them back to 6,144.",
        counted: "2 per weight per token, 2 × 8,192 × 6,144 = 101 MFLOP a token; the cast of o to MXFP8 before the GEMM is not counted.",
        operands: (scale) => ({
          input: [operand("o", [...rows(scale), qWidth], "BF16")],
          weight: [operand("W_o", [qWidth, hidden], "MXFP8")],
          output: [operand("out", [...rows(scale), hidden], "BF16")],
          flops: 2 * rowCount(scale) * Q_HEADS * HEAD_DIM * HIDDEN,
          summed: ["H_q·d"],
        }),
      },
    ],
    branch: {
      title: "Write caches",
      role: "write",
      place: "under",
      formula: "quantize, append k, v and k_I",
      does: {
        prefill: "Stores the chunk's k and v in the K/V cache and its k_I in the index cache, in FP8. It is the end of the Norms, RoPE kernel.",
        decode: "Stores this token's k and v in the K/V cache and its k_I in the index cache, in FP8. It is the end of the Norms, RoPE kernel.",
      },
      counted: "about 1 per number to quantize.",
      operands: (scale) => ({
        input: [
          operand("k", [...rows(scale), kvHeads, headDim], "BF16", { onChip: true }),
          operand("v", [...rows(scale), kvHeads, headDim], "BF16"),
          operand("k_I", [...rows(scale), indexDim], "BF16", { onChip: true }),
        ],
        output: [operand("k, v", [...rows(scale), axis("2·H_kv", 2 * KV_HEADS), headDim], "FP8"), operand("k_I", [...rows(scale), indexDim], "FP8")],
        flops: rowCount(scale) * (2 * KV_HEADS * HEAD_DIM + INDEX_DIM),
        summed: [],
      }),
      // v skips the norms and RoPE: it goes from the projection to the insert
      from: [
        { step: "norms", label: "k, k_I" },
        { step: "project", label: "v (no norm, no RoPE)", labelSide: "left" },
      ],
    },
    cache: {
      title: "Two caches, kept for the whole request",
      rows: ["k_I", "K, V"],
      labelWidth: 62,
      newLabel: { prefill: "chunk", decode: "new" },
      // decode: L counts this token too, so the earlier columns end at t(L−2)
      cells: { prefill: undefined, decode: ["t0", "t1", "t2", "…", "t(L−2)"] },
      groups: { old: { prefill: "L earlier tokens", decode: "L − 1 earlier tokens" }, new: { prefill: "N in this chunk", decode: "this token" } },
      note: "k_I: 128 FP8 = 128 B; K, V: 4 heads × 128 × 2 FP8 = 1,024 B; 1,152 B per token and layer, 67.5 KiB over 60 layers",
      reads: [
        { step: "index", row: 0, label: { prefill: "reads k_I: L earlier + N in chunk", decode: "reads every k_I: all L tokens" } },
        { step: "attend", row: 1, label: "reads 16 blocks per group" },
      ],
    },
  };
})();
