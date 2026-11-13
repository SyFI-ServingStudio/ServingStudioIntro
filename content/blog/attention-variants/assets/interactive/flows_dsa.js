/* Flow figure for the sparse-attention subsection
   (`<!-- interactive: flow_dsa -->`): one Gated DSA layer of Kimi K3.1 Pro.
   Layout contract: see flows.js.

   Config main/model/config/kimi_k31_pro.json: the MLA dimensions of the
   flow_mla figure (hidden 7,168, 96 heads, q_lora_rank 1,536,
   kv_lora_rank 512, qk_nope 128, qk_rope 64 (k_s; Kimi K3 applies no RoPE), v 128, output
   gate) plus dsa_config: index_num_heads 32, index_head_dim 128,
   index_topk 2,048, index_rope_dim 0, index_k_norm_type layernorm,
   index_topk_freq 1 (every one of the 24 layers runs its own indexer).

   What vLLM runs (kimi_k3/nvidia/dsa.py, KimiK3DSAAttention, fused forward):
   - One GEMM over x with packed [q_a | kv_a | wk | weights_proj] gives c_q,
     c_kv, k_pe (k_s here), the index key k_I (128) and the per-head index
     weights w (32).
   - dsa_fused_prologue: RMSNorm of c_q and c_kv, LayerNorm of k_I, FP8 quant,
     and the writes into both caches (latent cache and index-key cache).
   - One GEMM over c_q with packed [q_b | wq_index] gives q (96 x 192) and
     the index query q_I (32 x 128).
   - indexer.score -> sparse_attn_indexer: q_I in FP8 with its scale and the
     softmax and head scales folded into w; DeepGEMM MQA logits
     sum_h w_h * relu(q_I,h . k_I) over every earlier index key; top-k 2,048
     (decode: cooperative_topk for at most 64 rows, persistent_topk past
     that, indexer_topk.py AUTO_COOPERATIVE_MAX_ROWS; prefill: per-row top-k).
   - Sparse MLA (MLAModules is_sparse=True): absorbed q x W_UK_T, attention
     over the selected 576-number rows only, x W_UV. Prefill takes the same
     absorbed path for long prompts. vLLM falls back to dense MHA when a
     prompt fits in 2,048 tokens (sparse_mla_attention.py
     _use_dense_mha_prefill) and, at TP8 with FlashMLA sparse, to a masked
     expanded MHA for sequences up to 32K when enough queries arrive at once
     (mla_attention.py _MASKED_MHA_THRESHOLDS). The figure shows the long-
     prompt path.
   - g_proj (x -> 12,288) is its own GEMM here, on a side stream for small
     batches (the Gate proj. card, `gate` lane); then o * sigmoid(g) (the
     Output gate card) and o_proj.
   Numbers cover all 96 heads and the whole indexer of one layer (DP
   attention: one GPU runs them all). Latent cache in vLLM's fp8_ds_mla
   layout, as in the post's memory figures: 656 B = c_kv 512 FP8 + 16 B of
   scales + k_s 64 BF16 (128 B); vLLM picks it because its plain FP8 sparse
   MLA path does not run 96 heads. Index key 128 FP8 + a 4-byte scale =
   132 B (kimi_k3/common/dsa_cache.py indexer_cache_entry_bytes).
   dtypes: the quantization_config (compressed-tensors MXFP4) ignores
   "re:.*self_attn.*", so every attention weight, the indexer's included,
   stays BF16; activations BF16. Operands (flows.js contract): decode L counts
   this token too; k = min(L, 2,048) selected rows; index scores and softmax
   weights are counted on chip. */

window.ATTN_BLOG_FLOWS = window.ATTN_BLOG_FLOWS || {};
(() => {
  const { axis, tokens, operand, rows, rowCount, cacheRows, keyAxis, causalPairs, cappedPairs, isDecode } = window.ATTN_FLOW_KIT;
  const HIDDEN = 7168;
  const HEADS = 96;
  const Q_LATENT = 1536;
  const KV_LATENT = 512;
  const D_U = 128;
  const D_S = 64;
  const D_V = 128;
  const GATE = HEADS * D_V; // 12,288
  const INDEX_HEADS = 32;
  const INDEX_DIM = 128;
  const TOP_K = 2048;
  const hidden = axis("D", HIDDEN);
  const heads = axis("H", HEADS);
  const qLatent = axis("r_q", Q_LATENT);
  const kvLatent = axis("r_kv", KV_LATENT);
  const dU = axis("d_u", D_U);
  const dS = axis("d_s", D_S);
  const dV = axis("d_v", D_V);
  const indexHeads = axis("H_I", INDEX_HEADS);
  const indexDim = axis("d_I", INDEX_DIM);
  const gateWidth = axis("H·d_v", GATE);
  // one FP32 scale per 128 FP8 numbers: k_I's 4 B, and c_kv's 16 B per row in fp8_ds_mla
  const fp8Scale = { every: 128, bytes: 4, note: "one FP32 scale per 128 FP8 numbers, counted in the bytes (k_I 128 + 4 B; c_kv 512 + 16 B)." };
  // keys one query may select from, and how many it keeps
  const selected = (scale) => tokens("k", Math.min(TOP_K, isDecode(scale) ? scale.context : scale.context + 1));
  // the rows a sparse step reads: decode, each request's k selected rows;
  // prefill, at most every row of the request once (each query picks its own k)
  const sparseRows = (scale) => (isDecode(scale) ? [tokens("B", scale.batch), selected(scale)] : cacheRows(scale));
  const sparsePairs = (scale) => cappedPairs(scale, TOP_K);
  const scores = (scale, name) => operand(name, [...rows(scale), heads, selected(scale)], "FP32", { onChip: true });
  const indices = (scale) => operand("idx", [...rows(scale), selected(scale)], "INT32");

  window.ATTN_BLOG_FLOWS.dsa = {
    title: "Gated DSA in one layer of Kimi K3.1 Pro",
    axes: [
      ["B", "requests"],
      ["N", "tokens in the chunk"],
      ["L", { decode: "tokens in each request's cache, the new one included", prefill: "earlier tokens in the cache" }],
      ["k", "selected rows: min(L, 2,048) per query"],
      ["H", "heads"],
      ["H_I", "index heads"],
      ["d_I", "index head dim"],
      ["D", "hidden size"],
      ["r_q", "query latent"],
      ["r_kv", "KV latent c_kv"],
      ["d_u", "query part that meets the rebuilt key"],
      ["d_s", "shared key part k_s / q_s"],
      ["d_v", "value head dim"],
    ],
    input: { label: "x", note: { prefill: "N × 7,168", decode: "B × 7,168" } },
    output: { label: "out", note: "to the MoE FFN" },
    // The indexer runs beside the attention query path (vLLM dsa.py: the
    // indexer scores and selects from q_I, w and the index-key cache while
    // q_u goes through W_UK); only the selected positions join, at Sparse scores.
    lanes: {
      // the gate projection reads only x: a lane above the main row, joining the Output gate
      gate: { side: "above", from: { step: "input", label: "x" }, to: [{ step: "gate", label: "g" }] },
      indexer: { side: "below", from: { step: "query", label: "q_I" }, to: [{ step: "score", label: "2,048 selected" }] },
    },
    // w comes out of the Down-proj GEMM (packed weights_proj) and skips to the
    // index scores on an edge of its own, under the write card's row; it hops
    // the write card's drop into the cache, the one crossing: the Down-proj
    // sits left of the norms → write → cache chain, the index scores right of it.
    // q_s skips Absorb W_UK, from Q up-proj. to the scores. g_proj is its own
    // GEMM over x (dsa.py forward): the `gate` lane.
    edges: [
      { from: "down", to: "index", label: "w", level: "underBelow", dy: 16, fromDx: -40, enter: "below", dx: -40, labelDx: -60 },
      { from: "query", to: "score", label: "q_s", level: "over", dy: 16 },
    ],
    steps: [
      // the gate's own GEMM: it reads only x, so it sits in a lane of its own
      // above the main row and joins the Output gate (vLLM may run it on a
      // side stream beside the attention for small batches)
      {
        key: "gateProj",
        title: "Gate proj.",
        fullTitle: "Gate projection",
        role: "weights",
        lane: "gate",
        formula: "g = x · W_G",
        does: "Projects each token's hidden state into 12,288 gate values g, one for each number of the attention output.",
        counted: "2 per weight per token, 2 × 7,168 × 12,288 = 176 MFLOP a token.",
        operands: (scale) => ({
          input: [operand("x", [...rows(scale), hidden], "BF16")],
          weight: [operand("W_G", [hidden, gateWidth], "BF16")],
          output: [operand("g", [...rows(scale), gateWidth], "BF16")],
          flops: 2 * rowCount(scale) * HIDDEN * GATE,
          summed: ["D"],
        }),
      },
      {
        key: "down",
        title: "Down-proj.",
        fullTitle: "Down-projection",
        role: "weights",
        formula: "x · [W_DQ, W_DKV, W_KS, W_IK, W_IW]",
        does: "Gives the MLA inputs c_q, c_kv and k_s, as in Kimi K3. The same GEMM also gives the indexer's key k_I and one weight per index head, w.",
        counted: "2 per weight per token, 2 × 7,168 × 2,272 = 32.6 MFLOP a token (one fused GEMM).",
        operands: (scale) => ({
          input: [operand("x", [...rows(scale), hidden], "BF16")],
          weight: [
            operand("W_DQ", [hidden, qLatent], "BF16"),
            operand("W_DKV", [hidden, kvLatent], "BF16"),
            operand("W_KS", [hidden, dS], "BF16"),
            operand("W_IK", [hidden, indexDim], "BF16"),
            operand("W_IW", [hidden, indexHeads], "BF16"),
          ],
          output: [
            operand("c_q", [...rows(scale), qLatent], "BF16"),
            operand("c_kv", [...rows(scale), kvLatent], "BF16"),
            operand("k_s", [...rows(scale), dS], "BF16"),
            operand("k_I", [...rows(scale), indexDim], "BF16"),
            operand("w", [...rows(scale), indexHeads], "BF16"),
          ],
          flops: 2 * rowCount(scale) * HIDDEN * (Q_LATENT + KV_LATENT + D_S + INDEX_DIM + INDEX_HEADS),
          summed: ["D"],
        }),
      },
      {
        key: "norm",
        title: "Norms",
        role: "elementwise",
        formula: "RMSNorm(c_q), RMSNorm(c_kv), LayerNorm(k_I)",
        does: "Rescales c_q, c_kv and k_I, each with its own learned weights. k_s is not normalized.",
        counted: "about 4 per number for an RMSNorm, 5 for the LayerNorm.",
        operands: (scale) => ({
          input: [
            operand("c_q", [...rows(scale), qLatent], "BF16"),
            operand("c_kv", [...rows(scale), kvLatent], "BF16"),
            operand("k_I", [...rows(scale), indexDim], "BF16"),
          ],
          weight: [operand("γ_q", [qLatent], "BF16"), operand("γ_kv", [kvLatent], "BF16"), operand("γ_I, β_I", [axis("2·d_I", 2 * INDEX_DIM)], "BF16")],
          output: [
            operand("c_q", [...rows(scale), qLatent], "BF16"),
            operand("c_kv", [...rows(scale), kvLatent], "BF16"),
            operand("k_I", [...rows(scale), indexDim], "BF16"),
          ],
          flops: rowCount(scale) * (4 * (Q_LATENT + KV_LATENT) + 5 * INDEX_DIM),
          summed: ["r_q", "r_kv", "d_I"],
        }),
      },
      {
        key: "query",
        title: "Q up-proj.",
        fullTitle: "Query up-projection",
        role: "weights",
        edgeIn: "c_q",
        formula: "c_q · [W_UQ, W_IQ]",
        does: "Expands c_q into the attention query, q_u and q_s for each of the 96 heads, and into the indexer's query q_I for each of its 32 heads. Nothing is rotated by position.",
        counted: "2 per weight per token, 2 × 1,536 × 22,528 = 69.2 MFLOP a token.",
        operands: (scale) => ({
          input: [operand("c_q", [...rows(scale), qLatent], "BF16")],
          weight: [operand("W_UQ", [qLatent, axis("H·(d_u+d_s)", HEADS * (D_U + D_S))], "BF16"), operand("W_IQ", [qLatent, axis("H_I·d_I", INDEX_HEADS * INDEX_DIM)], "BF16")],
          output: [
            operand("q_u", [...rows(scale), heads, dU], "BF16"),
            operand("q_s", [...rows(scale), heads, dS], "BF16"),
            operand("q_I", [...rows(scale), indexHeads, indexDim], "BF16"),
          ],
          flops: 2 * rowCount(scale) * Q_LATENT * (HEADS * (D_U + D_S) + INDEX_HEADS * INDEX_DIM),
          summed: ["r_q"],
        }),
      },
      {
        key: "index",
        title: "Index scores",
        role: "core",
        lane: "indexer",
        formula: "Σₕ wₕ · ReLU(q_I,ₕ · k_I)",
        does: {
          prefill: "Scores every key each query in the chunk may see: the L earlier tokens and the chunk's tokens up to its own position. Each of the 32 index heads scores the key, and w, one weight per index head that the Down-proj. computes from x, sums them into one score. The query is cast to FP8, like the keys.",
          decode: "Scores every earlier token with the 32 small index heads in FP8. w, one weight per index head that the Down-proj. computes from x, sums the heads' scores into one number per token.",
        },
        counted: {
          prefill: "2 × 32 × 128 = 8.2 KFLOP per query–key pair, over the N × L + N(N+1)/2 causal pairs (the head weighting is not counted). The scores stay on chip.",
          decode: "2 × 32 × 128 = 8.2 KFLOP per query–key pair, B × L pairs (the head weighting is not counted). The scores stay on chip.",
        },
        operands: (scale) => ({
          input: [operand("q_I", [...rows(scale), indexHeads, indexDim], "BF16"), operand("w", [...rows(scale), indexHeads], "BF16")],
          cache: [operand("k_I", [...cacheRows(scale), indexDim], "FP8", { scale: fp8Scale })],
          output: [operand("I", [...rows(scale), keyAxis(scale)], "FP32", { onChip: true })],
          flops: 2 * INDEX_HEADS * INDEX_DIM * causalPairs(scale),
          summed: ["d_I", "H_I"],
        }),
      },
      {
        key: "topk",
        title: "Top-k",
        role: "elementwise",
        lane: "indexer",
        edgeIn: "scores",
        formula: "top 2,048",
        does: {
          prefill: "For each query, keeps the positions of its 2,048 highest-scoring keys. A query that sees 2,048 keys or fewer keeps them all.",
          decode: "Keeps the positions of the 2,048 highest-scoring earlier tokens. With 2,048 or fewer earlier tokens, it keeps them all.",
        },
        counted: "a selection, not arithmetic: about 1 comparison per score, counted as 1 FLOP. The scores come on chip from the index kernel; only the positions go to HBM.",
        operands: (scale) => ({
          input: [operand("I", [...rows(scale), keyAxis(scale)], "FP32", { onChip: true })],
          output: [indices(scale)],
          flops: causalPairs(scale),
          summed: [],
        }),
      },
      {
        key: "absorbKey",
        title: "Absorb W_UK",
        role: "weights",
        edgeIn: "q_u",
        formula: "q̃ = q_u · W_UKᵀ, head by head",
        does: {
          prefill: "Moves each head's key matrix onto its query, for every query in the chunk, so attention runs over the cached latents as in decode.",
          decode: "Moves each head's key matrix onto its query. q̃ · c_kv gives the same score as q_u · (c_kv · W_UK), so no key is ever built.",
        },
        counted: "one batched GEMM, a 128 × 512 matrix per head: 2 × 96 × 128 × 512 = 12.6 MFLOP a token.",
        operands: (scale) => ({
          input: [operand("q_u", [...rows(scale), heads, dU], "BF16")],
          weight: [operand("W_UK", [heads, dU, kvLatent], "BF16")],
          output: [operand("q̃", [...rows(scale), heads, kvLatent], "BF16")],
          flops: 2 * rowCount(scale) * HEADS * D_U * KV_LATENT,
          summed: ["d_u"],
        }),
      },
      {
        key: "score",
        title: "Sparse scores",
        role: "core",
        edgeIn: "q̃",
        formula: "[q̃, q_s] · [c_kv, k_s]ᵀ ÷ √192, selected rows only",
        does: {
          prefill: "Compares each query, head by head, with the c_kv and k_s of its own 2,048 selected tokens only.",
          decode: "Compares each head's q̃ and q_s with the c_kv and k_s of the 2,048 selected tokens only. All 96 heads read the same rows.",
        },
        counted: {
          prefill: "2 × 96 × 576 = 111 KFLOP per query–key pair; the query at position j keeps min(j, 2,048) keys. Each query gathers its own rows; read once each, the chunk's gathers touch at most the L + N rows of the request, counted as all of them.",
          decode: "2 × 96 × 576 = 111 KFLOP per query–key pair, B × k pairs. The k rows are gathered from the request's L rows by idx; all H heads share them.",
        },
        operands: (scale) => ({
          input: [operand("q̃", [...rows(scale), heads, kvLatent], "BF16"), operand("q_s", [...rows(scale), heads, dS], "BF16"), indices(scale)],
          cache: [operand("c_kv", [...sparseRows(scale), kvLatent], "FP8", { scale: fp8Scale }), operand("k_s", [...sparseRows(scale), dS], "BF16")],
          output: [scores(scale, "S")],
          flops: 2 * HEADS * (KV_LATENT + D_S) * sparsePairs(scale),
          summed: ["r_kv", "d_s"],
        }),
      },
      {
        key: "softmax",
        title: "Softmax",
        role: "elementwise",
        formula: "exp, ÷ row sum",
        does: "Turns each row of scores into weights that add up to 1.",
        counted: "about 5 per score (max, subtract, exp, add, divide).",
        operands: (scale) => ({
          input: [scores(scale, "S")],
          output: [scores(scale, "P")],
          flops: 5 * HEADS * sparsePairs(scale),
          summed: ["k"],
        }),
      },
      {
        key: "sum",
        title: "Weighted sum",
        role: "core",
        formula: "weights · c_kv, selected rows only",
        does: "Adds up the selected tokens' c_kv, each times its weight. Each head gets its own mix of latents.",
        counted: "2 × 96 × 512 = 98.3 KFLOP per query–key pair, as many pairs as the sparse scores.",
        operands: (scale) => ({
          input: [scores(scale, "P")],
          cache: [operand("c_kv", [...sparseRows(scale), kvLatent], "FP8", { reused: true })],
          output: [operand("o_c", [...rows(scale), heads, kvLatent], "BF16")],
          flops: 2 * HEADS * KV_LATENT * sparsePairs(scale),
          summed: ["k"],
        }),
      },
      {
        key: "absorbValue",
        title: "Absorb W_UV",
        role: "weights",
        edgeIn: "o_c",
        formula: "o = o_c · W_UV, head by head",
        does: "Applies each head's value matrix after the sum, once for each query.",
        counted: "one batched GEMM, a 512 × 128 matrix per head: 2 × 96 × 512 × 128 = 12.6 MFLOP a token.",
        operands: (scale) => ({
          input: [operand("o_c", [...rows(scale), heads, kvLatent], "BF16")],
          weight: [operand("W_UV", [heads, kvLatent, dV], "BF16")],
          output: [operand("o", [...rows(scale), heads, dV], "BF16")],
          flops: 2 * rowCount(scale) * HEADS * KV_LATENT * D_V,
          summed: ["r_kv"],
        }),
      },
      {
        key: "gate",
        title: "Output gate",
        role: "elementwise",
        edgeIn: "o",
        formula: "o × sigmoid(g)",
        does: "Squashes each of the 12,288 gate values g into (0, 1) and scales the matching number of o by it.",
        counted: "about 5 per number (sigmoid and multiply).",
        operands: (scale) => ({
          input: [operand("o", [...rows(scale), gateWidth], "BF16"), operand("g", [...rows(scale), gateWidth], "BF16")],
          output: [operand("o", [...rows(scale), gateWidth], "BF16")],
          flops: 5 * rowCount(scale) * GATE,
          summed: [],
        }),
      },
      {
        key: "out",
        title: "Output proj.",
        fullTitle: "Output projection",
        role: "weights",
        formula: "concat(o) · W_O",
        does: "Joins the 96 gated heads into 12,288 numbers and maps them back to 7,168.",
        counted: "2 per weight per token, 2 × 12,288 × 7,168 = 176 MFLOP a token.",
        operands: (scale) => ({
          input: [operand("o", [...rows(scale), gateWidth], "BF16")],
          weight: [operand("W_O", [gateWidth, hidden], "BF16")],
          output: [operand("out", [...rows(scale), hidden], "BF16")],
          flops: 2 * rowCount(scale) * GATE * HIDDEN,
          summed: ["H·d_v"],
        }),
      },
    ],
    branch: {
      title: "Write caches",
      role: "write",
      formula: "quantize, append to both caches",
      does: {
        prefill: "Stores the chunk's k_I in FP8 with one scale per token in the index-key cache, and its c_kv and k_s in the latent cache.",
        decode: "Stores this token's k_I in FP8 with one scale in the index-key cache, and its c_kv and k_s in the latent cache.",
      },
      counted: "about 1 per FP8 number (one multiply by its scale); k_s is copied as it is.",
      operands: (scale) => ({
        input: [
          operand("k_I", [...rows(scale), indexDim], "BF16"),
          operand("c_kv", [...rows(scale), kvLatent], "BF16"),
          operand("k_s", [...rows(scale), dS], "BF16"),
        ],
        output: [
          operand("k_I", [...rows(scale), indexDim], "FP8", { scale: fp8Scale }),
          operand("c_kv", [...rows(scale), kvLatent], "FP8", { scale: fp8Scale }),
          operand("k_s", [...rows(scale), dS], "BF16"),
        ],
        flops: rowCount(scale) * (INDEX_DIM + KV_LATENT),
        summed: [],
      }),
      // k_s is not normalized: it comes straight from the Down-proj GEMM
      from: [
        { step: "norm", label: "k_I, c_kv" },
        { step: "down", label: "k_s, not normalized" },
      ],
    },
    cache: {
      title: "Two caches, kept for the whole request",
      rows: ["k_I", "c_kv, k_s"],
      labelWidth: 86,
      newLabel: { prefill: "chunk", decode: "new" },
      // decode: L counts this token too, so the earlier columns end at t(L−2)
      cells: { prefill: undefined, decode: ["t0", "t1", "t2", "…", "t(L−2)"] },
      groups: { old: { prefill: "L earlier tokens", decode: "L − 1 earlier tokens" }, new: { prefill: "N in this chunk", decode: "this token" } },
      note: "k_I: 128 FP8 + 4 B scale = 132 B; c_kv 512 FP8 + 16 B scales + k_s 64 BF16 = 656 B; 788 B per token and layer, 18.5 KiB over 24 layers",
      reads: [
        { step: "index", row: 0, label: { prefill: "reads k_I: L earlier + N in chunk", decode: "reads every k_I: all L tokens" } },
        { step: "score", row: 1, label: "reads only the 2,048 selected rows" },
        { step: "sum", row: 1, label: "same rows, as the values" },
      ],
    },
  };
})();
