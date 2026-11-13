/* Flow figure for the MLA subsection (`<!-- interactive: flow_mla -->`):
   one dense Gated MLA layer of Kimi K3. Layout contract: see flows.js.

   Kimi K3 is the dense predecessor of Kimi K3.1 Pro. No separate K3 config
   is available, so this figure ASSUMES that K3's MLA layer has K3.1 Pro's
   dimensions without the DSA indexer (main/model/config/kimi_k31_pro.json):
   hidden 7,168, 96 heads, q_lora_rank 1,536, kv_lora_rank 512,
   qk_nope_head_dim 128, qk_rope_head_dim 64, v_head_dim 128, 24 full layers,
   mla_use_nope and mla_use_output_gate both true. vLLM serves K3 and K3.1
   with the same code (vllm/models/kimi_k3/nvidia/).

   What vLLM runs (kimi_k3/nvidia/mla.py, MultiHeadLatentAttention):
   - One input GEMM `fused_qkv_a_g_proj` gives c_q (1,536), c_kv (512), the
     64-number k_pe and the gate g (96 x 128); below 512 tokens
     (_GATE_MULTI_STREAM_TOKEN_THRESHOLD) the gate rows run as their own GEMM
     on a side stream. We draw x · W_G as its own Gate proj. card (the
     `gate` lane) and the Output gate as o × sigmoid(g), as flow_dsa
     does for K3.1 Pro (whose dsa.py always has a separate g_proj), so the two
     figures show the same gate in the same place; the blog calls a DSA layer
     this MLA layer with an indexer added.
   - mla_use_nope: the model asserts NoPE ("Kimi-K3 MLA is NoPE-only",
     model.py) and builds no rotary_emb, so k_pe and the query's last 64
     numbers per head are plain projections with no rotation. vLLM still
     caches k_pe next to c_kv (head_size = 512 + 64 = 576). We call it k_s,
     the shared key, and the query's 64 numbers q_s; q_u is the 128-number
     part that meets the key W_UK rebuilds.
   - Softmax scale 1/sqrt(128 + 64).
   - Prefill: kv_b_proj expands the new tokens' latents to k_nope and v,
     K = [k_nope | k_s]; earlier tokens are gathered from the cache in chunks,
     expanded the same way, and their partial results merged by LSE.
   - Decode: q_u x W_UK_T (absorb), MQA over the 576-number cache rows,
     then x W_UV.
   - Output: attn_out * sigmoid(g), then o_proj.
   Numbers cover all 96 heads (whole model, before tensor parallelism), like
   the GQA figure. The cache is FP8 as everywhere in the post: 576 B per
   token and layer (vLLM's plain FP8 layout; its fp8_ds_mla layout, used by
   the sparse K3.1 layers, takes 656 B).
   dtypes: K3.1 Pro's quantization_config (compressed-tensors MXFP4) ignores
   "re:.*self_attn.*", so every attention weight stays BF16; activations BF16,
   cache FP8. Operands (flows.js contract): decode L counts this token too. */

(() => {
  const { axis, tokens, operand, rows, rowCount, cacheRows, keyAxis, causalPairs, isDecode } = window.ATTN_FLOW_KIT;
  const HIDDEN = 7168;
  const HEADS = 96;
  const Q_LATENT = 1536;
  const KV_LATENT = 512;
  const D_U = 128; // the part of a query head that meets the key W_UK rebuilds
  const D_S = 64; // the shared key part, k_s / q_s
  const D_V = 128;
  const GATE = HEADS * D_V; // 12,288
  const hidden = axis("D", HIDDEN);
  const heads = axis("H", HEADS);
  const qLatent = axis("r_q", Q_LATENT);
  const kvLatent = axis("r_kv", KV_LATENT);
  const dU = axis("d_u", D_U);
  const dS = axis("d_s", D_S);
  const dV = axis("d_v", D_V);
  const dQK = axis("d_u+d_s", D_U + D_S);
  const row576 = axis("r_kv+d_s", KV_LATENT + D_S);
  const gateWidth = axis("H·d_v", GATE);
  // scores and softmax weights stay on chip inside the fused attention kernel
  const scores = (scale, name) => operand(name, [...rows(scale), heads, keyAxis(scale)], "FP32", { onChip: true });
  // the latent cache row: c_kv and k_s, one row per token, shared by every head
  const latentCache = (scale) => operand("c_kv, k_s", [...cacheRows(scale), row576], "FP8");

  window.ATTN_BLOG_FLOWS.mla = {
    title: "Gated MLA in one layer of Kimi K3",
    axes: [
      ["B", "requests"],
      ["N", "tokens in the chunk"],
      ["L", { decode: "tokens in each request's cache, the new one included", prefill: "earlier tokens in the cache" }],
      ["H", "heads"],
      ["D", "hidden size"],
      ["r_q", "query latent"],
      ["r_kv", "KV latent c_kv"],
      ["d_u", "query part that meets the rebuilt key"],
      ["d_s", "shared key part k_s / q_s"],
      ["d_v", "value head dim"],
    ],
    input: { label: "x", note: { prefill: "N × 7,168", decode: "B × 7,168" } },
    output: { label: "out", note: "to the MoE FFN" },
    // Every tensor on its own edge (flows.js `edges`): q_s (and in prefill all
    // of q) skips the card between Q up-proj. and the scores. The gate's GEMM
    // is the `gate` lane, as in flow_dsa.
    edges: [
      // decode: q_s skips Absorb W_UK (prefill: all of q reaches the scores on the main arrow)
      { from: "query", to: "score", only: "decode", label: "q_s", level: "over", dy: 16 },
      // prefill: the expanded V goes to the weighted sum under the write card's row, right of the cache read into Expand
      { from: "expand", to: "sum", only: "prefill", label: "V", level: "underBelow", dy: 14, fromDx: 40, enter: "below", dx: -40 },
    ],
    // Prefill: Expand K, V reads only the cache, so it sits off the main row
    // in a lane of its own with no fork (`from.hidden`; `from` only places it
    // after Q up-proj.) and hands K up into the scores.
    lanes: {
      // the gate projection reads only x: a lane above the main row, joining the Output gate
      gate: { side: "above", from: { step: "input", label: "x" }, to: [{ step: "gate", label: "g" }] },
      kv: { side: "below", from: { step: "query", hidden: true }, to: [{ step: "score", label: "K" }] },
    },
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
        formula: "x · [W_DQ, W_DKV, W_KS]",
        does: "Compresses each token's hidden state into c_q for the query and the latent c_kv for the key and the value. It also gives k_s, a 64-number key part that all heads share.",
        counted: "2 per weight per token, 2 × 7,168 × 2,112 = 30.3 MFLOP a token (one fused GEMM).",
        operands: (scale) => ({
          input: [operand("x", [...rows(scale), hidden], "BF16")],
          weight: [
            operand("W_DQ", [hidden, qLatent], "BF16"),
            operand("W_DKV", [hidden, kvLatent], "BF16"),
            operand("W_KS", [hidden, dS], "BF16"),
          ],
          output: [
            operand("c_q", [...rows(scale), qLatent], "BF16"),
            operand("c_kv", [...rows(scale), kvLatent], "BF16"),
            operand("k_s", [...rows(scale), dS], "BF16"),
          ],
          flops: 2 * rowCount(scale) * HIDDEN * (Q_LATENT + KV_LATENT + D_S),
          summed: ["D"],
        }),
      },
      {
        key: "norm",
        title: "RMSNorm",
        role: "elementwise",
        formula: "c_q, c_kv ÷ rms × γ",
        does: "Rescales c_q and c_kv, each with its own learned weights. k_s is not normalized.",
        counted: "about 4 per number (square and add for the rms, then two multiplies).",
        operands: (scale) => ({
          input: [operand("c_q", [...rows(scale), qLatent], "BF16"), operand("c_kv", [...rows(scale), kvLatent], "BF16")],
          weight: [operand("γ_q", [qLatent], "BF16"), operand("γ_kv", [kvLatent], "BF16")],
          output: [operand("c_q", [...rows(scale), qLatent], "BF16"), operand("c_kv", [...rows(scale), kvLatent], "BF16")],
          flops: 4 * rowCount(scale) * (Q_LATENT + KV_LATENT),
          summed: ["r_q", "r_kv"],
        }),
      },
      {
        key: "query",
        title: "Q up-proj.",
        fullTitle: "Query up-projection",
        role: "weights",
        edgeIn: "c_q",
        formula: "c_q · W_UQ",
        does: "Expands c_q into a 192-number query for each of the 96 heads: q_u (128) meets the key rebuilt from c_kv, and q_s (64) meets k_s. Kimi K3 rotates neither part by position.",
        counted: "2 per weight per token, 2 × 1,536 × 18,432 = 56.6 MFLOP a token.",
        operands: (scale) => ({
          input: [operand("c_q", [...rows(scale), qLatent], "BF16")],
          weight: [operand("W_UQ", [qLatent, axis("H·(d_u+d_s)", HEADS * (D_U + D_S))], "BF16")],
          output: [operand("q_u", [...rows(scale), heads, dU], "BF16"), operand("q_s", [...rows(scale), heads, dS], "BF16")],
          flops: 2 * rowCount(scale) * Q_LATENT * HEADS * (D_U + D_S),
          summed: ["r_q"],
        }),
      },
      {
        key: "expand",
        title: "Expand K, V",
        role: "weights",
        only: "prefill",
        lane: "kv",
        formula: "K = [c_kv · W_UK, k_s],  V = c_kv · W_UV",
        does: "Rebuilds every head's key and value from the latent, for the L earlier tokens and the N tokens in the chunk. Each key ends with the shared k_s. The engine expands the earlier tokens a slice at a time, and drops each slice after use.",
        counted: "2 per weight per latent, 2 × 512 × 24,576 = 25.2 MFLOP for each of the L + N latents. K and V go to HBM in BF16 for the attention kernel.",
        operands: (scale) => ({
          cache: [latentCache(scale)],
          weight: [operand("W_UK", [kvLatent, axis("H·d_u", HEADS * D_U)], "BF16"), operand("W_UV", [kvLatent, axis("H·d_v", HEADS * D_V)], "BF16")],
          output: [operand("K", [...cacheRows(scale), heads, dQK], "BF16"), operand("V", [...cacheRows(scale), heads, dV], "BF16")],
          flops: 2 * (scale.context + scale.chunk) * KV_LATENT * HEADS * (D_U + D_V),
          summed: ["r_kv"],
        }),
      },
      {
        key: "absorbKey",
        title: "Absorb W_UK",
        role: "weights",
        only: "decode",
        edgeIn: "q_u",
        formula: "q̃ = q_u · W_UKᵀ, head by head",
        does: "Moves each head's key matrix onto its query. q̃ · c_kv gives the same score as q_u · (c_kv · W_UK), so no key is ever built.",
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
        title: "Scores",
        role: "core",
        edgeIn: { prefill: "q_u, q_s", decode: "q̃" },
        formula: { prefill: "Q · Kᵀ ÷ √192", decode: "[q̃, q_s] · [c_kv, k_s]ᵀ ÷ √192" },
        does: {
          prefill: "Compares every query in the chunk with the expanded keys of the L earlier tokens and of the chunk's own tokens, head by head.",
          decode: "Compares each head's q̃ with every earlier token's c_kv, and its q_s with that token's k_s. All 96 heads read the same cache row.",
        },
        counted: {
          prefill: "2 × 96 × 192 = 36.9 KFLOP per query–key pair, over the N × L + N(N+1)/2 pairs the causal mask keeps. K comes from Expand, not the cache.",
          decode: "2 × 96 × 576 = 111 KFLOP per query–key pair, B × L pairs. The cache row has no head axis: all H heads read the same row, so it is loaded once for all of them.",
        },
        operands: (scale) =>
          isDecode(scale)
            ? {
                input: [operand("q̃", [...rows(scale), heads, kvLatent], "BF16"), operand("q_s", [...rows(scale), heads, dS], "BF16")],
                cache: [latentCache(scale)],
                output: [scores(scale, "S")],
                flops: 2 * HEADS * (KV_LATENT + D_S) * causalPairs(scale),
                summed: ["r_kv", "d_s", "r_kv+d_s"],
              }
            : {
                input: [
                  operand("q_u, q_s", [...rows(scale), heads, dQK], "BF16"),
                  operand("K", [...cacheRows(scale), heads, dQK], "BF16"),
                ],
                output: [scores(scale, "S")],
                flops: 2 * HEADS * (D_U + D_S) * causalPairs(scale),
                summed: ["d_u+d_s"],
              },
      },
      {
        key: "mask",
        title: "Causal mask",
        role: "elementwise",
        only: "prefill",
        formula: "score = −∞ where key > query",
        does: "Hides the tokens that come later in the chunk, so each token sees only itself and earlier ones.",
        counted: "none in practice: the kernel skips the hidden pairs instead of computing them.",
        operands: (scale) => ({ input: [scores(scale, "S")], output: [scores(scale, "S")], flops: 0, summed: [] }),
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
          flops: 5 * HEADS * causalPairs(scale),
          summed: [keyAxis(scale).label],
        }),
      },
      {
        key: "sum",
        title: "Weighted sum",
        role: "core",
        formula: { prefill: "weights · V", decode: "weights · c_kv" },
        does: {
          prefill: "Adds up the expanded values of the tokens each query may see, each times its weight. The engine merges the partial sums of the earlier-token slices and of the chunk.",
          decode: "Adds up the earlier tokens' c_kv, each times its weight. Each head gets its own mix of latents.",
        },
        counted: {
          prefill: "2 × 96 × 128 = 24.6 KFLOP per query–key pair, as many pairs as the scores. V comes from Expand, not the cache.",
          decode: "2 × 96 × 512 = 98.3 KFLOP per query–key pair, B × L pairs.",
        },
        operands: (scale) =>
          isDecode(scale)
            ? {
                input: [scores(scale, "P")],
                cache: [operand("c_kv", [...cacheRows(scale), kvLatent], "FP8", { reused: true })],
                output: [operand("o_c", [...rows(scale), heads, kvLatent], "BF16")],
                flops: 2 * HEADS * KV_LATENT * causalPairs(scale),
                summed: ["L"],
              }
            : {
                input: [scores(scale, "P"), operand("V", [...cacheRows(scale), heads, dV], "BF16")],
                output: [operand("o", [...rows(scale), heads, dV], "BF16")],
                flops: 2 * HEADS * D_V * causalPairs(scale),
                summed: ["L+N"],
              },
      },
      {
        key: "absorbValue",
        title: "Absorb W_UV",
        role: "weights",
        only: "decode",
        edgeIn: "o_c",
        formula: "o = o_c · W_UV, head by head",
        does: "Applies each head's value matrix after the sum: once for the new token, instead of once for every earlier token.",
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
      title: "Write c_kv, k_s",
      role: "write",
      formula: "quantize, append",
      does: { prefill: "Stores the chunk's latents and shared keys in FP8, so later chunks and tokens can read them.", decode: "Stores this token's c_kv and k_s in FP8, so every later token can read them." },
      counted: "about 1 per number (one multiply by the FP8 scale).",
      operands: (scale) => ({
        input: [operand("c_kv", [...rows(scale), kvLatent], "BF16"), operand("k_s", [...rows(scale), dS], "BF16")],
        output: [operand("c_kv, k_s", [...rows(scale), row576], "FP8")],
        flops: rowCount(scale) * (KV_LATENT + D_S),
        summed: [],
      }),
      from: [
        { step: "norm", label: "c_kv" },
        { step: "down", label: "k_s, not normalized" },
      ],
    },
    cache: {
      title: "Latent cache, kept for the whole request",
      rows: ["c_kv, k_s"],
      labelWidth: 86,
      newLabel: { prefill: "chunk", decode: "new" },
      // decode: L counts this token too, so the earlier columns end at t(L−2)
      cells: { prefill: undefined, decode: ["t0", "t1", "t2", "…", "t(L−2)"] },
      groups: { old: { prefill: "L earlier tokens", decode: "L − 1 earlier tokens" }, new: { prefill: "N in this chunk", decode: "this token" } },
      note: "c_kv (512 numbers) and k_s (64), read by all heads: 576 B per token and layer in FP8; 13.5 KiB over 24 layers",
      reads: [
        { step: "expand", row: 0, only: "prefill", label: "reads c_kv, k_s: L earlier + N in chunk" },
        { step: "score", row: 0, only: "decode", label: "reads c_kv, k_s: all L tokens" },
        { step: "sum", row: 0, only: "decode", label: "same c_kv, as the values" },
      ],
    },
  };
})();
