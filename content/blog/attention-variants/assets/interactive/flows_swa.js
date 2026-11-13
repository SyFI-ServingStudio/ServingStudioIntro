/* Sliding-window flow for the "Attention Variants" section; follows the
   layout contract in flows.js's header comment.

   MiMo V2.6 Flash's GQA window layer, from main/model/config/
   mimo_v2_6_flash_rl.json: hidden 4,096; swa_num_attention_heads 64,
   swa_num_key_value_heads 8; swa_head_dim 192 for Q and K but swa_v_head_dim
   128 for V; sliding_window 128 (keys a query sees, itself included);
   partial_rotary_factor 0.334, so RoPE turns int(192 × 0.334) = 64 channels;
   learned sink logits on window layers (add_swa_attention_sink_bias); V scaled
   by attention_value_scale. The launch order follows ServingStudio Sim's
   simulator/src/worklet/mimo_v2_attn_local.rs (fused QKV GEMM, RoPE + V
   scale, KV insert, FA4 with window and sinks, BF16 o_proj). The cache is
   counted in FP8 like the rest of the post (vLLM keeps it in BF16).
   Costs cover the whole layer, as if it ran on one GPU, like the GQA figure.
   dtypes, from that config's quantization_config: quant_method fp8 (e4m3),
   weight_block_size [128, 128], o_proj in ignored_layers. So W_qkv is FP8 with
   one FP32 scale per 128 × 128 block, W_o BF16; the GEMM's input x is cast to
   FP8 with one UE8M0 scale per 128 numbers (the worklet's step 1). Other
   activations BF16; the cache FP8 like the rest of the post. Operands
   (flows.js contract): decode L counts this token too; W = 128 keys at most. */

(() => {
  const { axis, tokens, operand, rows, rowCount, cappedPairs, isDecode } = window.ATTN_FLOW_KIT;
  const HIDDEN = 4096;
  const Q_HEADS = 64;
  const KV_HEADS = 8;
  const D_QK = 192;
  const D_V = 128;
  const WINDOW = 128; // keys a query sees, itself included
  const ROPE = 64;
  const hidden = axis("D", HIDDEN);
  const qHeads = axis("H_q", Q_HEADS);
  const kvHeads = axis("H_kv", KV_HEADS);
  const dQK = axis("d_qk", D_QK);
  const dV = axis("d_v", D_V);
  const qkvWidth = Q_HEADS * D_QK + KV_HEADS * (D_QK + D_V); // 14,848
  const fp8Weight = { every: 128 * 128, bytes: 4, note: "one FP32 scale per 128 × 128 block of the weight, counted in the bytes." };
  const fp8Input = { every: 128, bytes: 1, note: "one UE8M0 scale per 128 numbers of x, counted in the bytes." };
  // the keys one decode query sees: the last W tokens, itself included
  const windowKeys = (scale) => (scale.context >= WINDOW ? tokens("W", WINDOW) : tokens("L", scale.context));
  // the cache rows a step reads: decode, the window of each request; prefill,
  // the last W − 1 earlier tokens and the chunk
  const cacheRows = (scale) =>
    isDecode(scale)
      ? [tokens("B", scale.batch), windowKeys(scale)]
      : [scale.context >= WINDOW - 1 ? tokens("W−1+N", WINDOW - 1 + scale.chunk) : tokens("L+N", scale.context + scale.chunk)];
  // one query's keys: at most W in both modes
  const keyAxis = (scale) => (isDecode(scale) ? windowKeys(scale) : tokens("W", WINDOW));
  const scores = (scale, name) => operand(name, [...rows(scale), qHeads, keyAxis(scale)], "FP32", { onChip: true });
  const q = (scale) => operand("q", [...rows(scale), qHeads, dQK], "BF16");
  const k = (scale) => operand("k", [...rows(scale), kvHeads, dQK], "BF16");
  const v = (scale) => operand("v", [...rows(scale), kvHeads, dV], "BF16");
  const pairs = (scale) => cappedPairs(scale, WINDOW);

  window.ATTN_BLOG_FLOWS.swa = {
    title: "Sliding window in one window layer of MiMo V2.6 Flash",
    note: "Costs are for one whole layer on one GPU, as in the other figures.",
    axes: [
      ["B", "requests"],
      ["N", "tokens in the chunk"],
      ["L", { decode: "tokens in each request's context, the new one included", prefill: "earlier tokens in the context" }],
      ["W", "window: 128 keys a query sees, itself included"],
      ["H_q", "query heads"],
      ["H_kv", "KV heads"],
      ["d_qk", "query and key head dim"],
      ["d_v", "value head dim"],
      ["D", "hidden size"],
    ],
    input: { label: "x", note: { prefill: "N × 4,096", decode: "B × 4,096" } },
    output: { label: "out", note: "to the MoE FFN" },
    steps: [
      {
        key: "project",
        title: "Projection",
        role: "weights",
        formula: "x · W_qkv",
        does: "Turns each token's hidden state into a query, a key and a value. Q and K heads have 192 numbers, V heads 128.",
        counted: "2 per weight per token, 2 × 4,096 × 14,848 = 122 MFLOP a token.",
        operands: (scale) => ({
          input: [operand("x", [...rows(scale), hidden], "FP8", { scale: fp8Input })],
          weight: [operand("W_qkv", [hidden, axis("H_q·d_qk+H_kv·(d_qk+d_v)", qkvWidth)], "FP8", { scale: fp8Weight })],
          output: [q(scale), k(scale), v(scale)],
          flops: 2 * rowCount(scale) * HIDDEN * qkvWidth,
          summed: ["D"],
        }),
      },
      {
        key: "rope",
        title: "RoPE",
        role: "elementwise",
        edgeIn: "q, k, v",
        formula: "rotate 64 of 192 channels",
        does: "Turns the first 64 channels of each q and k head by an angle set by the token's position. The same kernel scales v by 0.707.",
        counted: "about 3 per rotated number (each pair takes 4 multiplies and 2 adds), 1 per number of v.",
        operands: (scale) => ({
          input: [q(scale), k(scale), v(scale), operand("cos, sin", [...rows(scale), axis("d_rot", ROPE)], "BF16")],
          output: [q(scale), k(scale), v(scale)],
          flops: rowCount(scale) * (3 * ROPE * (Q_HEADS + KV_HEADS) + KV_HEADS * D_V),
          summed: [],
        }),
      },
      {
        key: "score",
        title: "Scores",
        role: "core",
        edgeIn: "q",
        formula: "q · Kᵀ ÷ √192",
        does: {
          prefill: "Compares every query in the chunk with the keys of the 127 tokens before it, at most. Each KV head serves 8 query heads.",
          decode: "Compares q with the keys of the last 127 tokens and its own, never more. Each KV head serves 8 query heads.",
        },
        counted: {
          prefill: "2 × 64 × 192 = 24.6 KFLOP per query–key pair; the query at position j sees min(j, 128) keys.",
          decode: "2 × 64 × 192 = 24.6 KFLOP per query–key pair, B × min(L, 128) pairs: the same at any L ≥ 128.",
        },
        operands: (scale) => ({
          input: [q(scale)],
          cache: [operand("K", [...cacheRows(scale), kvHeads, dQK], "FP8")],
          output: [scores(scale, "S")],
          flops: 2 * Q_HEADS * D_QK * pairs(scale),
          summed: ["d_qk"],
        }),
      },
      {
        key: "mask",
        title: "Window mask",
        role: "elementwise",
        only: "prefill",
        formula: "score = −∞ where key > query or key ≤ query − 128",
        does: "Hides the later tokens of the chunk and every token more than 127 positions back.",
        counted: "none in practice: the kernel skips the hidden pairs instead of computing them.",
        operands: (scale) => ({ input: [scores(scale, "S")], output: [scores(scale, "S")], flops: 0, summed: [] }),
      },
      {
        key: "softmax",
        title: "Softmax",
        role: "elementwise",
        formula: "exp, ÷ (row sum + e^sink)",
        does: "Turns each row of scores into weights. A learned sink per head joins the sum, so the weights may add up to less than 1.",
        counted: "about 5 per score (max, subtract, exp, add, divide).",
        operands: (scale) => ({
          input: [scores(scale, "S")],
          weight: [operand("sink", [qHeads], "FP32")],
          output: [scores(scale, "P")],
          flops: 5 * Q_HEADS * pairs(scale),
          summed: [keyAxis(scale).label],
        }),
      },
      {
        key: "sum",
        title: "Weighted sum",
        role: "core",
        formula: "weights · V",
        does: "Adds up the values of the tokens in the window, each times its weight.",
        counted: "2 × 64 × 128 = 16.4 KFLOP per query–key pair, as many pairs as the scores.",
        operands: (scale) => ({
          input: [scores(scale, "P")],
          cache: [operand("V", [...cacheRows(scale), kvHeads, dV], "FP8")],
          output: [operand("o", [...rows(scale), qHeads, dV], "BF16")],
          flops: 2 * Q_HEADS * D_V * pairs(scale),
          summed: [keyAxis(scale).label],
        }),
      },
      {
        key: "out",
        title: "Output proj.",
        fullTitle: "Output projection",
        role: "weights",
        edgeIn: "o",
        formula: "concat(o) · W_o",
        does: "Joins the 64 heads into 8,192 numbers and maps them back to 4,096.",
        counted: "2 per weight per token, 2 × 8,192 × 4,096 = 67.1 MFLOP a token. The checkpoint leaves W_o unquantized.",
        operands: (scale) => ({
          input: [operand("o", [...rows(scale), axis("H_q·d_v", Q_HEADS * D_V)], "BF16")],
          weight: [operand("W_o", [axis("H_q·d_v", Q_HEADS * D_V), hidden], "BF16")],
          output: [operand("out", [...rows(scale), hidden], "BF16")],
          flops: 2 * rowCount(scale) * Q_HEADS * D_V * HIDDEN,
          summed: ["H_q·d_v"],
        }),
      },
    ],
    branch: {
      title: "Write K, V",
      role: "write",
      formula: "write into a 128-token window",
      does: {
        prefill: "Stores the chunk's keys and values. Once the chunk is done, only its last 128 tokens stay; the older slots are freed.",
        decode: "Stores this token's k and v, and drops the token that falls out of the window, 128 positions back.",
      },
      counted: "about 1 per number (one multiply by the FP8 scale).",
      operands: (scale) => ({
        input: [k(scale), v(scale)],
        output: [operand("K", [...rows(scale), kvHeads, dQK], "FP8"), operand("V", [...rows(scale), kvHeads, dV], "FP8")],
        flops: rowCount(scale) * KV_HEADS * (D_QK + D_V),
        summed: [],
      }),
      // the RoPE kernel scales v too, so the window stores v from it, not from the projection
      from: [{ step: "rope", label: "k, v" }],
    },
    cache: {
      title: "Window cache: only the last 128 tokens",
      rows: ["K", "V"],
      // the two oldest columns are gone: the window keeps the last 127 earlier tokens
      // (decode: L counts this token too, so they end at t(L−2))
      cells: {
        prefill: [{ label: "t0", dropped: true }, { label: "…", dropped: true }, "t(L−127)", "…", "t(L−1)"],
        decode: [{ label: "t0", dropped: true }, { label: "…", dropped: true }, "t(L−128)", "…", "t(L−2)"],
      },
      cellWidth: 66,
      newLabel: { prefill: "chunk", decode: "new" },
      groups: {
        dropped: "dropped",
        old: "last 127 tokens",
        new: { prefill: "N in this chunk", decode: "this token" },
      },
      note: "8 KV heads × (192 + 128) = 2.5 KiB per token and layer in FP8; 320 KiB per layer at most, 12 MiB over 39 window layers",
      reads: [
        { step: "score", row: 0, label: { prefill: "reads K: last 127 + N in chunk", decode: "reads K: last 127 + this token" } },
        { step: "sum", row: 1, label: { prefill: "reads V: last 127 + N in chunk", decode: "reads V: last 127 + this token" } },
      ],
    },
  };
})();
