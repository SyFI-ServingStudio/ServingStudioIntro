/* Linear-attention flow for the "Attention Variants" section; follows the
   layout contract in flows.js's header comment, with a fixed-size state
   drawn as blocks (`cache.layout: "blocks"`) instead of token rows.

   GLM-5.3-Flash's Kimi Delta Attention (KDA) layer, from main/model/config/
   glm53_flash_nvfp4.json: hidden 4,096; linear_attn_config num_heads 64,
   head_dim 128, short_conv_kernel_size 4; 34 KDA layers. The launch order and
   shapes follow ServingStudio Sim's simulator/src/worklet/
   glm53_kda_attn_local.rs and the vLLM fork's glm5next/common/kda.py:
   - in_proj_qkvbfg_a: one BF16 GEMM over x for q, k, v, one beta logit b
     per head, and the two gate factors f_a, g_a (128 each).
   - f_b_proj(f_a) gives f, the raw decay input (g1), g_b_proj(g_a) the
     output gate g (g2). Both read only the projection's output, never q, k
     or v, so they run beside the short conv: the gates lane. The decay α,
     exp(lower_bound · sigmoid(exp(A_log) · (f + dt_bias))) with lower_bound
     −5 (the bounded "safe" gate), is computed in the core kernels in both
     modes (prefill: inside FlashKDA; decode: fused_recurrent_kda
     compute_gate=True), so the Gates card outputs f, not α. b stays raw
     until the core in both modes: decode sigmoids it in-kernel
     (sigmoid_beta=True), and so does FlashKDA. (Only the Triton prefill
     path, chunk_kda_with_fused_gate, needs a separate FP32 sigmoid first,
     _cast_sigmoid.) Decode sends b straight to the core; prefill lets it
     ride the gates lane untouched, because a direct edge would cross the
     q edge over Block prep.
   - causal_conv1d over the merged q, k, v, then SiLU; the conv weights are
     FP32 (_fused_qkv_conv1d, params_dtype float32). Prefill:
     causal_conv1d_fn reads the prefix's last 3 inputs from conv_state as
     left padding and rewrites conv_state in place with the chunk's last 3;
     decode: causal_conv1d_update shifts the window by one token.
   - The KDA core. Decode: fused_recurrent_kda, one kernel that reads S and
     writes it back in place. Prefill: gather_initial_states, then
     chunk_kda_with_fused_gate (q and k L2-normalized in the kernel), then
     scatter_states writes the final S. Its FLA stages (third_party/
     flash_linear_attention/ops/kda.py chunk_kda_with_fused_gate_fwd;
     nvidia copy ops/third_party/kda/kernels.py
     _chunk_kda_fwd_with_cumulative_g) are drawn as three prefill-only
     cards: Block prep (gate cumsum, chunk_kda_scaled_dot_kkt_fwd →
     A and Q Kᵀ, solve_tril, recompute_w_u_fwd → w, u), State pass
     (chunk_gated_delta_rule_fwd_h: the only loop over blocks, carrying S,
     emitting each block's starting S and v_new = D) and Block output
     (chunk_gla_fwd_o_gk). FLA writes w, u, Q Kᵀ, the per-block S and D to
     HBM between launches; the ideal count keeps them on chip.
     The measured prefill (measured_data.js backend "FlashKDA prefill") is
     not this chain but FlashKDA, vLLM's default on B200: one fused kernel,
     same math, in 16-token blocks. The cards keep FLA's 64 because the
     prefill FLOPs figure counts with 64 (GDN_CHUNK_FLOP).
   - o_norm: FusedRMSNormGated(o, g) with a sigmoid gate; then o_proj.
   dtypes: the self_attn / linear_attn weights are BF16 (the NVFP4
   quantization_config covers the MoE experts); state S FP32, conv window
   BF16. State: FP32 [64, 128, 128] = 4 MiB, plus the conv window,
   3 × 24,576 BF16 = 144 KiB. Both are read and written once per decode step,
   as decode_bytes counts them.
   FLOPs: decode, the recurrence's 6 · d_k · d_v per token and head
   (main/model/work/attention/kda.py); prefill, the chunked kernel's
   10.7 MFLOP per head per chunk of 64 tokens, the rule the prefill_flops
   figure uses (video/attn-tools/chapters/ch5/prefill_flops.py
   GDN_CHUNK_FLOP), split over the three cards: K Kᵀ, Q Kᵀ, the inverse and
   the inverse times the values in Block prep; w S and Kᵀ D in State pass;
   Q S and the masked Q Kᵀ times D in Block output. The sum is the rule.
   Costs cover the whole layer, as if it ran on one GPU, like the GQA figure. */

(() => {
  const { axis, tokens, operand, rows, rowCount, isDecode } = window.ATTN_FLOW_KIT;
  const HIDDEN = 4096;
  const HEADS = 64;
  const HEAD_DIM = 128; // d_k = d_v
  const GATE_RANK = 128;
  const CONV = 4;
  const CHUNK = 64;
  const QKV = 3 * HEADS * HEAD_DIM; // 24,576
  const IN_WIDTH = QKV + HEADS + 2 * GATE_RANK; // 24,896
  const VALUE_WIDTH = HEADS * HEAD_DIM; // 8,192
  const hidden = axis("D", HIDDEN);
  const heads = axis("H", HEADS);
  const dK = axis("d_k", HEAD_DIM);
  const dV = axis("d_v", HEAD_DIM);
  const d = axis("d", HEAD_DIM);
  const qkvWidth = axis("3·H·d", QKV);
  const gateRank = axis("r", GATE_RANK);
  const valueWidth = axis("H·d", VALUE_WIDTH);
  const window3 = axis("3 tokens", CONV - 1);
  // per-request state: decode one per request; prefill the one request's
  const perRequest = (scale) => (isDecode(scale) ? [tokens("B", scale.batch)] : []);
  const stateS = (scale, onChip = false) => operand("S", [...perRequest(scale), heads, dK, dV], "FP32", { onChip });
  const convState = (scale) => operand("conv", [...perRequest(scale), window3, qkvWidth], "BF16");
  // one prefill card's share of the chunked kernel: per head and 64-token block, times heads and blocks
  const blockFlops = (scale, perHeadBlock) => perHeadBlock * HEADS * Math.ceil(scale.chunk / CHUNK);

  window.ATTN_BLOG_FLOWS.linear = {
    title: "Kimi Delta Attention in one linear layer of GLM-5.3-Flash",
    note: "Costs are for one whole layer on one GPU, as in the other figures.",
    axes: [
      ["B", "requests"],
      ["N", "tokens in the chunk"],
      ["N/64", "64-token blocks in the chunk"],
      ["64", "tokens in one block"],
      ["H", "heads"],
      ["d", "head dim"],
      ["d_k", "key channels: one row of S each"],
      ["d_v", "value channels: one column of S each"],
      ["r", "rank of the gate factors"],
      ["D", "hidden size"],
    ],
    input: { label: "x", note: { prefill: "N × 4,096", decode: "B × 4,096" } },
    output: { label: "out", note: "to the MoE FFN" },
    // The gate GEMMs read only the projection's f_a, g_a, so they run beside
    // the short conv; f joins the core, g the gated norm.
    lanes: {
      gates: {
        side: "above",
        // prefill: b rides this lane untouched (a direct edge would cross the
        // q edge over Block prep); decode sends it straight to the core (`edges`)
        from: { step: "project", label: { prefill: "f_a, g_a, b", decode: "f_a, g_a" } },
        // prefill joins the first chunked card, decode the one recurrent kernel
        to: [
          { step: "prep", label: "f, b" },
          { step: "core", label: "f" },
          { step: "norm", label: "g" },
        ],
      },
    },
    // decode: b skips Gates. Prefill: Block output reads q from the short conv
    // and Q Kᵀ from Block prep, both over the main row; q, the longer one,
    // runs higher and hops the Gates → Block prep join.
    edges: [
      { from: "project", to: "core", only: "decode", label: "b", level: "over", dy: 9, fromDx: 76, dx: -74 },
      { from: "conv", to: "readout", only: "prefill", label: "q", level: "over", dy: 38, fromDx: 40, dx: 40 },
      { from: "prep", to: "readout", only: "prefill", label: "Q Kᵀ", level: "over", dy: 14, fromDx: 40, dx: -30 },
    ],
    steps: [
      {
        key: "project",
        title: "Projection",
        role: "weights",
        formula: "x · [W_q, W_k, W_v, W_b, W_fa, W_ga]",
        does: "Turns each token's hidden state into a query, a key and a value for each of the 64 heads, plus the inputs of the gates: one write-strength logit b per head and two 128-number factors f_a, g_a.",
        counted: "2 per weight per token, 2 × 4,096 × 24,896 = 204 MFLOP a token (one GEMM).",
        operands: (scale) => ({
          input: [operand("x", [...rows(scale), hidden], "BF16")],
          weight: [operand("W_in", [hidden, axis("3·H·d+H+2r", IN_WIDTH)], "BF16")],
          output: [
            operand("q, k, v", [...rows(scale), qkvWidth], "BF16"),
            operand("b", [...rows(scale), heads], "BF16"),
            operand("f_a, g_a", [...rows(scale), axis("2r", 2 * GATE_RANK)], "BF16"),
          ],
          flops: 2 * rowCount(scale) * HIDDEN * IN_WIDTH,
          summed: ["D"],
        }),
      },
      {
        key: "conv",
        title: "Short conv",
        role: "elementwise",
        edgeIn: "q, k, v",
        formula: "Σ over the last 4 tokens, per channel; SiLU",
        does: {
          prefill: "Mixes each channel of q, k and v with the same channel of the 3 tokens before it. The chunk's first tokens take theirs from the conv state.",
          decode: "Mixes each channel of q, k and v with the same channel of the 3 tokens before it, which it takes from the conv state.",
        },
        counted: "2 per tap per channel, 2 × 4 × 24,576 = 197 KFLOP a token, plus about 4 per number for the SiLU.",
        operands: (scale) => ({
          input: [operand("q, k, v", [...rows(scale), qkvWidth], "BF16")],
          weight: [operand("w_conv", [qkvWidth, axis("4 taps", CONV)], "FP32")],
          state: [convState(scale)],
          output: [operand("q, k, v", [...rows(scale), qkvWidth], "BF16")],
          flops: rowCount(scale) * QKV * (2 * CONV + 4),
          summed: ["4 taps"],
        }),
      },
      {
        key: "gates",
        title: "Gates",
        role: "weights",
        lane: "gates",
        formula: "f = f_a · W_fb,  g = g_a · W_gb",
        does: {
          prefill:
            "Expands f_a into f, one number per key channel of each head, and g_a into the output gate g, which the gated norm applies after the core. f is not the decay yet: the core kernel turns it into α, between 0 and 1. b passes by unchanged; the core kernel turns it into β = sigmoid(b).",
          decode:
            "Expands f_a into f, one number per key channel of each head, and g_a into the output gate g, which the gated norm applies after the core. f is not the decay yet: the core kernel turns it into α, between 0 and 1. b skips this step and goes straight to the core kernel, which turns it into β = sigmoid(b) itself.",
        },
        counted: "2 per weight per token, 2 × 2 × 128 × 8,192 = 4.2 MFLOP a token.",
        operands: (scale) => ({
          // b is no operand here: the core kernel reads it raw in both modes
          input: [operand("f_a, g_a", [...rows(scale), axis("2r", 2 * GATE_RANK)], "BF16")],
          weight: [operand("W_fb", [gateRank, valueWidth], "BF16"), operand("W_gb", [gateRank, valueWidth], "BF16")],
          output: [
            operand("f", [...rows(scale), heads, dK], "BF16"),
            operand("g", [...rows(scale), valueWidth], "BF16"),
          ],
          flops: rowCount(scale) * 2 * 2 * GATE_RANK * VALUE_WIDTH,
          summed: ["r"],
        }),
      },
      // Prefill: chunk_kda_with_fused_gate, split into its three stages so the
      // blocks show (FLA chunk_kda_fwd: kkt + solve_tril + recompute_w_u, then
      // chunk_gated_delta_rule_fwd_h, then chunk_gla_fwd_o_gk). The three cards'
      // FLOPs are GDN_CHUNK_FLOP's terms, split by stage; their sum is the rule.
      {
        key: "prep",
        title: "Block prep",
        fullTitle: "Per-block preparation",
        role: "core",
        caption: "64-token blocks",
        only: "prefill",
        edgeIn: "q, k, v",
        formula: "per block of 64:  α from f, β from b;  A = β K Kᵀ (lower);  (I + A)⁻¹;  w, u = (I + A)⁻¹ (β K, β V);  Q Kᵀ (lower)",
        does:
          "Cuts the chunk into blocks of 64 tokens and works on all blocks at once. It turns f into the decay α and b into the write strength β = sigmoid(b). Inside each block it measures how the 64 writes overlap, K Kᵀ, and solves that triangular system, so the block's 64 one-token writes become one matrix update, w and u. It also takes Q Kᵀ inside the block for the output step. Nothing here reads S.",
        counted:
          "per head and block: K Kᵀ and Q Kᵀ, 2 × 2 × 64² × 128 = 2.1 MFLOP; the triangular inverse, 2 × 64³ ÷ 3 = 0.17 MFLOP; the inverse times the values, 2 × 64² × 128 = 1.05 MFLOP. α, β and the L2 norms of q and k are not counted. w, u and Q Kᵀ count as on chip.",
        operands: (scale) => ({
          input: [
            operand("q, k, v", [...rows(scale), axis("3·H", 3 * HEADS), d], "BF16"),
            operand("f", [...rows(scale), heads, dK], "BF16"),
            operand("b", [...rows(scale), heads], "BF16"),
          ],
          weight: [operand("A, bias", [axis("H·d", VALUE_WIDTH)], "FP32")],
          output: [operand("w, u", [...rows(scale), heads, axis("d_k+d_v", 2 * HEAD_DIM)], "BF16", { onChip: true }), operand("Q Kᵀ", [...rows(scale), heads, axis("64", CHUNK)], "FP32", { onChip: true })],
          flops: blockFlops(scale, 2 * 2 * CHUNK * CHUNK * HEAD_DIM + (2 * CHUNK ** 3) / 3 + 2 * CHUNK * CHUNK * HEAD_DIM),
          summed: ["d"],
        }),
      },
      {
        key: "pass",
        title: "State pass",
        fullTitle: "State pass, block by block",
        role: "core",
        caption: "S, block to block",
        only: "prefill",
        edgeIn: "w, u",
        formula: "for each block in order:  D = u − w S;  S ← Diag(α) S + Kᵀ D",
        does:
          "The one serial step: it walks the blocks in order, carrying S from each to the next. For a block, w S is what S already stores for its keys; subtracting it from u leaves the correction D. Then S decays across the block and takes Kᵀ D. S is read once, before the first block, and the S after the last block is written back once. The S each block started from goes on to the output step.",
        counted: "per head and block: w S, 2 × 64 × 128² = 2.1 MFLOP, and Kᵀ D, 2 × 128 × 64 × 128 = 2.1 MFLOP.",
        operands: (scale) => ({
          input: [operand("w, u", [...rows(scale), heads, axis("d_k+d_v", 2 * HEAD_DIM)], "BF16", { onChip: true })],
          state: [stateS(scale)],
          output: [operand("S per block", [axis("N/64", Math.ceil(scale.chunk / CHUNK)), heads, dK, dV], "FP32", { onChip: true }), operand("D", [...rows(scale), heads, dV], "BF16", { onChip: true }), stateS(scale, true)],
          flops: blockFlops(scale, 2 * CHUNK * HEAD_DIM * HEAD_DIM + 2 * HEAD_DIM * CHUNK * HEAD_DIM),
          summed: ["d_k", "64"],
        }),
      },
      {
        key: "readout",
        title: "Block output",
        fullTitle: "Per-block read-out",
        role: "core",
        caption: "64-token blocks",
        only: "prefill",
        edgeIn: "S, D",
        formula: "per block:  o = Q S_start + (Q Kᵀ, lower) D",
        does:
          "Works on all blocks at once again. A token's output is what S held at its block's start, read with its query and decayed to its position, plus what the earlier tokens of its own block wrote, through Q Kᵀ and D.",
        counted:
          "per head and block: Q S, 2 × 64 × 128² = 2.1 MFLOP, and the masked Q Kᵀ times D, 2 × 64² × 128 = 1.05 MFLOP. The three prefill cards come to 10.7 MFLOP per head and block, the rule of the prefill FLOPs figure, the same at any L.",
        operands: (scale) => ({
          input: [
            operand("q", [...rows(scale), heads, dK], "BF16", { reused: true }),
            operand("S per block", [axis("N/64", Math.ceil(scale.chunk / CHUNK)), heads, dK, dV], "FP32", { onChip: true }),
            operand("D", [...rows(scale), heads, dV], "BF16", { onChip: true }),
            operand("Q Kᵀ", [...rows(scale), heads, axis("64", CHUNK)], "FP32", { onChip: true }),
          ],
          output: [operand("o", [...rows(scale), heads, dV], "BF16")],
          flops: blockFlops(scale, 2 * CHUNK * HEAD_DIM * HEAD_DIM + 2 * CHUNK * CHUNK * HEAD_DIM),
          summed: ["d_k", "64"],
        }),
      },
      {
        key: "core",
        title: "Delta rule",
        fullTitle: "Delta-rule update and read-out",
        role: "core",
        caption: "GEMM · state",
        only: "decode",
        edgeIn: "q, k, v",
        formula: "α = decay(f), β = sigmoid(b);  S ← Diag(α) S;  S ← S + β k (v − Sᵀk)ᵀ;  o = Sᵀ q",
        does: "Turns f into the decay α and b into β. Then it decays each row of S by its α, reads what S stores for the key, Sᵀk, and writes the correction β(v − Sᵀk), so a key written again replaces its old value. Last it reads the output with q. One kernel, the same cost at any L.",
        counted: "6 × 128 × 128 = 98 KFLOP per head and token: decay, Sᵀk, the rank-1 write and Sᵀq; the same at any L. α, β and the L2 norms of q and k, computed in the kernel, are not counted.",
        operands: (scale) => ({
          input: [
            operand("q, k, v", [...rows(scale), axis("3·H", 3 * HEADS), d], "BF16"),
            operand("f", [...rows(scale), heads, dK], "BF16"),
            operand("b", [...rows(scale), heads], "BF16"),
          ],
          weight: [operand("A, bias", [axis("H·d", VALUE_WIDTH)], "FP32")],
          state: [stateS(scale)],
          output: [operand("o", [...rows(scale), heads, dV], "BF16"), stateS(scale, true)],
          flops: 6 * HEAD_DIM * HEAD_DIM * HEADS * scale.batch,
          summed: ["d_k", "d_v"],
        }),
      },
      {
        key: "norm",
        title: "Gated norm",
        role: "elementwise",
        edgeIn: "o",
        formula: "o ÷ rms × γ ⊙ sigmoid(g)",
        does: "Rescales each head's output, then lets the output gate g scale every channel. This gate is separate from the decay α.",
        counted: "about 4 per number for the norm and 5 for the sigmoid gate.",
        operands: (scale) => ({
          input: [operand("o", [...rows(scale), heads, dV], "BF16"), operand("g", [...rows(scale), valueWidth], "BF16")],
          weight: [operand("γ", [dV], "BF16")],
          output: [operand("o", [...rows(scale), valueWidth], "BF16")],
          flops: 9 * rowCount(scale) * VALUE_WIDTH,
          summed: ["d_v"],
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
        counted: "2 per weight per token, 2 × 8,192 × 4,096 = 67.1 MFLOP a token.",
        operands: (scale) => ({
          input: [operand("o", [...rows(scale), valueWidth], "BF16")],
          weight: [operand("W_o", [valueWidth, hidden], "BF16")],
          output: [operand("out", [...rows(scale), hidden], "BF16")],
          flops: 2 * rowCount(scale) * VALUE_WIDTH * HIDDEN,
          summed: ["H·d"],
        }),
      },
    ],
    // Two write cards, one per state, so no card does two jobs. Like every
    // write card in these figures, each stands for the write itself; which
    // kernel performs it is in its text (conv: the conv kernel; S: the
    // recurrent kernel in decode, its own scatter_states kernel in prefill).
    branch: [
      {
        title: "Write conv",
        role: "write",
        caption: "state write",
        formula: "overwrite in place",
        does: {
          prefill: "Replaces all 3 conv columns with the chunk's last 3 inputs; the prefix's 3 are no longer needed. The conv kernel does this itself, after the convolution.",
          decode: "Shifts the conv window by one: this token's inputs enter on the right, the oldest leave. The conv kernel does this itself.",
        },
        counted: "no arithmetic: a copy.",
        operands: (scale) => ({
          input: [operand("q, k, v", [...(isDecode(scale) ? rows(scale) : [tokens("3", CONV - 1)]), qkvWidth], "BF16", { onChip: isDecode(scale) })],
          output: [convState(scale)],
          flops: 0,
          summed: [],
        }),
        // what is kept is the conv's input, not its output
        from: [{ step: "conv", label: { prefill: "last 3 inputs", decode: "inputs" } }],
      },
      {
        title: "Write S",
        role: "write",
        caption: "state write",
        formula: "overwrite in place",
        does: {
          prefill: "Writes the S left after the chunk's last block over the old one. vLLM runs this as a small kernel of its own (scatter_states) after the chunked kernel.",
          decode: "Writes the updated S over the old one. The recurrent kernel does this itself, at its end.",
        },
        counted: "no arithmetic: a copy.",
        operands: (scale) => ({
          input: [stateS(scale, true)],
          output: [stateS(scale)],
          flops: 0,
          summed: [],
        }),
        // S leaves the card that updates it (`fromCard`), right of the read
        // that rises into that card; the main arrow there carries o, not S
        from: [{ step: { prefill: "pass", decode: "core" }, label: "S", fromCard: true, dx: 70 }],
      },
    ],
    cache: {
      title: "State, fixed in size",
      layout: "blocks",
      // conv window left, under the short conv; S right, under the delta
      // rule (prefill: the state pass). Each read rises into its card between
      // the two write cards; each write card drops into its own block.
      blocks: [
        {
          key: "conv",
          kind: "columns",
          under: "conv",
          cells: { prefill: ["t(L−3)", "t(L−2)", "t(L−1)"], decode: ["t−3", "t−2", "t−1"] },
          // prefill: the whole chunk enters at once; its last 3 (the strips) become the state
          slotWidth: { prefill: 92, decode: null },
          slotText: { prefill: "N", decode: "" },
          keepLast: { prefill: 3, decode: 0 },
          inLabel: { prefill: "N in this chunk", decode: "t in" },
          outLabel: { prefill: "all 3 out", decode: "t−3 out" },
          caption: {
            prefill: "conv, 144 KiB: the prefix's last 3 pad the chunk; its last 3 replace them",
            decode: "conv: last 3 inputs, 3 × 24,576 BF16 = 144 KiB",
          },
          captionWidth: 250,
        },
        {
          key: "S",
          kind: "matrix",
          under: { prefill: "pass", decode: "core" },
          dx: -10,
          inner: "S",
          dims: "d_k × d_v\n128 × 128",
          caption: {
            prefill: "S: 64 × 128 × 128 FP32 = 4 MiB; read once, written once",
            decode: "S: 64 heads × 128 × 128, FP32 = 4 MiB",
          },
          captionWidth: 200,
        },
      ],
      groups: { state: "updated in place; never grows" },
      note: "4.1 MiB per layer, whatever L is; 141 MiB per request over 34 KDA layers",
      reads: [
        { step: "conv", block: "conv", label: { prefill: "pads with last 3", decode: "reads last 3" } },
        { step: "pass", only: "prefill", block: "S", dx: 10, label: "reads S once", labelDy: 40 },
        { step: "core", only: "decode", block: "S", dx: 10, label: "reads S", labelDy: 40 },
      ],
      writes: [
        { block: "conv", branch: 0, label: "" },
        { block: "S", branch: 1, dx: 60, label: "" },
      ],
    },
  };
})();
