/* Compressed-and-shared-cache flow for the "Attention Variants" section;
   follows the layout contract in flows.js's header comment.

   Layer 2 of DeepSeek V4.1 Flash, the first layer that writes a compressed
   cache. From main/model/config/deepseek_v41_flash.json: hidden 5,120;
   64 heads of 512 (key = value, one shared KV head); q_lora_rank 1,280;
   qk_rope_head_dim 64; o_groups 8 × o_lora_rank 1,024; sliding_window 128;
   compress_ratios[2] = 2. Only kv_source_layer_ids [2, 8, 14, 20] write a
   compressed cache; every other layer reads the nearest writer at or before
   it. The indexer runs in index_source_layer_ids [2, 8, 14, 20, 24, 28, 32,
   36]: the 4 writers, and 24–36, which select again from layer 20's cache;
   every other layer reuses the selection of the nearest indexer layer before
   it (vLLM fork models/deepseek_v41/attention.py:333-337). For layer 2 both
   come to the same thing: layers 3–7 read its cache and reuse its selection.
   index_n_heads 32 × index_head_dim 128; index_topk 512.
   Record sizes from main/model/work/attention/deepseek_v41.py and the
   simulator's kv layout (simulator/src/arch/deepseek_v41_vllm.rs): window
   entry 528 B (MXFP8), compressed entry 288 B (NVFP4), indexer key 68 B (FP4;
   vLLM keeps 132 B in FP8). Each query attends to min(pos + 1, 128) window
   keys plus min((pos + 1) / 2, 512) compressed entries, 640 at most
   (timing/kernels/compressed_sparse_mla_rope_cast.rs). Each card is one job:
   a GEMM card holds only GEMMs, and the elementwise and scoring work sits
   in cards of its own.
   Edges follow vLLM's forward (attention.py _run_parallel_input_projections,
   _split_qkv_and_norm, _prepare_and_attn, DeepseekV4Indexer.forward;
   compressor.py):
   - x feeds three GEMMs at once: fused_wqa_wkv (c_q, kv) on the main
     stream, and the compressor's fused_wkv_wgate and the indexer's
     weights_proj (w) on side streams. The two side GEMMs share the Side
     proj. card; w skips to the index scores on an edge of its own.
   - fused_q_kv_rmsnorm normalizes c_q and kv in one kernel; the paths split
     after it. kv goes only to the window write, which vLLM fuses with the
     q RoPE (fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert: RoPE, MXFP8
     quant, insert), so kv runs over Q up-proj. into the q RoPE card and drops
     from it into the write card; attention reads it back from the window
     ("attention reads kv solely via swa_kv_cache"). c_q goes to W_qb and to
     the indexer's own wq_b (W_IQ): two GEMMs, drawn as one Q up-proj. card
     as in the DSA figure. The q RoPE card also rotates q_I, which vLLM does
     with its FP4 cast in fused_indexer_q_rope_quant.
     No per-head q norm: _fused_qnorm_rope_kv_insert passes apply_q_norm=False
     ("qr is normed before wq_b", attention.py:1045) and FlashMLA's mega
     kernel enable_q_norm=False (nvidia/flash_mla_mega_attn.py:372), so the
     q RoPE card stays RoPE only.
   - The compressor (fused_save_compress_norm: state save, pair merge and
     RMSNorm in one kernel) merges each pair into a pre-RoPE latent c. The
     Index key proj. card is the indexer's wk(c) GEMM (_produce_k); the
     write card stands for rope_quant_insert (c) and indexer_k_norm_rope_store
     (k_norm, RoPE, FP4, store) besides the window insert. All writes land
     before attention and the index scores read the caches.
   - The indexer scores (indexer_op) and keeps the top 512 into
     topk_indices_buffer, which forward_mqa reads.
   - The attention kernel (FlashMLA mega attention) also does the inverse
     RoPE on the output's last 64 channels and the FP8 cast before W_oa
     (attention.py _o_proj docstring: "a layer whose attention kernel
     already did the inverse RoPE and the FP8 cast"), so the Attention card
     says so and its output o is FP8.
   Lanes (flows.js contract): both run in the row under the main row. The
   compressor lane leaves the input and ends in the write card, which sits
   right under the q RoPE card, so kv drops straight down. The indexer lane
   leaves the main line after the q RoPE, right of the write card, so no
   line crosses a card row, and joins attention with the selected positions.
   w runs under that row into the index scores and hops the write card's
   drop, the figure's one crossing. The panel hangs under the write card;
   the index scores and attention read from the rows' right ends.
   Costs cover the whole layer, as if it ran on one GPU, like the GQA figure.
   Counted as the ideal: W_k over each new entry (vLLM runs it over every
   token and skips the rows that close no pair), and the index scores even
   when a request holds 512 entries or fewer (vLLM then keeps them all
   without scoring). The compressor GEMM's FP32 output v, s is listed now
   that the GEMM and the merge are separate cards.
   dtypes: the config's quantization_config is quant_method fp8 with
   weight_block_size [32, 32] and UE8M0 scales, so W_qa, W_kv, W_qb, the
   indexer's W_IQ (wq_b) and W_oa, W_ob are FP8 with one 1-byte scale per 32 × 32
   block (the simulator runs them as MXFP8 GEMMs). The compressor (W_ckv,
   W_gate), the indexer's key projection W_k and its head weights W_IW stay
   BF16 (simulator/src/worklet/deepseek_v41_attention_tp.rs: bf16 inputs,
   ReplicatedLinear). The compressor state keeps 8 rows of 2 × 512 FP32 per
   writer layer (deepseek_v41_vllm.rs RING_ROWS), read and written each step,
   as decode_bytes counts it. Operands (flows.js contract): decode L counts
   this token too; m = 2 tokens per compressed entry in layer 2. */

(() => {
  const { axis, tokens, operand, rows, rowCount, cappedPairs, isDecode } = window.ATTN_FLOW_KIT;
  const HIDDEN = 5120;
  const HEADS = 64;
  const HEAD_DIM = 512; // key = value, one shared KV head
  const Q_LATENT = 1280;
  const ROPE = 64;
  const O_GROUPS = 8;
  const O_LATENT = 1024;
  const WINDOW = 128;
  const RATIO = 2; // m: tokens per compressed entry in layer 2
  const INDEX_HEADS = 32;
  const INDEX_DIM = 128;
  const TOP_K = 512;
  const STATE_ROWS = 8;
  const hidden = axis("D", HIDDEN);
  const heads = axis("H", HEADS);
  const headDim = axis("d", HEAD_DIM);
  const qLatent = axis("r_q", Q_LATENT);
  const indexHeads = axis("H_I", INDEX_HEADS);
  const indexDim = axis("d_I", INDEX_DIM);
  const fp8Weight = { every: 32 * 32, bytes: 1, note: "one UE8M0 scale per 32 × 32 block of the weight, counted in the bytes." };
  const fp4Key = { every: 128, bytes: 4, note: "one 4-byte scale per 128-number index key (68 B a key), counted in the bytes." };
  // per-token rows, and rows made one per m tokens (decode: B/m, half an entry per request and step on average)
  const entryRows = (scale) => (isDecode(scale) ? [tokens("B/m", scale.batch / RATIO)] : [tokens("N/m", scale.chunk / RATIO)]);
  // compressed entries one request holds: decode L/m; prefill the earlier and the chunk's, (L+N)/m
  const entryCache = (scale) =>
    isDecode(scale)
      ? [tokens("B", scale.batch), tokens("L/m", Math.floor(scale.context / RATIO))]
      : [tokens("(L+N)/m", Math.floor((scale.context + scale.chunk) / RATIO))];
  // entries one query keeps: decode min(L/m, 512); prefill, at most every entry once (each query picks its own)
  // every entry an index score compares against: decode L/m; prefill (L+N)/m
  const entryAxis = (scale) =>
    isDecode(scale) ? tokens("L/m", Math.floor(scale.context / RATIO)) : tokens("(L+N)/m", Math.floor((scale.context + scale.chunk) / RATIO));
  const selected = (scale) => tokens("k", Math.min(TOP_K, Math.floor((isDecode(scale) ? scale.context : scale.context + 1) / RATIO)));
  const selectedCache = (scale) => (isDecode(scale) ? [tokens("B", scale.batch), selected(scale)] : entryCache(scale));
  // the window rows a step reads: decode the last W of each request; prefill the last W − 1 earlier and the chunk
  const windowCache = (scale) =>
    isDecode(scale)
      ? [tokens("B", scale.batch), scale.context >= WINDOW ? tokens("W", WINDOW) : tokens("L", scale.context)]
      : [scale.context >= WINDOW - 1 ? tokens("W−1+N", WINDOW - 1 + scale.chunk) : tokens("L+N", scale.context + scale.chunk)];
  // window rows the write keeps: decode this token; prefill the chunk's last W
  const windowWrites = (scale) => (isDecode(scale) ? rows(scale) : [tokens("N", Math.min(scale.chunk, WINDOW))]);
  const compressorState = (scale) =>
    operand("state", [...(isDecode(scale) ? [tokens("B", scale.batch)] : []), axis("8 rows", STATE_ROWS), axis("2·d", 2 * HEAD_DIM)], "FP32");
  const attentionPairs = (scale) => cappedPairs(scale, WINDOW) + cappedPairs(scale, TOP_K, RATIO);
  const indexPairs = (scale) => cappedPairs(scale, Infinity, RATIO);

  window.ATTN_BLOG_FLOWS.compressed = {
    title: "Compressed and shared caches in layer 2 of DeepSeek V4.1 Flash",
    note: "Costs are for one whole layer on one GPU, as in the other figures.",
    axes: [
      ["B", "requests"],
      ["N", "tokens in the chunk"],
      ["L", { decode: "tokens in each request's context, the new one included", prefill: "earlier tokens in the context" }],
      ["m", "tokens per compressed entry: 2 in layer 2"],
      ["W", "window: 128 tokens, itself included"],
      ["k", "selected entries: min(L/m, 512) per query"],
      ["H", "heads"],
      ["d", "head dim; key = value"],
      ["r_q", "query latent"],
      ["H_I", "index heads"],
      ["d_I", "index head dim"],
      ["D", "hidden size"],
    ],
    input: { label: "x", note: { prefill: "N × 5,120", decode: "B × 5,120" } },
    output: { label: "out", note: "to the MoE FFN" },
    // w = indexer.weights_proj(x) runs on a side stream beside the projection
    // (attention.py _run_parallel_input_projections), in the Side proj. card;
    // it runs under the lane row and hops the write card's drop into the
    // cache, the one crossing. kv rides over Q up-proj. into the q RoPE card,
    // whose kernel also inserts kv into the window.
    edges: [
      { from: "compressProj", to: "index", label: "w", level: "underBelow", dy: 16, enter: "below", dx: -40, labelDx: 40 },
      { from: "norms", to: "rope", label: "kv", level: "over", dy: 16 },
    ],
    lanes: {
      compressor: { side: "below", from: { step: "input", label: "x" } },
      indexer: { side: "below", from: { step: "rope", label: "q_I" }, to: [{ step: "attend", label: "512 selected" }] },
    },
    steps: [
      {
        key: "project",
        title: "Projection",
        role: "weights",
        formula: "x · [W_qa, W_kv]",
        does: "Compresses each token's hidden state into c_q for the query and makes one 512-number kv, the key and the value of every head.",
        counted: "2 per weight per token: 2 × 5,120 × 1,792 = 18.4 MFLOP a token for the fused W_qa, W_kv GEMM.",
        operands: (scale) => ({
          input: [operand("x", [...rows(scale), hidden], "BF16")],
          weight: [
            operand("W_qa", [hidden, qLatent], "FP8", { scale: fp8Weight }),
            operand("W_kv", [hidden, headDim], "FP8", { scale: fp8Weight }),
          ],
          output: [operand("c_q", [...rows(scale), qLatent], "BF16"), operand("kv", [...rows(scale), headDim], "BF16")],
          flops: 2 * rowCount(scale) * HIDDEN * (Q_LATENT + HEAD_DIM),
          summed: ["D"],
        }),
      },
      {
        key: "norms",
        title: "Norms",
        role: "elementwise",
        edgeIn: "c_q, kv",
        formula: "RMSNorm(c_q), RMSNorm(kv)",
        does: "Rescales c_q and kv, each with its own learned weights, in one kernel. Then the paths split: c_q to the query and the indexer, kv to the window.",
        counted: "about 4 per number.",
        operands: (scale) => ({
          input: [operand("c_q", [...rows(scale), qLatent], "BF16"), operand("kv", [...rows(scale), headDim], "BF16")],
          weight: [operand("γ_q", [qLatent], "BF16"), operand("γ_kv", [headDim], "BF16")],
          output: [operand("c_q", [...rows(scale), qLatent], "BF16"), operand("kv", [...rows(scale), headDim], "BF16")],
          flops: 4 * rowCount(scale) * (Q_LATENT + HEAD_DIM),
          summed: ["r_q", "d"],
        }),
      },
      {
        key: "query",
        title: "Q up-proj.",
        fullTitle: "Query up-projection",
        role: "weights",
        edgeIn: "c_q",
        formula: "c_q · [W_qb, W_IQ]",
        does: "Expands c_q into a 512-number query for each of the 64 heads, and into the indexer's query q_I for each of its 32 heads. vLLM runs the two as separate GEMMs.",
        counted: "2 per weight per token, 2 × 1,280 × (32,768 + 4,096) = 94.4 MFLOP a token.",
        operands: (scale) => ({
          input: [operand("c_q", [...rows(scale), qLatent], "BF16")],
          weight: [
            operand("W_qb", [qLatent, axis("H·d", HEADS * HEAD_DIM)], "FP8", { scale: fp8Weight }),
            operand("W_IQ", [qLatent, axis("H_I·d_I", INDEX_HEADS * INDEX_DIM)], "FP8", { scale: fp8Weight }),
          ],
          output: [operand("q", [...rows(scale), heads, headDim], "BF16"), operand("q_I", [...rows(scale), indexHeads, indexDim], "BF16")],
          flops: 2 * rowCount(scale) * Q_LATENT * (HEADS * HEAD_DIM + INDEX_HEADS * INDEX_DIM),
          summed: ["r_q"],
        }),
      },
      {
        key: "rope",
        title: "q RoPE",
        role: "elementwise",
        edgeIn: "q, q_I",
        formula: "rotate the last 64 channels of each head of q and q_I",
        does: "Turns the last 64 channels of each head's query by an angle set by the token's position, for the attention query q and the index query q_I. vLLM rotates q in the kernel that writes kv into the window (so kv comes here), or inside the attention kernel; it rotates q_I and casts it to FP4 in one more kernel. q is not normalized again: c_q was normalized before W_qb.",
        counted: "about 3 per rotated number.",
        operands: (scale) => ({
          input: [
            operand("q", [...rows(scale), heads, headDim], "BF16"),
            operand("q_I", [...rows(scale), indexHeads, indexDim], "BF16"),
            operand("cos, sin", [...rows(scale), axis("d_rot", ROPE)], "BF16"),
          ],
          output: [operand("q", [...rows(scale), heads, headDim], "BF16"), operand("q_I", [...rows(scale), indexHeads, indexDim], "FP4", { scale: fp4Key })],
          flops: 3 * rowCount(scale) * (HEADS + INDEX_HEADS) * ROPE,
          summed: [],
        }),
      },
      {
        key: "compressProj",
        title: "Side proj.",
        fullTitle: "Side-stream projections",
        role: "weights",
        lane: "compressor",
        formula: "v, s = x · [W_ckv, W_gate],  w = x · W_IW",
        does: "Two small GEMMs over x that vLLM runs on side streams beside the Projection. The compressor's (fused_wkv_wgate) makes a 512-number value v and 512 gate scores s for every token. The indexer's makes one weight per index head, w.",
        counted: "2 per weight per token, 2 × 5,120 × (1,024 + 32) = 10.8 MFLOP a token.",
        operands: (scale) => ({
          input: [operand("x", [...rows(scale), hidden], "BF16")],
          weight: [operand("W_ckv", [hidden, headDim], "BF16"), operand("W_gate", [hidden, headDim], "BF16"), operand("W_IW", [hidden, indexHeads], "BF16")],
          output: [operand("v, s", [...rows(scale), axis("2·d", 2 * HEAD_DIM)], "FP32"), operand("w", [...rows(scale), indexHeads], "BF16")],
          flops: 2 * rowCount(scale) * HIDDEN * (2 * HEAD_DIM + INDEX_HEADS),
          summed: ["D"],
        }),
      },
      {
        key: "compress",
        title: "Compress",
        role: "elementwise",
        lane: "compressor",
        edgeIn: "v, s",
        formula: "c = RMSNorm(Σ_t softmax(s_t) ⊙ v_t) over 2 tokens",
        does: {
          prefill: "Merges each pair of tokens into one 512-number entry, weighted channel by channel by the softmax of their gate scores, then normalizes it. One kernel (fused_save_compress_norm) also saves the chunk's last unpaired token in a small FP32 buffer.",
          decode: "Every second token completes a pair, which merges into one entry, weighted by the gate scores, and is normalized. The first token of a pair waits in a small FP32 buffer. One kernel (fused_save_compress_norm) does all three.",
        },
        counted: "about 4 per number for the RMSNorm of each entry; the merge is not counted. Decode makes half an entry per request and step on average (B/m).",
        operands: (scale) => ({
          input: [operand("v, s", [...rows(scale), axis("2·d", 2 * HEAD_DIM)], "FP32")],
          weight: [operand("γ_c", [headDim], "BF16")],
          state: [compressorState(scale)],
          output: [operand("c", [...entryRows(scale), headDim], "BF16"), compressorState(scale)],
          flops: (rowCount(scale) / RATIO) * 4 * HEAD_DIM,
          summed: [],
        }),
      },
      {
        key: "indexKey",
        title: "Index key proj.",
        fullTitle: "Index key projection",
        role: "weights",
        lane: "compressor",
        edgeIn: "c",
        formula: "k_c = c · W_k",
        does: "Turns each compressed entry into a 128-number index key for the indexer. c passes on to the write unchanged.",
        counted: "2 per weight per entry, 2 × 512 × 128 = 0.13 MFLOP an entry (vLLM runs W_k over every token and skips the rows that close no pair).",
        operands: (scale) => ({
          input: [operand("c", [...entryRows(scale), headDim], "BF16")],
          weight: [operand("W_k", [headDim, indexDim], "BF16")],
          output: [operand("k_c", [...entryRows(scale), indexDim], "BF16")],
          flops: (rowCount(scale) / RATIO) * 2 * HEAD_DIM * INDEX_DIM,
          summed: ["d"],
        }),
      },
      {
        key: "index",
        title: "Index scores",
        role: "core",
        lane: "indexer",
        edgeIn: "q_I",
        formula: "Σ_h w_h · ReLU(q_I,h · k_I)",
        does: {
          prefill: "Scores every compressed entry each query may see. Each of the 32 index heads scores the entry, and w sums the heads' scores into one number per entry.",
          decode: "Scores every compressed entry of the request. Each of the 32 index heads scores the entry, and w sums the heads' scores into one number per entry.",
        },
        counted: "2 × 32 × 128 = 8.2 KFLOP per query–entry pair (a query at position j sees j/m entries). The scores stay on chip.",
        operands: (scale) => ({
          input: [operand("q_I", [...rows(scale), indexHeads, indexDim], "FP4", { scale: fp4Key }), operand("w", [...rows(scale), indexHeads], "BF16")],
          cache: [operand("k_I", [...entryCache(scale), indexDim], "FP4", { scale: fp4Key })],
          output: [operand("I", [...rows(scale), entryAxis(scale)], "FP32", { onChip: true })],
          flops: 2 * INDEX_HEADS * INDEX_DIM * indexPairs(scale),
          summed: ["d_I", "H_I"],
        }),
      },
      {
        key: "topk",
        title: "Top-k",
        role: "elementwise",
        lane: "indexer",
        edgeIn: "scores",
        formula: "top 512",
        does: {
          prefill: "For each query, keeps the positions of its 512 highest-scoring entries. A query that sees 512 entries or fewer keeps them all.",
          decode: "Keeps the positions of the 512 highest-scoring entries. Layers 3–7 have no indexer and reuse this selection.",
        },
        counted: "a selection, not arithmetic: about 1 comparison per score, counted as 1 FLOP. The scores come on chip from the index kernel; only the positions go to HBM.",
        operands: (scale) => ({
          input: [operand("I", [...rows(scale), entryAxis(scale)], "FP32", { onChip: true })],
          output: [operand("idx", [...rows(scale), selected(scale)], "INT32")],
          flops: indexPairs(scale),
          summed: [],
        }),
      },
      {
        key: "attend",
        title: "Attention",
        fullTitle: "Attention over the window and the selected entries",
        role: "core",
        edgeIn: "q",
        formula: "softmax(q · Kᵀ ÷ √512) · K,  K = 128 window keys + 512 selected entries;  then rotate o back",
        does: {
          prefill: "Each query attends to the last 128 tokens and to its 512 selected compressed entries. Every entry serves as both key and value, for all 64 heads. The values carry the keys' rotation, so the kernel turns the last 64 channels of each head's output back (inverse RoPE) and casts it to FP8.",
          decode: "The new token attends to the last 128 tokens, itself included, and to the 512 selected compressed entries. Every entry serves as both key and value, for all 64 heads. The values carry the keys' rotation, so the kernel turns the last 64 channels of each head's output back (inverse RoPE) and casts it to FP8.",
        },
        counted: "2 × 64 × (512 + 512) = 131 KFLOP per query–key pair (scores and weighted sum); a query at position j sees min(j, 128) window keys and min(j/m, 512) entries. Prefill: each query gathers its own entries; read once each, the chunk touches at most all (L+N)/m of them. Scores and weights stay on chip.",
        operands: (scale) => ({
          input: [operand("q", [...rows(scale), heads, headDim], "BF16"), operand("idx", [...rows(scale), selected(scale)], "INT32")],
          cache: [operand("kv_w", [...windowCache(scale), headDim], "MXFP8"), operand("c", [...selectedCache(scale), headDim], "NVFP4")],
          output: [operand("o", [...rows(scale), heads, headDim], "FP8")],
          flops: 2 * HEADS * 2 * HEAD_DIM * attentionPairs(scale),
          summed: ["d", "W", "k"],
        }),
      },
      {
        key: "out",
        title: "Output proj.",
        fullTitle: "Output projection",
        role: "weights",
        edgeIn: "o",
        formula: "o · W_oa (8 groups) · W_ob",
        does: "Maps each group of 8 heads to 1,024 numbers, then the 8,192 results back to 5,120.",
        counted: "two GEMMs, 2 × 32,768 × 1,024 + 2 × 8,192 × 5,120 = 67.1 + 83.9 = 151 MFLOP a token; the 8,192-number middle result is not listed.",
        operands: (scale) => ({
          input: [operand("o", [...rows(scale), axis("H·d", HEADS * HEAD_DIM)], "FP8")],
          weight: [
            operand("W_oa", [axis("G", O_GROUPS), axis("H·d/G", (HEADS * HEAD_DIM) / O_GROUPS), axis("r_o", O_LATENT)], "FP8", { scale: fp8Weight }),
            operand("W_ob", [axis("G·r_o", O_GROUPS * O_LATENT), hidden], "FP8", { scale: fp8Weight }),
          ],
          output: [operand("out", [...rows(scale), hidden], "BF16")],
          flops: 2 * rowCount(scale) * (HEADS * HEAD_DIM * O_LATENT + O_GROUPS * O_LATENT * HIDDEN),
          summed: ["H·d/G", "G·r_o"],
        }),
      },
    ],
    branch: {
      title: "Write caches",
      role: "write",
      formula: "rotate, quantize, append: kv to the window;  every 2 tokens, c and RMSNorm(k_c)",
      does: {
        prefill: "Rotates the chunk's kv and keeps its last 128 tokens in the window. Rotates the chunk's compressed entries, normalizes and rotates their index keys, and appends both to their caches. vLLM uses three kernels: the window insert, rope_quant_insert for c, and indexer_k_norm_rope_store for k_c.",
        decode: "Rotates this token's kv and stores it in the window, dropping the token 128 positions back. Every second token, it also stores one compressed entry and its normalized index key.",
      },
      counted: "about 1 per number to quantize (MXFP8, NVFP4 or FP4), 3 per rotated number (the last 64 channels of each vector) and 4 per number for the index key's RMSNorm.",
      operands: (scale) => ({
        input: [operand("kv", [...windowWrites(scale), headDim], "BF16"), operand("c", [...entryRows(scale), headDim], "BF16"), operand("k_c", [...entryRows(scale), indexDim], "BF16")],
        output: [operand("kv_w", [...windowWrites(scale), headDim], "MXFP8"), operand("c", [...entryRows(scale), headDim], "NVFP4"), operand("k_I", [...entryRows(scale), indexDim], "FP4", { scale: fp4Key })],
        flops:
          (isDecode(scale) ? scale.batch : Math.min(scale.chunk, WINDOW)) * (HEAD_DIM + 3 * ROPE) +
          (rowCount(scale) / RATIO) * (HEAD_DIM + INDEX_DIM + 2 * 3 * ROPE + 4 * INDEX_DIM),
        summed: [],
      }),
      from: [
        { step: "indexKey", label: "c, k_c" },
        { step: "rope", label: "kv" },
      ],
    },
    cache: {
      title: "Caches of layer 2; layers 3–7 read its compressed rows",
      // indexer keys first: the indexer, the leftmost reader, must not cross the other reads
      // (decode: L counts this token too, so the earlier columns end one token sooner)
      rows: [
        {
          label: "index keys",
          cells: { prefill: ["t0–1", "t2–3", "t4–5", "…", "t(L−2)–(L−1)"], decode: ["t0–1", "t2–3", "t4–5", "…", "t(L−3)–(L−2)"] },
          newCell: { prefill: "N / 2", decode: "every 2nd" },
        },
        {
          label: "compressed",
          cells: { prefill: ["t0–1", "t2–3", "t4–5", "…", "t(L−2)–(L−1)"], decode: ["t0–1", "t2–3", "t4–5", "…", "t(L−3)–(L−2)"] },
          newCell: { prefill: "N / 2", decode: "every 2nd" },
        },
        {
          label: "window",
          cells: {
            prefill: [{ label: "t0", dropped: true }, { label: "…", dropped: true }, "t(L−127)", "…", "t(L−1)"],
            decode: [{ label: "t0", dropped: true }, { label: "…", dropped: true }, "t(L−128)", "…", "t(L−2)"],
          },
          newCell: { prefill: "last 128", decode: "new" },
        },
      ],
      labelWidth: 104,
      cellWidth: 88,
      // the window row's dropped columns: resolveField picks the mode's cells, so oldCells follows the first row
      groups: { old: { prefill: "L earlier tokens", decode: "L − 1 earlier tokens" }, new: { prefill: "N in this chunk", decode: "this token" } },
      note: "Per 2 tokens: one 288 B entry (NVFP4) and one 68 B index key (FP4), kept for the whole request. Window: 128 × 528 B (MXFP8)",
      reads: [
        { step: "index", row: 0, dx: 20, label: "reads all index keys" },
        { step: "attend", row: 1, dx: -12, label: "" },
        { step: "attend", row: 2, dx: 12, label: { prefill: "reads 512 entries + window per query", decode: "reads 512 entries + 128 in window" } },
      ],
    },
  };
})();
