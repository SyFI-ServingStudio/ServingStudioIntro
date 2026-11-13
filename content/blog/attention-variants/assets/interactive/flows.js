/* Per-variant operation flows for the "Attention Variants" section: one entry
   per variant, drawn by viz.js as a horizontally scrolling flowchart
   (`<!-- interactive: flow_<key> -->` in blog.md) with a Prefill / Decode
   switch.

   Layout contract, shared by every variant so the figures read alike:
   - `steps` run left to right along the top row, one card per operation.
     `role` picks the card's color: "weights" (a GEMM against model weights),
     "core" (a GEMM against the cache), "elementwise" (per-number math).
     `only: "prefill"` keeps a step out of the decode view. `edgeIn` labels
     the arrow that enters the card.
   - `branch` is the write into the cache, a card under the top row (or a
     list of write cards, one per thing written, so no card does two jobs;
     the first places the cache panel, each later one sits in the same row
     after its own first source). A write card stands for the write itself
     even when another kernel performs it; its `does` says which. `from`
     lists the steps whose outputs feed it (one arrow each, labeled); a
     source with `fromCard: true` (and `dx`) runs from its card's bottom
     edge even when it is the first, not from the main arrow. It
     hangs under the arrow leaving its first `from` step, or right after it
     when that step is a lane card; `place: "under"` puts it right under its
     first source card instead (under a "below" lane card: in the deep row),
     fed straight down. Later sources enter from the top or the facing side.
   - `cache` is what the request keeps between steps, drawn below; `reads`
     names the steps that read it back (one arrow each, labeled). A read may
     carry `only: "prefill"` / `"decode"` like a step; `labelWidth` widens the
     row-label column for longer row names.
   - `lanes` (optional): parallel paths beside the main row, drawn where the
     real data dependencies put them (a step that reads only an earlier
     step's output, not the main row's latest one). A step with `lane: name`
     leaves the main row and goes into that lane's row:
       { name: { side: "above" | "below" | "deep",
                 from: { step: <main step key> | "input", label },
                 at?: <main step key>, offset?: n,
                 to?: [{ step: <main step key>, label, dx? }] } }
     "below" shares the write card's row, "deep" is a row of its own under
     it, "above" shifts the main row down. A lane starts in the column after
     its `from` step (`offset` more to the right, or under `at`), so lane cards
     line up with the main steps of the same stage. The fork leaves the main
     arrow in the gap left of the lane's first card (an "above" lane leaves
     the source card's top edge; "input" leaves the input arrow); each `to`
     join runs from the lane's last card into that main step, which moves
     right of the lane if needed (leaving an empty column is fine).
     `from: { step, hidden: true }` places a lane after `step` without a
     fork, for a first card that reads only the cache (its input is the
     cache read arrow; no main-row arrow may imply one).
     Agent note: arrows must not cross. A lane whose fork is left of the
     write card's chain to the cache and whose join is right of it cannot be
     drawn without a crossing; move the write (`branch.place`) or the fork.
   - Every card on the track shows only its title and role caption; the
     track's size never depends on the data below. Clicking a card opens a
     detail card with `formula` (one line per statement: viz.js breaks it
     after each ";" and at the two-space separators ",  " and ":  ") and
     `does`, then either
     - the operands (the step or branch has `operands(scale)`; every variant
       uses them): Input /
       Weight-Cache-State / Output rows with axis-labeled shapes and dtypes,
       a cost line (FLOPs, bytes moved, intensity = FLOPs / bytes), and small
       print: the `axes` legend entries the step uses and the notes behind a
       "+s" or "reused" dtype cell. `counted` (how the FLOPs are counted) is
       not shown; it documents the `flops` field and is what
       tmp/flow-shapes/counted_check.cjs checks against it; or
     - the old text: `shape` and `cost` (rows of [label, text]).
     A flow with any operand step gets batch / context sliders (viz.js
     FLOW_SLIDERS) beside the mode switch; moving one rewrites the open
     detail in place. `axes` is a list of [label, meaning] (meaning may be
     `{ prefill, decode }`).

   `operands(scale)` contract. `scale` is what the sliders set:
   { mode: "prefill" | "decode", batch: B, chunk: N, context: L }. Decode is
   B requests, one new token each, each with L tokens in its cache (the new
   one included, so a decode step at L reads L entries, as decode_bytes
   counts them). Prefill is one request: a chunk of N tokens after L earlier
   ones. It returns
     { input: [operand], weight | cache | state: [operand], output: [operand],
       flops: number, summed: ["d"] }
   built with the ATTN_FLOW_KIT helpers below. The middle group's key names
   its group label: "weight" for model weights, "cache" for a GEMM against the
   cache, "state" for a linear-attention state. `summed` lists the axis labels
   the step reduces over; the detail underlines them. An operand's bytes are its
   element count × its dtype's size, read or written once (ideal traffic, no
   re-reads); an `onChip` operand (scores and weights inside a fused attention
   kernel) is shown as "on chip" and not counted. A multiply-add is 2 FLOPs.
   Axis labels: B requests, N chunk tokens, L cached tokens, H_q / H_kv heads,
   d head dim, D hidden size. "_x" renders as a subscript everywhere on the
   figure (titles, formulas, `does`, operand names, pills), to match the
   prose's $x_{y}$ math. Token axes (built
   with `kit.tokens`) print as 128K; others print in full (6,144).

   Optional extensions (each variant's flows_<key>.js uses them):
   - `note` on the flow: one line after the click hint, e.g. what the costs
     cover. `caption` on a step or the branch replaces its role caption
     ("GEMM · state" instead of "GEMM · cache").
   - `cache.cells` replaces the earlier-token columns (t0 … t(L−1)); a cell is
     a string or `{ label, dropped: true }` for a slot a window no longer keeps
     (drawn empty and dashed). With `groups.dropped`, the leading dropped
     columns get their own bracket and `groups.old` covers the rest.
     `cache.cellWidth` widens every cell for longer labels.
   - A `cache.rows` entry may be an object instead of a name: `{ label,
     cells, newCell }` gives that row its own labeled cells and new-cell
     text (a compressed row of token pairs next to a window row), and
     `{ label, state }` draws one fixed-size block that never grows (a
     linear-attention state); `groups.state` then labels one bracket under
     the whole block, and a cache of states only keeps one width in both modes.
   - A read may carry `dx` (shift its arrow off the card's center, so two
     reads can enter one card), `dy` (shift it within the row) and an empty
     `label` (arrow only). `cache.writes` lists steps that write a row
     themselves rather than through the write card ({ step, row, dx, dy,
     label }); the arrow is a read's path reversed. A read rising straight
     up may take `labelSide: "right"` and `labelDy` to move its pill.
   - `cache.newUnder: { step, dx, writeDx }` hangs the newest column under a
     step that reads it instead of under the write card, so the panel sits
     under its readers; the write arrow turns along the gap above the panel
     and drops `writeDx` from the column's center (left of any read that
     rises from the same column, or they cross).
   - `branch.from` entries may give `step` and `label` as `{ prefill,
     decode }` when the card that feeds the write differs by mode.
   - A later `branch.from` entry left of the write card may set
     `labelSide: "left"` to print its label left of its drop, when the run
     into the card is too short for it.
   - `edges` (optional): a tensor that skips from a card (or the input) to
     a later card along no lane, e.g. an indexer weight made from x:
       [{ from: <step> | "input", to: <step>, label, level: "over" | "below" |
          "deep" | "underBelow", dy, fromDx, enter?: "below", dx, labelDx,
          only?: "prefill" | "decode" }]
     It drops from the source to that level, runs right and enters the
     target's left side (or its bottom, `dx` off center). "over" instead
     leaves the source's top edge and enters the target's top, `dy` above
     the main row (in the gap under an "above" lane, or a gutter added for
     it); nest two over edges with the longer one higher. An over edge hops
     each join that drops from an "above" lane across its path. Draw every tensor
     on its own edge; never let one ride another edge's label. Where it
     must pass the write card's drop into the cache it hops over it.
   - `cache.layout: "blocks"` draws a fixed-size state as shapes side by side
     instead of token rows: `cache.blocks` is a list of
     { key, kind: "columns" | "matrix", under: <step>, dx, caption,
       captionWidth } (`under` may differ by mode) — "columns" (a short
     input window: `cells` one label per tall column, oldest left, `inLabel`
     for the dashed slot the newest enters, `outLabel` for the oldest
     leaving; `slotWidth`, `slotText` and `keepLast` widen the slot for a
     whole chunk, name it, and mark its last inputs, the next state, as
     strips) or "matrix" (one square per
     head, a few drawn stacked: `inner`, `dims`). Each block hangs under the
     step that reads it; `reads` are { step, block, dx, label, labelSide }
     and rise straight up, `writes` are { block, branch, dx, label } and drop
     from write card number `branch` (default 0) into the block (a columns
     block: into its dashed slot). A read whose
     step sits above the panel rises straight from the row's top edge, so only
     the first row should be read that way; one left of the panel leaves the
     row's left end.
   Any text field may be `{ prefill, decode }` when the two modes differ, or a
   function of `scale` when it follows the sliders.
   Numbers are per layer, from the model's config.json (main/model/config/).
   Keep descriptions to what the operation does; the prose explains why. */

/* Helpers for `operands(scale)`; every flows_<key>.js uses them. */
window.ATTN_FLOW_KIT = (() => {
  // Bytes per number. NVFP4 and MXFP8 carry their block scales in the number
  // (one FP8 scale per 16 numbers, one 1-byte scale per 32); other per-block
  // scales go in an operand's `scale` option instead.
  const DTYPE_BYTES = { FP32: 4, INT32: 4, BF16: 2, FP8: 1, MXFP8: 1 + 1 / 32, FP4: 0.5, NVFP4: 0.5 + 1 / 16 };
  const axis = (label, size) => ({ label, size });
  // a token-count axis (B, N, L, L+N): printed compactly, 128K rather than 131,072
  const tokens = (label, size) => ({ label, size, tokens: true });
  // options: onChip (stays in the fused kernel, not counted), reused (loaded by
  // the previous step of the same fused kernel, counted there), scale:
  // { every, bytes, note } (one scale of `bytes` per `every` numbers, counted;
  // the dtype cell shows "+s" and the small print shows `note`)
  const operand = (name, axes, dtype, options = {}) => ({ name, axes, dtype, ...options });
  const elements = (entry) => entry.axes.reduce((product, dimension) => product * dimension.size, 1);
  const bytes = (entry) => {
    if (entry.onChip || entry.reused) return 0;
    const count = elements(entry);
    return count * DTYPE_BYTES[entry.dtype] + (entry.scale ? (count / entry.scale.every) * entry.scale.bytes : 0);
  };
  const dtypeNotes = (entry) => {
    const notes = [];
    if (entry.onChip) return notes;
    if (entry.reused) notes.push("reused: the fused kernel already loaded it for the step before; counted there.");
    else if (entry.scale?.note) notes.push(`+s: ${entry.scale.note}`);
    return notes;
  };
  const isDecode = (scale) => scale.mode === "decode";
  // leading axes of a per-token activation: B requests (decode) or N chunk tokens (prefill)
  const rows = (scale) => (isDecode(scale) ? [tokens("B", scale.batch)] : [tokens("N", scale.chunk)]);
  const rowCount = (scale) => (isDecode(scale) ? scale.batch : scale.chunk);
  // leading axes of a full cache read: B requests × L entries, or the L + N entries of one request
  const cacheRows = (scale) =>
    isDecode(scale) ? [tokens("B", scale.batch), tokens("L", scale.context)] : [tokens("L+N", scale.context + scale.chunk)];
  // the key axis of one query's scores: L (decode) or L + N (prefill)
  const keyAxis = (scale) => (isDecode(scale) ? tokens("L", scale.context) : tokens("L+N", scale.context + scale.chunk));
  // Query–key pairs: a query at position j (1-based) sees min(floor(j / perEntry), cap)
  // keys. Decode: B queries at j = L. Prefill: the chunk's queries j = L+1 .. L+N.
  // perEntry > 1 counts compressed entries (one per perEntry tokens).
  const cappedPairs = (scale, cap = Infinity, perEntry = 1) => {
    const seen = (position) => Math.min(Math.floor(position / perEntry), cap);
    if (isDecode(scale)) return scale.batch * seen(scale.context);
    if (cap === Infinity && perEntry === 1) return scale.chunk * scale.context + (scale.chunk * (scale.chunk + 1)) / 2;
    let pairs = 0;
    for (let position = scale.context + 1; position <= scale.context + scale.chunk; position++) pairs += seen(position);
    return pairs;
  };
  // query–key pairs a causal layer computes: B × L in decode; in prefill each
  // chunk token sees the L earlier tokens and the chunk tokens up to itself
  const causalPairs = (scale) => cappedPairs(scale);
  return { DTYPE_BYTES, axis, tokens, operand, elements, bytes, dtypeNotes, isDecode, rows, rowCount, cacheRows, keyAxis, cappedPairs, causalPairs };
})();

/* GQA: one layer of Qwen3-Coder-480B-A35B (main/model/config/qwen3_coder_480b.json,
   the config.json of Qwen/Qwen3-Coder-480B-A35B-Instruct): hidden 6,144, 96 query heads
   and 8 KV heads of 128, 62 layers. dtypes: that config has torch_dtype bfloat16 and no
   quantization_config, so the released weights are BF16 (Qwen also publishes a separate
   -FP8 checkpoint; the blog's baseline is the BF16 one). Activations are BF16; the KV
   cache is FP8, as everywhere in the post. */
(() => {
  const kit = window.ATTN_FLOW_KIT;
  const { axis, operand, rows, rowCount, cacheRows, keyAxis, causalPairs } = kit;
  const HIDDEN = 6144;
  const Q_HEADS = 96;
  const KV_HEADS = 8;
  const HEAD_DIM = 128;
  const hidden = axis("D", HIDDEN);
  const headDim = axis("d", HEAD_DIM);
  const qHeads = axis("H_q", Q_HEADS);
  const kvHeads = axis("H_kv", KV_HEADS);
  const qWidth = axis("H_q·d", Q_HEADS * HEAD_DIM);
  const kvWidth = axis("H_kv·d", KV_HEADS * HEAD_DIM);
  // per-token q, k, v as the projection writes them
  const q = (scale) => operand("q", [...rows(scale), qHeads, headDim], "BF16");
  const k = (scale) => operand("k", [...rows(scale), kvHeads, headDim], "BF16");
  const v = (scale) => operand("v", [...rows(scale), kvHeads, headDim], "BF16");
  // scores and softmax weights stay on chip inside the fused attention kernel
  const scores = (scale, name) => operand(name, [...rows(scale), qHeads, keyAxis(scale)], "FP32", { onChip: true });
  const keyLabel = (scale) => keyAxis(scale).label;

  window.ATTN_BLOG_FLOWS = {
    gqa: {
      title: "GQA in one layer of Qwen3-Coder-480B-A35B",
      axes: [
        ["B", "requests"],
        ["N", "tokens in the chunk"],
        ["L", { decode: "tokens in each request's cache, the new one included", prefill: "earlier tokens in the cache" }],
        ["H_q", "query heads"],
        ["H_kv", "KV heads"],
        ["d", "head dim"],
        ["D", "hidden size"],
      ],
      input: { label: "x", note: (scale) => (scale.mode === "decode" ? `B × 6,144` : `N × 6,144`) },
      output: { label: "out", note: "to the MoE FFN" },
      steps: [
        {
          key: "project",
          title: "Projection",
          role: "weights",
          formula: "x · W_q, W_k, W_v",
          does: "Turns each token's hidden state into a query, a key and a value.",
          counted: "2 per weight per token, 2 × 6,144 × 14,336 = 176 MFLOP a token.",
          operands: (scale) => ({
            input: [operand("x", [...rows(scale), hidden], "BF16")],
            weight: [operand("W_q", [hidden, qWidth], "BF16"), operand("W_k", [hidden, kvWidth], "BF16"), operand("W_v", [hidden, kvWidth], "BF16")],
            output: [q(scale), k(scale), v(scale)],
            flops: 2 * rowCount(scale) * HIDDEN * (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM,
            summed: ["D"],
          }),
        },
        {
          key: "norm",
          title: "QK-norm",
          role: "elementwise",
          edgeIn: "q, k",
          formula: "q, k ÷ rms × γ",
          does: "Rescales each head's q and k to the same size, so scores stay in range.",
          counted: "about 4 per number (square and add for the rms, then two multiplies).",
          operands: (scale) => ({
            input: [q(scale), k(scale)],
            weight: [operand("γ_q", [headDim], "BF16"), operand("γ_k", [headDim], "BF16")],
            output: [q(scale), k(scale)],
            flops: 4 * rowCount(scale) * (Q_HEADS + KV_HEADS) * HEAD_DIM,
            summed: ["d"],
          }),
        },
        {
          key: "rope",
          title: "RoPE",
          role: "elementwise",
          edgeIn: "q, k",
          formula: "rotate by position",
          does: "Turns each pair of dimensions by an angle set by the token's position. cos, sin hold 64 values each, looked up by position.",
          counted: "about 3 per number (each pair takes 4 multiplies and 2 adds).",
          operands: (scale) => ({
            input: [q(scale), k(scale), operand("cos, sin", [...rows(scale), headDim], "BF16")],
            output: [q(scale), k(scale)],
            flops: 3 * rowCount(scale) * (Q_HEADS + KV_HEADS) * HEAD_DIM,
            summed: [],
          }),
        },
        {
          key: "score",
          title: "Scores",
          role: "core",
          edgeIn: "q",
          formula: "q · Kᵀ ÷ √128",
          does: {
            prefill: "Compares every query in the chunk with the keys of the L earlier tokens and of the chunk's own tokens. Each KV head serves 12 query heads.",
            decode: "Compares q with the key of every token in the cache. Each KV head serves 12 query heads.",
          },
          counted: {
            prefill: "2 × 96 × 128 = 24.6 KFLOP per query–key pair, over the N × L + N(N+1)/2 pairs the causal mask keeps.",
            decode: "2 × 96 × 128 = 24.6 KFLOP per query–key pair, B × L pairs.",
          },
          operands: (scale) => ({
            input: [q(scale)],
            cache: [operand("K", [...cacheRows(scale), kvHeads, headDim], "FP8")],
            output: [scores(scale, "S")],
            flops: 2 * Q_HEADS * HEAD_DIM * causalPairs(scale),
            summed: ["d"],
          }),
        },
        {
          key: "mask",
          title: "Causal mask",
          role: "elementwise",
          only: "prefill",
          formula: "score = −∞ where key > query",
          does: "Hides the tokens that come later in the chunk, so each token sees only itself and earlier ones.",
          counted: "none in practice: the kernel skips the hidden pairs instead of computing them.",
          operands: (scale) => ({
            input: [scores(scale, "S")],
            output: [scores(scale, "S")],
            flops: 0,
            summed: [],
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
            flops: 5 * Q_HEADS * causalPairs(scale),
            summed: [keyLabel(scale)],
          }),
        },
        {
          key: "sum",
          title: "Weighted sum",
          role: "core",
          formula: "P · V",
          does: "Adds up the values of the tokens it may see, each times its weight.",
          counted: "2 × 96 × 128 = 24.6 KFLOP per query–key pair, as many pairs as the scores.",
          operands: (scale) => ({
            input: [scores(scale, "P")],
            cache: [operand("V", [...cacheRows(scale), kvHeads, headDim], "FP8")],
            output: [operand("o", [...rows(scale), qHeads, headDim], "BF16")],
            flops: 2 * Q_HEADS * HEAD_DIM * causalPairs(scale),
            summed: [keyLabel(scale)],
          }),
        },
        {
          key: "out",
          title: "Output proj.",
          fullTitle: "Output projection",
          role: "weights",
          edgeIn: "o",
          formula: "concat(o) · W_o",
          does: "Joins the 96 heads into 12,288 numbers and maps them back to 6,144.",
          counted: "2 per weight per token, 2 × 12,288 × 6,144 = 151 MFLOP a token.",
          operands: (scale) => ({
            input: [operand("o", [...rows(scale), qWidth], "BF16")],
            weight: [operand("W_o", [qWidth, hidden], "BF16")],
            output: [operand("out", [...rows(scale), hidden], "BF16")],
            flops: 2 * rowCount(scale) * Q_HEADS * HEAD_DIM * HIDDEN,
            summed: ["H_q·d"],
          }),
        },
      ],
      branch: {
        title: "Write K, V",
        role: "write",
        formula: "quantize, append",
        does: { prefill: "Stores the chunk's keys and values in FP8, so later chunks and tokens can read them.", decode: "Stores this token's k and v in FP8, so every later token can read them." },
        counted: "about 1 per number (one multiply by the FP8 scale).",
        operands: (scale) => ({
          input: [k(scale), v(scale)],
          output: [operand("K", [...rows(scale), kvHeads, headDim], "FP8"), operand("V", [...rows(scale), kvHeads, headDim], "FP8")],
          flops: 2 * rowCount(scale) * KV_HEADS * HEAD_DIM,
          summed: [],
        }),
        from: [
          { step: "rope", label: "k" },
          { step: "project", label: "v (no norm, no RoPE)" },
        ],
      },
      cache: {
        title: "KV cache, kept for the whole request",
        rows: ["K", "V"],
        newLabel: { prefill: "chunk", decode: "new" },
        // decode: L counts this token too, so the earlier columns end at t(L−2)
        cells: { prefill: undefined, decode: ["t0", "t1", "t2", "…", "t(L−2)"] },
        // group labels over the cells: the earlier tokens, and what this step adds
        groups: { old: { prefill: "L earlier tokens", decode: "L − 1 earlier tokens" }, new: { prefill: "N in this chunk", decode: "this token" } },
        note: "8 KV heads × 128 × (K + V) = 2 KiB per token and layer in FP8; 124 KiB per token over 62 layers",
        reads: [
          { step: "score", row: 0, label: { prefill: "reads K: L earlier + N in chunk", decode: "reads K: all L tokens" } },
          { step: "sum", row: 1, label: { prefill: "reads V: L earlier + N in chunk", decode: "reads V: all L tokens" } },
        ],
      },
    },
  };
})();
