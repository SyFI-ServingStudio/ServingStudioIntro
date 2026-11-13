/* Model data of the interactive post (the file name is historical: nothing in
   it is a placeholder any more). viz.js reads the storage parts below for the
   kv_storage, kv_growth and decode_bytes figures; the measured times
   (measured_data.js) and the prefill FLOPs (prefill_flops_data.js) are separate
   generated files keyed by the same model names.

   Attention runs data-parallel (DP attention, `tp` 1 for every model): each
   GPU runs all of a model's attention heads and a request sits on one GPU, so
   one copy of a request's state is also what its GPU holds and reads.

   Per model, storage parts, one copy of the model's own bytes: `perToken`
   bytes for every context token, or only for the last `window` tokens, plus
   `fixed` bytes per request. For decode_bytes (viz.js decodeReadBytes), a part may also set
   `tokensPerEntry` (a compressed entry covers that many tokens), `readTopk`
   (a sparse layer reads only that many entries of the cache) and `readBytes`
   (bytes a decode step reads per entry, summed over every layer that reads the
   part; default: the entry's stored bytes).

   The parts reproduce blog/data/kv_layouts.json part for part (one copy, no
   padding): storage totals equal blog/scripts/kv_storage.py. Both follow the
   release's precision rule (kv_layouts.json `precision_rule`: FP8 caches unless
   the model specifies FP4, FP32 linear states). decodeReadBytes follows the
   rule of tmp/attn/new-models/compute_new_models.py (decode_read_per_gpu, with
   V4.1 read layer by layer as v41_decode_read) at this precision rule and at
   TP1, i.e. one request's whole read. Kimi K3.1 Pro's MLA cache is vLLM's
   fp8_ds_mla row, 656 B a layer (512 FP8 NoPE values, 4 FP32 scales, 64 BF16
   RoPE values), the layout its FlashMLA sparse backend runs on at 96 heads;
   Kimi K3's dense MLA keeps plain FP8, 576 B. MiniMax-M3.1 Flash is counted in FP8
   (K/V and index keys), although vLLM's Blackwell recipe serves it in nvfp4. Inkling's
   K/V is counted in FP8 as well (its checkpoint serves BF16); its short-conv state is an
   activation window, kept BF16 as served, like the KDA short-conv state inside GLM-5.3-Flash's
   KDA state. DeepSeek V4 Pro is counted as vLLM stores it, which already is FP8
   (fp8_ds_mla rows and FP8 index keys); its compressor states stay FP32, like V4.1's
   compressor ring. */

window.ATTN_BLOG_DATA = {
  tp: 1, // DP attention: no model splits its heads over GPUs
  gpu: "B200",
  models: [
    {
      name: "Qwen3-Coder-480B-A35B",
      baseline: true,
      maxContext: 262144,
      storage: [
        { kind: "cache", label: "GQA K/V, 62 layers", perToken: 126976 },
      ],
    },
    {
      name: "DeepSeek V4.1 Flash",
      maxContext: 1048576,
      storage: [
        // Writer layers 2, 8, 14 (one entry per 2 tokens) and 20 (one per token), 288 B an entry.
        // A decode step: layers 2-19 read 512 entries of the first cache, 20-39 of the second;
        // layers 2, 8, 14 score the whole first index, 20, 24, 28, 32, 36 the whole second one.
        { kind: "cache", label: "Compressed cache, layers 2, 8, 14 (2 tokens an entry)", perToken: 432, tokensPerEntry: 2,
          readTopk: 512, readBytes: 18 * 288 },
        { kind: "cache", label: "Compressed cache, layer 20", perToken: 288, readTopk: 512, readBytes: 20 * 288 },
        { kind: "index", label: "Indexer keys (FP4), layers 2, 8, 14", perToken: 102, tokensPerEntry: 2 },
        { kind: "index", label: "Indexer keys (FP4), layer 20", perToken: 68, readBytes: 5 * 68 },
        { kind: "window", label: "128-token windows, 40 layers", perToken: 21120, window: 128 },
        { kind: "state", label: "Compressor ring", fixed: 98304 },
      ],
    },
    {
      name: "DeepSeek V4 Pro",
      maxContext: 1048576,
      storage: [
        // kv_layouts.json parts, already FP8 as vLLM stores them (584 B fp8_ds_mla rows, 132 B index keys).
        // A decode step: each CSA layer reads its top-1024 compressed entries, the whole index and the
        // window; each HCA layer every compressed entry (no indexer) and the window.
        { kind: "cache", label: "CSA compressed cache, 30 layers (4 tokens an entry)", perToken: 4380, tokensPerEntry: 4,
          readTopk: 1024 },
        { kind: "cache", label: "HCA compressed cache, 31 layers (128 tokens an entry)", perToken: 18104 / 128,
          tokensPerEntry: 128 },
        { kind: "index", label: "Indexer keys, 30 CSA layers", perToken: 990, tokensPerEntry: 4 },
        { kind: "window", label: "128-token windows, 61 layers", perToken: 35624, window: 128 },
        // c4_compressor_state (8 tokens x 307,200 B) + c128_compressor_state (128 x 126,976 B), FP32 as served.
        { kind: "state", label: "Compressor state (FP32)", fixed: 8 * 307200 + 128 * 126976 },
      ],
    },
    {
      name: "GLM-5.3-Flash",
      maxContext: 1048576,
      storage: [
        { kind: "cache", label: "MLA cache, 11 DSA layers", perToken: 5632, readTopk: 2048 },
        { kind: "index", label: "Pooled index (1 key / 4 tokens)", perToken: 363 },
        { kind: "state", label: "KDA state, 34 layers", fixed: 147619840 },
        { kind: "state", label: "Pooled-index tail (last partial block)", fixed: 22528 },
      ],
    },
    {
      name: "Qwen3.8 Flash Next",
      maxContext: 1048576,
      storage: [
        // The 12 QSA layers only, without the MTP draft layer, and no speculative draft
        // tokens in the GDN / PLE conv windows or the ring (as in kv_layouts.json).
        { kind: "cache", label: "QSA K/V, 12 layers", perToken: 12288, readTopk: 2048 },
        { kind: "index", label: "Block index (1 key / 4 tokens)", perToken: 384 },
        { kind: "state", label: "QSA ring", fixed: 12288 },
        { kind: "state", label: "GDN state, 36 layers", fixed: 115458048 },
        { kind: "state", label: "PLE conv state", fixed: 184320 },
      ],
    },
    {
      name: "MiMo V2.6 Flash",
      maxContext: 1048576,
      storage: [
        { kind: "cache", label: "Global K/V, 9 layers", perToken: 11520 },
        { kind: "window", label: "128-token windows, 39 layers", perToken: 99840, window: 128 },
      ],
    },
    {
      // Simulator kv_cache_layout (arch inkling_vllm_nvfp4_rel_attn_moe), K/V halved from BF16 to FP8:
      // global 11 layers x 8 KV heads x (128 + 128); local 55 layers x 16 KV heads x (128 + 128) over the
      // last 512 tokens. The short-conv state keeps the last 4 tokens' pre-conv K, V and attention / MLP
      // outputs of all 66 layers (2,117,632 B a token, BF16, unpadded), read and written every step.
      name: "Inkling",
      maxContext: 1048576,
      storage: [
        { kind: "cache", label: "Global K/V, 11 layers", perToken: 22528 },
        { kind: "window", label: "512-token windows, 55 layers", perToken: 225280, window: 512 },
        { kind: "state", label: "Short-conv state (BF16), 66 layers", fixed: 8470528 },
      ],
    },
    {
      name: "GLM-5.3",
      maxContext: 1048576,
      storage: [
        { kind: "cache", label: "MLA cache, 78 DSA layers", perToken: 44928, readTopk: 2048 },
        { kind: "index", label: "Indexer keys, 21 indexing layers", perToken: 2772 },
      ],
    },
    {
      // 60 MSA layers, FP8 (vLLM's Blackwell recipe serves nvfp4; the precision rule counts FP8).
      // K/V: 4 KV heads x 128 x 2 x 60 layers. A decode step reads every index key and, per KV group,
      // its 16 selected 128-token blocks; the 4 groups select on their own, so the K/V read is 2,048
      // tokens of all 4 heads (readTopk 2048 over the whole perToken row).
      name: "MiniMax-M3.1 Flash",
      maxContext: 1048576,
      storage: [
        { kind: "cache", label: "MSA K/V, 60 layers", perToken: 61440, readTopk: 2048 },
        { kind: "index", label: "Index keys, 60 layers", perToken: 7680 },
      ],
    },
    {
      // Kimi K3.1 Pro with its 24 DSA layers dense: gated MLA without an indexer (NoPE, 576 B
      // plain FP8 latent, which its FlashInfer cute-dsl decode reads).
      name: "Kimi K3",
      maxContext: 1048576,
      storage: [
        { kind: "cache", label: "MLA cache, 24 Gated MLA layers", perToken: 13824 },
        { kind: "state", label: "KDA state (FP32), 69 layers", fixed: 449372160 },
      ],
    },
    {
      // MLA cache in vLLM's fp8_ds_mla row: 24 layers x 656 B.
      name: "Kimi K3.1 Pro",
      maxContext: 1048576,
      storage: [
        { kind: "cache", label: "MLA cache (fp8_ds_mla), 24 Gated DSA layers", perToken: 15744, readTopk: 2048 },
        { kind: "index", label: "Indexer keys, 24 layers", perToken: 3168 },
        { kind: "state", label: "KDA state (FP32), 69 layers", fixed: 449372160 },
      ],
    },
    {
      // Dense, 256K native; past 256K served as YaRN (rope factor 4), as its model card describes.
      name: "Qwen3.8 27B",
      maxContext: 1048576,
      storage: [
        { kind: "cache", label: "GQA K/V, 16 layers", perToken: 32768 },
        { kind: "state", label: "GDN state, 48 layers", fixed: 153944064 },
      ],
    },
  ],
};
