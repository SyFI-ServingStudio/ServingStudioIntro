/* The twelve models' attention layers, for the model cards (Table 1). Facts,
   not placeholders: sizes and context lengths from each model's Hugging Face
   card, layer counts from the checkpoints' config.json (main/model/config/).
   GLM-5.3's Hugging Face card gives no active count; ~40B is from NVIDIA's
   NIM card (docs.api.nvidia.com/nim/reference/z-ai-glm-5-3).

   Each layer tag has a `family`, which picks its color and matches the
   storage figure's kinds: full attention keeps the whole context (blue), a
   sparse layer's indexer is the index (orange), a window (green), a linear
   layer's state (yellow). Each label is the layer's published name and
   nothing else: the post's later sections explain every kind, so the
   cards carry no detail lines or notes. */

window.ATTN_BLOG_MODELS = [
  {
    name: "Qwen3-Coder-480B-A35B",
    role: "Baseline",
    facts: ["480B total", "35B active", "262K context"],
    layers: [{ family: "full", label: "Grouped-query attention (GQA)", count: 62 }],
  },
  {
    name: "DeepSeek V4.1 Flash",
    facts: ["552B backbone", "8B active in prefill", "16B active in decode", "1M context"],
    layers: [
      { family: "window", label: "Sliding-window attention (SWA)", count: 40 },
      { family: "sparse", label: "Compressed Sparse Attention (CSA2)", count: 38 },
    ],
  },
  {
    // huggingface.co/deepseek-ai/DeepSeek-V4-Pro: 1.6T total, 49B activated, one million tokens of context.
    // config.json: 61 layers; compress_ratios lists 62, of which 30 are 4 (CSA) and 31 are 128 (HCA); the
    // 62nd, 0, is the MTP draft layer (num_nextn_predict_layers 1), not one of the 61. HCA attends every
    // compressed entry (no indexer), so it takes the full family; every layer also has a 128-token window.
    name: "DeepSeek V4 Pro",
    facts: ["1.6T total", "49B active", "1M context"],
    layers: [
      { family: "sparse", label: "Compressed Sparse Attention (CSA)", count: 30 },
      { family: "full", label: "Heavily Compressed Attention (HCA)", count: 31 },
    ],
  },
  {
    name: "GLM-5.3-Flash",
    facts: ["320B total", "18B active", "1M context"],
    layers: [
      { family: "linear", label: "Kimi Delta Attention (KDA)", count: 34 },
      { family: "sparse", label: "DeepSeek Sparse Attention (DSA)", count: 11 },
    ],
  },
  {
    name: "Qwen3.8 Flash Next",
    facts: ["125B total", "6B active", "51B n-gram embedding", "262K context, 1M with YaRN"],
    layers: [
      { family: "linear", label: "Gated DeltaNet (GDN)", count: 36 },
      { family: "sparse", label: "Qwen Sparse Attention (QSA)", count: 12 },
    ],
  },
  {
    name: "MiMo V2.6 Flash",
    facts: ["309B total", "15B active", "1M context"],
    layers: [
      { family: "window", label: "Sliding-window attention (SWA)", count: 39 },
      { family: "full", label: "Grouped-query attention (GQA)", count: 9 },
    ],
  },
  {
    // huggingface.co/thinkingmachines/Inkling-NVFP4: 975B total, 41B active, text / image / audio input,
    // context 1,048,576 (config text_config.model_max_length). main/model/config/inkling_nvfp4.json: 66 layers,
    // 55 in local_layer_ids (sliding window of 512 tokens, 16 KV heads) and 11 global (8 KV heads). Both
    // kinds carry a learned relative-position bias instead of RoPE; the labels name the layers only.
    name: "Inkling",
    facts: ["975B total", "41B active", "1M context"],
    layers: [
      { family: "window", label: "Sliding-window attention (SWA)", count: 55 },
      { family: "full", label: "Grouped-query attention (GQA)", count: 11 },
    ],
  },
  {
    name: "GLM-5.3",
    facts: ["753B total", "~40B active", "1M context"],
    layers: [
      { family: "sparse", label: "DeepSeek Sparse Attention (DSA)", count: 78 },
    ],
  },
  {
    // huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash (card and config.json, sha ce7f4df8): "approximately 428B
    // total; 23B activated per token", context up to 1M tokens, MiniMax Sparse Attention (MSA) throughout:
    // sparse_attention_freq is 1 on all 60 layers (MiniMax-M3 kept 3 full-attention layers).
    name: "MiniMax-M3.1 Flash",
    facts: ["~428B total", "~23B active", "1M context"],
    layers: [
      { family: "sparse", label: "MiniMax Sparse Attention (MSA)", count: 60 },
    ],
  },
  {
    name: "Kimi K3",
    facts: ["2.8T total", "104B active", "1M context"],
    layers: [
      { family: "linear", label: "Kimi Delta Attention (KDA)", count: 69 },
      { family: "full", label: "Gated MLA", count: 24 },
    ],
  },
  {
    name: "Kimi K3.1 Pro",
    facts: ["2.8T total", "104B active", "1M context"],
    layers: [
      { family: "linear", label: "Kimi Delta Attention (KDA)", count: 69 },
      { family: "sparse", label: "Gated DSA", count: 24 },
    ],
  },
  {
    // huggingface.co/Qwen/Qwen3.8-27B (the base of nvidia/Qwen3.8-27B-NVFP4): 27B parameters, a dense FFN
    // (no experts), 64 layers as 16 x (3 Gated DeltaNet + 1 Gated Attention), context 262,144 natively and
    // extensible up to 1,000,000 tokens with YaRN.
    name: "Qwen3.8 27B",
    facts: ["27B dense", "262K context, 1M with YaRN"],
    layers: [
      { family: "linear", label: "Gated DeltaNet (GDN)", count: 48 },
      { family: "full", label: "Gated attention (GQA)", count: 16 },
    ],
  },
];
