"""Hand-written documentation for the kernels that get a detail page.

Prototype only: the real KernelDoc should live beside each kernel module in
ServingStudio Sim (profiling/kernels/<kind>.py) so the text is reviewed with
the code it describes. Every statement here was taken from that module or its
runner.

Arg roles:
  config  fixed by the model and deployment (weights, heads, dtype, GPU count)
  sweep   varies with the batch at run time; the first sweep arg is the x axis
"""

CUPTI = (
    "GPU kernel time from CUPTI activity records, averaged over repeated launches "
    "in one capture. The L2 cache is flushed before each launch, so the numbers are "
    "cold-cache times."
)

DOCS = {
    "single_gemm": {
        "title": "Dense GEMM",
        "default_metric": "tflops",
        "summary": "One matrix multiply: an activation block of m tokens times a weight matrix.",
        "description": (
            "Every linear layer in a transformer is a dense GEMM: QKV and output "
            "projections, dense MLPs, the LM head. m is the number of tokens in the "
            "batch; n and k come from the weight shape. Backends that share this table "
            "differ in how the weight is laid out and which library dispatches the "
            "multiply, and the simulator picks the fastest measured backend per shape."
        ),
        "formula": [
            "C[m, n] = A[m, k] · B[k, n]",
            "TFLOPS = 2·m·n·k / time",
            "GB/s = (m·k + k·n + m·n) · bytes per element / time",
        ],
        "arg_docs": {
            "m": ("sweep", "tokens", "Rows of the activation: tokens in the batch."),
            "n": ("config", "elements", "Output features of the weight."),
            "k": ("config", "elements", "Input features, the reduction dimension."),
            "dtype": ("config", None, "Element type of A and B. fp8_e4m3 writes bf16 output."),
        },
        "method": CUPTI
        + " Blackwell can split one logical GEMM into overlapping launches; those are "
        "counted once, by the time the GPU is busy.",
        "caveats": [
            "Inputs are random normal tensors, so data-dependent effects such as sparsity are not measured.",
            "torch and torch_linear compute the same product. They are separate rows because the weight layout changes which cuBLAS kernel runs.",
        ],
        "backends": {
            "torch": ("torch.mm on a contiguous right-hand side.", "https://pytorch.org/docs/stable/generated/torch.mm.html"),
            "torch_linear": ("F.linear with the (n, k) weight layout used by vLLM and Transformers.", "https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html"),
            "torch_linear_vllm": ("The same F.linear call, run in vLLM's pinned environment.", "https://github.com/vllm-project/vllm"),
            "sglang_bf16_auto": ("SGLang's production BF16 dispatch with bf16_gemm_backend='auto' (CuTe DSL on SM100).", "https://github.com/sgl-project/sglang"),
            "sglang_fused_a_auto": ("SGLang's fused-A GEMM dispatch, for 16 tokens or fewer, on SM100.", "https://github.com/sgl-project/sglang"),
            "deepgemm": ("DeepGEMM FP8 dense kernel: fp8 in, bf16 out.", "https://github.com/deepseek-ai/DeepGEMM"),
        },
        "reference": {
            "note": "The torch backend is its own reference: the measured call is torch.mm(a, b).",
            "path": None,
        },
        "runner": "profiling/runners/gemm/torch.py",
    },
    "flashinfer_attn_decode": {
        "title": "Paged decode attention",
        # Decode reads the whole KV cache per token, so bandwidth is the telling metric.
        "default_metric": "memory_bandwidth_gbps",
        "summary": "One query token per request attending over its paged KV cache, for a whole decode batch.",
        "description": (
            "The attention step of decode for GQA and MHA models. Each request brings "
            "one new query token and reads its full KV cache from paged memory. The "
            "measurement uses a uniform batch: every request gets the mean KV length, "
            "total_tokens divided by batch_size."
        ),
        "formula": [
            "O = softmax(q · Kᵀ / √head_dim) · V, per request and head",
            "kv_len = max(1, total_tokens / batch_size), q_len = 1",
        ],
        "arg_docs": {
            "total_tokens": ("sweep", "tokens", "KV tokens summed over the batch."),
            "batch_size": ("sweep", "requests", "Requests decoded together."),
            "num_qo_heads": ("config", "heads", "Query and output heads on this GPU."),
            "num_kv_heads": ("config", "heads", "KV heads on this GPU."),
            "head_dim": ("config", "elements", "Dimension of each head."),
            "q_dtype": ("config", None, "Query element type."),
            "kv_dtype": ("config", None, "KV cache element type."),
            "o_dtype": ("config", None, "Output element type."),
        },
        "method": CUPTI,
        "caveats": [
            "A real batch mixes short and long requests. The uniform-length batch measured here can differ from it, most for skewed batches.",
            "fa2_cudagraph runs the same FA2 math with FlashInfer's CUDA-graph plan, which matches the split and merge launches vLLM uses in pure decode.",
        ],
        "backends": {
            "fa2": ("FlashInfer BatchDecodeWithPagedKVCacheWrapper, FA2 kernels.", "https://docs.flashinfer.ai/api/decode.html"),
            "fa2_cudagraph": ("FA2 with the CUDA-graph plan: a split main kernel plus a merge kernel.", "https://docs.flashinfer.ai/api/decode.html"),
            "fa3": ("FlashInfer FA3 kernels (Hopper).", "https://docs.flashinfer.ai/api/decode.html"),
        },
        "reference": None,
        "runner": "profiling/runners/attention/flashinfer_decode.py",
    },
    "all_reduce": {
        "title": "All-reduce",
        "default_metric": "busbw_gbps",
        "summary": "Sum one buffer across the GPUs of a tensor-parallel group.",
        "description": (
            "Tensor parallelism ends most attention and MLP blocks with an all-reduce "
            "of the block output. message_size_bytes is the full buffer each GPU "
            "contributes, not a shard. Measured inside one node over NVLink."
        ),
        "formula": [
            "algbw = message_size_bytes / time",
            "busbw = algbw · 2(N − 1) / N, for N GPUs",
        ],
        "arg_docs": {
            "message_size_bytes": ("sweep", "bytes", "Full buffer each GPU all-reduces."),
            "num_gpus": ("config", "GPUs", "GPUs in the group."),
            "dtype": ("config", None, "Element type of the buffer."),
            "fabric": ("config", None, "Interconnect the group sits on."),
        },
        "method": (
            "CUDA events on rank 0's stream around repeated all-reduces inside one live "
            "process group, divided by the repeat count. Energy is not measured for "
            "collectives."
        ),
        "caveats": [
            "busbw is the number to compare with the link: it corrects for the 2(N − 1)/N ring traffic. The NVLink figure in the GPU catalog counts both directions.",
        ],
        "backends": {
            "nccl": ("torch.distributed all_reduce on the NCCL backend.", "https://github.com/NVIDIA/nccl"),
            "nvshmem": ("nvshmem4py reduce over symmetric memory.", "https://github.com/NVIDIA/nvshmem"),
        },
        "reference": None,
        "runner": "profiling/runners/comm/nccl.py",
    },
    "residual_rms_norm": {
        "title": "Residual add + RMSNorm",
        "default_metric": "memory_bandwidth_gbps",
        "summary": "Add the residual stream, then normalize: the fused step between transformer blocks.",
        "description": (
            "Before each attention and MLP block, the model adds the previous output to "
            "the residual stream and RMS-normalizes the sum. vLLM fuses both into one "
            "kernel that returns the normalized output and the new residual. The torch "
            "backend runs the same math as separate launches."
        ),
        "formula": [
            "s = x + residual",
            "y = s / √(mean(s²) + ε) · weight",
            "returns (y, s)",
        ],
        "arg_docs": {
            "m": ("sweep", "tokens", "Tokens in the batch."),
            "hidden": ("config", "elements", "Hidden size of the model."),
            "dtype": ("config", None, "Element type of x, residual and weight."),
        },
        "method": CUPTI,
        "caveats": [
            "The sum and the RMS reduction run in fp32, whatever the input dtype.",
        ],
        "backends": {
            "vllm_cuda": ("vLLM fused_add_rms_norm CUDA kernel.", "https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu"),
            "torch": ("The same math as separate torch launches.", "https://pytorch.org/docs/stable/generated/torch.rsqrt.html"),
        },
        "reference": {
            "note": "Semantic reference the backends are checked against.",
            "path": "profiling/runners/norm/residual_rms_norm_reference.py",
        },
        "runner": "profiling/runners/norm/residual_rms_norm_vllm_cuda.py",
    },
}

# Short public titles for the catalog. The module docstrings describe the code
# for maintainers ("kernel kind", design-doc sections), which reads badly on a
# public list. A kernel missing here falls back to its docstring title.
TITLES = {
    "all_reduce_fusion": "All-reduce, FlashInfer TRT-LLM",
    "all_reduce_residual_rms_norm": "All-reduce + residual add + RMSNorm",
    "batched_gemm": "Batched GEMM",
    "bf16_fused_moe": "Fused MoE, BF16",
    "clamped_swiglu": "Clamped SwiGLU",
    "deepseek_v4_fused_inv_rope_fp8_quant": "Inverse RoPE + FP8 quantization",
    "deepseek_v4_fused_q_kv_rmsnorm": "Q and KV RMSNorm",
    "deepseek_v4_indexer_mqa_logits_decode": "Indexer logits, decode",
    "deepseek_v4_indexer_mqa_logits_prefill": "Indexer logits, prefill",
    "deepseek_v4_indexer_q_rope_quant": "Indexer query RoPE + FP8 quantization",
    "deepseek_v4_indexer_topk_decode": "Indexer top-k, decode",
    "deepseek_v4_indexer_topk_prefill": "Indexer top-k, prefill",
    "deepseek_v4_packed_cache_gather": "Packed cache gather",
    "deepseek_v4_qnorm_rope_kv_insert": "Q norm + RoPE + KV insert",
    "deepseek_v4_sparse_attn_compress_store": "Sparse-attention compress and store",
    "deepseek_v4_sparse_mla_decode": "Sparse MLA decode, FP8",
    "deepseek_v4_sparse_mla_prefill": "Sparse MLA prefill, BF16",
    "deepseek_v4_terminal_mhc_head": "Final MHC head + RMSNorm",
    "dsa_index_cache_append": "Indexer key cache append",
    "dsa_indexer_q_rope_quant": "Indexer query RoPE + FP8 quantization",
    "dsa_mqa_logits_prefill": "Indexer logits, prefill",
    "dsa_paged_mqa_logits_decode": "Indexer logits, paged decode",
    "dsa_persistent_topk_decode": "Indexer top-k, decode",
    "dsa_sparse_index_remap": "Sparse index remap",
    "dsa_sparse_mla_attention": "Sparse MLA attention",
    "dsa_sparse_mla_prefill": "Sparse MLA prefill",
    "dsa_topk_prefill": "Indexer top-k, prefill",
    "flashinfer_attn_prefill": "Causal prefill attention",
    "flashinfer_attn_rect": "Non-causal attention",
    "fp8_block_quant": "FP8 block quantization",
    "fp8_blockscale_grouped_gemm": "FP8 block-scale grouped GEMM",
    "fp8_per_token_group_quant": "FP8 per-token-group quantization",
    "gdn_causal_conv_decode": "Causal convolution, decode",
    "gdn_causal_conv_prefill": "Causal convolution, prefill",
    "gdn_chunk_delta_rule": "Chunked delta rule, one fused launch",
    "gdn_chunk_local_cumsum": "Chunk-local cumulative decay",
    "gdn_chunk_output": "Chunk output",
    "gdn_chunk_recompute_w_u": "WY recomputation",
    "gdn_chunk_scaled_dot_kkt": "Scaled dot product KKᵀ",
    "gdn_chunk_solve_tril": "Triangular solve",
    "gdn_chunk_state_update": "Chunk state update",
    "gdn_gated_rms_norm": "Gated RMSNorm",
    "gdn_prefill_post_conv": "Post-convolution preparation",
    "gdn_recurrent_decode": "Recurrent decode",
    "gemm_fp32_output": "GEMM with FP32 output",
    "grouped_gemm": "Grouped GEMM",
    "logits_topk": "Row-wise top-k",
    "mhc_fused_post_pre_rms_norm": "MHC post and pre + RMSNorm",
    "mhc_pre_rms_norm": "MHC pre + RMSNorm",
    "mla_cache_append": "MLA cache append",
    "mla_rope_quantize_fp8": "MLA RoPE + FP8 quantization",
    "moe_align_block_size": "MoE block alignment",
    "moe_alltoall": "MoE all-to-all",
    "moe_alltoall_prepare": "MoE all-to-all preparation",
    "moe_ep_all_gather": "MoE all-gather, expert parallel",
    "moe_ep_reduce_scatter": "MoE reduce-scatter, expert parallel",
    "moe_finalize_fuse_shared": "MoE finalize + shared expert",
    "moe_finalize_routing": "MoE finalize",
    "moe_fused_topk": "MoE router softmax + top-k",
    "moe_sum": "MoE top-k sum",
    "moe_topk_softplus_sqrt": "MoE router sqrt-softplus top-k",
    "mxfp4_marlin_moe_gemm": "MXFP4 MoE GEMM, Marlin",
    "nvfp4_fused_moe": "Fused MoE, NVFP4",
    "nvfp4_moe": "NVFP4 MoE",
    "nvfp4_moe_bmm": "NVFP4 MoE expert GEMM",
    "nvfp4_moe_routing": "NVFP4 MoE routing",
    "nvfp4_quant": "NVFP4 quantization",
    "p2p_inter": "Point-to-point, across NVLink domains",
    "p2p_intra": "Point-to-point, within an NVLink domain",
    "vllm_fused_moe": "Fused MoE, vLLM Triton",
    "vllm_mla_rope": "MLA query RoPE",
}
