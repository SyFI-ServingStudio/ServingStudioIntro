import { Activity, ScanSearch, Code2, BadgeCheck } from "lucide-react";

export const qwenRounds = [
  {
    request:
      "Build Qwen3-235B support in Mini-SGLang. Use ServingStudio Sim to guide the implementation.",
    steps: [
      {
        icon: Code2,
        title: "Start with a working MoE implementation",
        body: "Mini-SGLang did not support MoE models, so I added expert routing, expert-parallel execution, and FP8 expert kernels. I connected them to the model loader, attention, KV cache, and serving runtime. This produced a working implementation that we could measure against ServingStudio Sim.",
      },
      {
        icon: Activity,
        title: "Create a practical kernel testbed",
        body: "I also built a reduced model with one real transformer layer, plus the embedding and output head. It used the production checkpoint and execution path, so the kernels retained their production input shapes. This let me test changes quickly while reserving serving throughput and full-model latency measurements for all 94 layers.",
      },
    ],
    finding:
      "We now have a working serving path and a smaller version for kernel experiments. The next step is to find where the implementation differs from the simulation.",
  },
  {
    request:
      "Compare the implementation with ServingStudio Sim. What explains the largest timing differences?",
    steps: [
      {
        icon: ScanSearch,
        title: "Attention and MoE communication stand out",
        body: "I profiled the one-layer model with a 16K-token prefill. Attention took 6.237 ms, compared with 1.227 ms in the prediction. MoE dispatch and combine were also slower. These gaps pointed to differences in the execution paths, so I checked the backends and communication operations used by each implementation.",
        chart: "kernels",
      },
      {
        icon: Code2,
        title: "Use the attention backend assumed by the simulation",
        body: "Mini-SGLang was using FA2 with BF16 queries and FP8 KV, while the simulation assumed FA3 with FP8 queries and KV. Switching to FA3 alone failed because that backend did not accept the existing dtype combination. After I added FP8 query conversion, the attention kernel averaged 1.313 ms. The full one-layer forward, including conversion and surrounding work, fell from about 21.9 to 17.2 ms.",
      },
      {
        icon: ScanSearch,
        title: "Avoid sending hidden states that every rank already has",
        body: "I initially reduced MoE traffic by sending each token once per destination rank. A closer look showed that tensor parallelism for attention had already replicated the hidden states across all four ranks, making the extra dispatch unnecessary. I replaced it with FlashInfer's native expert filtering: each rank runs only its local experts, followed by an all-reduce. Kernel launches fell from 150 to 55 per rank in the one-layer trial. I also updated the simulator to represent the new path.",
        chart: "after",
      },
    ],
    finding:
      "The comparison exposed two concrete opportunities: use FA3 with the correct input types, and remove redundant MoE communication. With those changes in place, we can look for smaller costs that repeat across the full model.",
  },
  {
    request:
      "Focus on the prefill-heavy workload. Use the kernel analysis to decide what to optimize next, and test the changes in the real implementation.",
    steps: [
      {
        icon: Activity,
        title: "Quantization costs nearly as much as some GEMMs",
        body: "I compared kernel times using the model's measured expert assignments. MoE gate/up quantization accounted for 12.65% of prefill time, compared with 15.08% for its GEMM. Across the four input-quantization operations below, quantization consumed 19.8% of prefill time. This made fusion the next target: the preceding kernels could produce FP8 inputs directly instead of leaving each intermediate tensor for a separate conversion kernel.",
        chart: "quantization",
      },
      {
        icon: Code2,
        title: "Fuse quantization into the producing kernels",
        body: "I fused expert-row expansion with gate/up quantization and SiLU-and-multiply with down-projection quantization. The GEMMs also had to accept the prequantized inputs. Removing the old conversion exposed a shared-buffer aliasing bug that produced NaNs; allocating a separate FP8 input buffer fixed it. I then fused residual addition and RMSNorm with QKV input quantization.",
      },
      {
        icon: BadgeCheck,
        title: "Keep the changes that help in the full execution path",
        body: "Some kernels that looked promising in isolation did not help in the full execution path, so I discarded them. Three fusions reduced recorded full-model TTFT from 724.38 to 616.99 ms. Because the baseline later failed during decode, this comparison supports only a prefill improvement. I then moved attention-output quantization into the FA3 epilogue. In a paired test with five requests per variant, median TTFT fell from 600.17 to 596.74 ms. A final profile confirmed that the separate input-quantization kernels had disappeared from the steady layers.",
      },
    ],
    finding:
      "The useful fusions remove complete conversion passes, including the work inside the consuming GEMMs. Kernel benchmarks helped choose candidates, but production measurements determined which changes to keep.",
  },
  {
    request:
      "Now benchmark the complete 94-layer model against vLLM on the same workload.",
    steps: [
      {
        icon: BadgeCheck,
        title: "Compare the complete serving implementations",
        body: "I ran both engines on four H200 GPUs with TP4/EP4: 256 requests, each with 16,384 input tokens and 64 output tokens, at concurrency 32. Both completed every request and returned the requested number of tokens. Mini-SGLang delivered 101.406 output tokens per second, compared with 80.741 for vLLM, a 25.6% increase. I checked that the prompt-token inputs matched and captured the output-token IDs. The two FP8 implementations did not produce identical sequences, so this measures throughput rather than equivalent model quality.",
        showOutcome: true,
      },
    ],
    finding:
      "The complete Mini-SGLang implementation was faster on this prefill-heavy workload. The gain reflects the serving system as a whole; the earlier experiments show why we chose each optimization, rather than attributing the entire gain to one fusion.",
  },
];
