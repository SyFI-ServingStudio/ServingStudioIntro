# Attention FLOPs to prefill one prompt

All the data behind this figure in "Beyond GQA: Characterizing Modern Attention Variants with ServingStudio".

whole model, one copy · score, weighted sum, indexer scoring and linear-layer chunks; projections, softmax and top-k not counted

*Total per model by prompt length*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 25.6 TFLOP | 102 TFLOP | 409 TFLOP | 1.64 PFLOP | 6.54 PFLOP | 26.2 PFLOP | 105 PFLOP | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 3.92 TFLOP | 14.0 TFLOP | 52.8 TFLOP | 205 TFLOP | 805 TFLOP | 3.19 PFLOP | 12.7 PFLOP | 50.8 PFLOP | 203 PFLOP |
| Inkling | 6.57 TFLOP | 19.4 TFLOP | 63.3 TFLOP | 224 TFLOP | 834 TFLOP | 3.22 PFLOP | 12.6 PFLOP | 50.0 PFLOP | 199 PFLOP |
| GLM-5.3 | 69.8 TFLOP | 165 TFLOP | 365 TFLOP | 799 TFLOP | 1.80 PFLOP | 4.37 PFLOP | 11.7 PFLOP | 35.3 PFLOP | 118 PFLOP |
| MiniMax-M3.1 Flash | 12.9 TFLOP | 30.9 TFLOP | 70.1 TFLOP | 161 TFLOP | 392 TFLOP | 1.05 PFLOP | 3.16 PFLOP | 10.6 PFLOP | 38.0 PFLOP |
| DeepSeek V4 Pro | 26.3 TFLOP | 72.3 TFLOP | 174 TFLOP | 414 TFLOP | 1.05 PFLOP | 2.91 PFLOP | 9.03 PFLOP | 30.9 PFLOP | 113 PFLOP |
| DeepSeek V4.1 Flash | 6.40 TFLOP | 13.6 TFLOP | 29.0 TFLOP | 63.9 TFLOP | 150 TFLOP | 389 TFLOP | 1.13 PFLOP | 3.67 PFLOP | 13.0 PFLOP |
| Kimi K3 | 16.9 TFLOP | 58.5 TFLOP | 216 TFLOP | 828 TFLOP | 3.24 PFLOP | 12.8 PFLOP | 51.0 PFLOP | 203 PFLOP | 812 PFLOP |
| Kimi K3.1 Pro | 37.7 TFLOP | 89.2 TFLOP | 202 TFLOP | 468 TFLOP | 1.16 PFLOP | 3.17 PFLOP | 9.73 PFLOP | 33.0 PFLOP | 120 PFLOP |
| GLM-5.3-Flash | 10.7 TFLOP | 24.9 TFLOP | 54.3 TFLOP | 118 TFLOP | 263 TFLOP | 625 TFLOP | 1.64 PFLOP | 4.83 PFLOP | 15.9 PFLOP |
| Qwen3.8 Flash Next | 3.06 TFLOP | 6.79 TFLOP | 14.4 TFLOP | 30.3 TFLOP | 64.4 TFLOP | 143 TFLOP | 339 TFLOP | 889 TFLOP | 2.62 PFLOP |
| Qwen3.8 27B | 4.87 TFLOP | 16.3 TFLOP | 59.1 TFLOP | 224 TFLOP | 870 TFLOP | 3.43 PFLOP | 13.6 PFLOP | 54.2 PFLOP | 217 PFLOP |

*Breakdown by kind*

| Model | Prompt | Global attention | Window attention | Linear attention (chunked) | Sparse attention | Indexer |
|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 4K | 25.6 TFLOP | — | — | — | — |
| Qwen3-Coder-480B-A35B | 8K | 102 TFLOP | — | — | — | — |
| Qwen3-Coder-480B-A35B | 16K | 409 TFLOP | — | — | — | — |
| Qwen3-Coder-480B-A35B | 32K | 1.64 PFLOP | — | — | — | — |
| Qwen3-Coder-480B-A35B | 64K | 6.54 PFLOP | — | — | — | — |
| Qwen3-Coder-480B-A35B | 128K | 26.2 PFLOP | — | — | — | — |
| Qwen3-Coder-480B-A35B | 256K | 105 PFLOP | — | — | — | — |
| MiMo V2.6 Flash | 4K | 3.09 TFLOP | 0.82 TFLOP | — | — | — |
| MiMo V2.6 Flash | 8K | 12.4 TFLOP | 1.66 TFLOP | — | — | — |
| MiMo V2.6 Flash | 16K | 49.5 TFLOP | 3.34 TFLOP | — | — | — |
| MiMo V2.6 Flash | 32K | 198 TFLOP | 6.69 TFLOP | — | — | — |
| MiMo V2.6 Flash | 64K | 792 TFLOP | 13.4 TFLOP | — | — | — |
| MiMo V2.6 Flash | 128K | 3.17 PFLOP | 26.8 TFLOP | — | — | — |
| MiMo V2.6 Flash | 256K | 12.7 PFLOP | 53.6 TFLOP | — | — | — |
| MiMo V2.6 Flash | 512K | 50.7 PFLOP | 107 TFLOP | — | — | — |
| MiMo V2.6 Flash | 1M | 203 PFLOP | 214 TFLOP | — | — | — |
| Inkling | 4K | 3.02 TFLOP | 3.54 TFLOP | — | — | — |
| Inkling | 8K | 12.1 TFLOP | 7.32 TFLOP | — | — | — |
| Inkling | 16K | 48.4 TFLOP | 14.9 TFLOP | — | — | — |
| Inkling | 32K | 194 TFLOP | 30.0 TFLOP | — | — | — |
| Inkling | 64K | 774 TFLOP | 60.2 TFLOP | — | — | — |
| Inkling | 128K | 3.10 PFLOP | 121 TFLOP | — | — | — |
| Inkling | 256K | 12.4 PFLOP | 242 TFLOP | — | — | — |
| Inkling | 512K | 49.5 PFLOP | 484 TFLOP | — | — | — |
| Inkling | 1M | 198 PFLOP | 967 TFLOP | — | — | — |
| GLM-5.3 | 4K | — | — | — | 68.4 TFLOP | 1.44 TFLOP |
| GLM-5.3 | 8K | — | — | — | 159 TFLOP | 5.77 TFLOP |
| GLM-5.3 | 16K | — | — | — | 342 TFLOP | 23.1 TFLOP |
| GLM-5.3 | 32K | — | — | — | 706 TFLOP | 92.4 TFLOP |
| GLM-5.3 | 64K | — | — | — | 1.44 PFLOP | 369 TFLOP |
| GLM-5.3 | 128K | — | — | — | 2.89 PFLOP | 1.48 PFLOP |
| GLM-5.3 | 256K | — | — | — | 5.81 PFLOP | 5.91 PFLOP |
| GLM-5.3 | 512K | — | — | — | 11.6 PFLOP | 23.6 PFLOP |
| GLM-5.3 | 1M | — | — | — | 23.3 PFLOP | 94.6 PFLOP |
| MiniMax-M3.1 Flash | 4K | — | — | — | 12.4 TFLOP | 0.52 TFLOP |
| MiniMax-M3.1 Flash | 8K | — | — | — | 28.9 TFLOP | 2.06 TFLOP |
| MiniMax-M3.1 Flash | 16K | — | — | — | 61.8 TFLOP | 8.25 TFLOP |
| MiniMax-M3.1 Flash | 32K | — | — | — | 128 TFLOP | 33.0 TFLOP |
| MiniMax-M3.1 Flash | 64K | — | — | — | 260 TFLOP | 132 TFLOP |
| MiniMax-M3.1 Flash | 128K | — | — | — | 524 TFLOP | 528 TFLOP |
| MiniMax-M3.1 Flash | 256K | — | — | — | 1.05 PFLOP | 2.11 PFLOP |
| MiniMax-M3.1 Flash | 512K | — | — | — | 2.11 PFLOP | 8.44 PFLOP |
| MiniMax-M3.1 Flash | 1M | — | — | — | 4.22 PFLOP | 33.8 PFLOP |
| DeepSeek V4 Pro | 4K | 4.71 TFLOP | — | — | 20.5 TFLOP | 1.03 TFLOP |
| DeepSeek V4 Pro | 8K | 10.6 TFLOP | — | — | 57.7 TFLOP | 4.12 TFLOP |
| DeepSeek V4 Pro | 16K | 25.4 TFLOP | — | — | 132 TFLOP | 16.5 TFLOP |
| DeepSeek V4 Pro | 32K | 68.0 TFLOP | — | — | 280 TFLOP | 66.0 TFLOP |
| DeepSeek V4 Pro | 64K | 204 TFLOP | — | — | 577 TFLOP | 264 TFLOP |
| DeepSeek V4 Pro | 128K | 681 TFLOP | — | — | 1.17 PFLOP | 1.06 PFLOP |
| DeepSeek V4 Pro | 256K | 2.45 PFLOP | — | — | 2.36 PFLOP | 4.22 PFLOP |
| DeepSeek V4 Pro | 512K | 9.27 PFLOP | — | — | 4.73 PFLOP | 16.9 PFLOP |
| DeepSeek V4 Pro | 1M | 36.0 PFLOP | — | — | 9.48 PFLOP | 67.6 PFLOP |
| DeepSeek V4.1 Flash | 4K | — | 0.14 TFLOP | — | 6.08 TFLOP | 0.19 TFLOP |
| DeepSeek V4.1 Flash | 8K | — | 0.27 TFLOP | — | 12.6 TFLOP | 0.72 TFLOP |
| DeepSeek V4.1 Flash | 16K | — | 0.55 TFLOP | — | 25.7 TFLOP | 2.82 TFLOP |
| DeepSeek V4.1 Flash | 32K | — | 1.10 TFLOP | — | 51.8 TFLOP | 11.1 TFLOP |
| DeepSeek V4.1 Flash | 64K | — | 2.20 TFLOP | — | 104 TFLOP | 44.0 TFLOP |
| DeepSeek V4.1 Flash | 128K | — | 4.40 TFLOP | — | 208 TFLOP | 176 TFLOP |
| DeepSeek V4.1 Flash | 256K | — | 8.79 TFLOP | — | 417 TFLOP | 704 TFLOP |
| DeepSeek V4.1 Flash | 512K | — | 17.6 TFLOP | — | 835 TFLOP | 2.81 PFLOP |
| DeepSeek V4.1 Flash | 1M | — | 35.2 TFLOP | — | 1.67 PFLOP | 11.3 PFLOP |
| Kimi K3 | 4K | 12.4 TFLOP | — | 4.52 TFLOP | — | — |
| Kimi K3 | 8K | 49.5 TFLOP | — | 9.04 TFLOP | — | — |
| Kimi K3 | 16K | 198 TFLOP | — | 18.1 TFLOP | — | — |
| Kimi K3 | 32K | 792 TFLOP | — | 36.2 TFLOP | — | — |
| Kimi K3 | 64K | 3.17 PFLOP | — | 72.3 TFLOP | — | — |
| Kimi K3 | 128K | 12.7 PFLOP | — | 145 TFLOP | — | — |
| Kimi K3 | 256K | 50.7 PFLOP | — | 289 TFLOP | — | — |
| Kimi K3 | 512K | 203 PFLOP | — | 578 TFLOP | — | — |
| Kimi K3 | 1M | 811 PFLOP | — | 1.16 PFLOP | — | — |
| Kimi K3.1 Pro | 4K | — | — | 4.52 TFLOP | 31.5 TFLOP | 1.65 TFLOP |
| Kimi K3.1 Pro | 8K | — | — | 9.04 TFLOP | 73.6 TFLOP | 6.60 TFLOP |
| Kimi K3.1 Pro | 16K | — | — | 18.1 TFLOP | 158 TFLOP | 26.4 TFLOP |
| Kimi K3.1 Pro | 32K | — | — | 36.2 TFLOP | 326 TFLOP | 106 TFLOP |
| Kimi K3.1 Pro | 64K | — | — | 72.3 TFLOP | 662 TFLOP | 422 TFLOP |
| Kimi K3.1 Pro | 128K | — | — | 145 TFLOP | 1.34 PFLOP | 1.69 PFLOP |
| Kimi K3.1 Pro | 256K | — | — | 289 TFLOP | 2.68 PFLOP | 6.76 PFLOP |
| Kimi K3.1 Pro | 512K | — | — | 578 TFLOP | 5.37 PFLOP | 27.0 PFLOP |
| Kimi K3.1 Pro | 1M | — | — | 1.16 PFLOP | 10.8 PFLOP | 108 PFLOP |
| GLM-5.3-Flash | 4K | — | — | 1.48 TFLOP | 9.07 TFLOP | 0.19 TFLOP |
| GLM-5.3-Flash | 8K | — | — | 2.97 TFLOP | 21.2 TFLOP | 0.76 TFLOP |
| GLM-5.3-Flash | 16K | — | — | 5.94 TFLOP | 45.4 TFLOP | 3.02 TFLOP |
| GLM-5.3-Flash | 32K | — | — | 11.9 TFLOP | 93.7 TFLOP | 12.1 TFLOP |
| GLM-5.3-Flash | 64K | — | — | 23.8 TFLOP | 190 TFLOP | 48.4 TFLOP |
| GLM-5.3-Flash | 128K | — | — | 47.5 TFLOP | 384 TFLOP | 194 TFLOP |
| GLM-5.3-Flash | 256K | — | — | 95.0 TFLOP | 771 TFLOP | 774 TFLOP |
| GLM-5.3-Flash | 512K | — | — | 190 TFLOP | 1.55 PFLOP | 3.10 PFLOP |
| GLM-5.3-Flash | 1M | — | — | 380 TFLOP | 3.09 PFLOP | 12.4 PFLOP |
| Qwen3.8 Flash Next | 4K | — | — | 1.18 TFLOP | 1.86 TFLOP | 0.03 TFLOP |
| Qwen3.8 Flash Next | 8K | — | — | 2.36 TFLOP | 4.33 TFLOP | 0.10 TFLOP |
| Qwen3.8 Flash Next | 16K | — | — | 4.72 TFLOP | 9.28 TFLOP | 0.41 TFLOP |
| Qwen3.8 Flash Next | 32K | — | — | 9.43 TFLOP | 19.2 TFLOP | 1.65 TFLOP |
| Qwen3.8 Flash Next | 64K | — | — | 18.9 TFLOP | 39.0 TFLOP | 6.60 TFLOP |
| Qwen3.8 Flash Next | 128K | — | — | 37.7 TFLOP | 78.5 TFLOP | 26.4 TFLOP |
| Qwen3.8 Flash Next | 256K | — | — | 75.5 TFLOP | 158 TFLOP | 106 TFLOP |
| Qwen3.8 Flash Next | 512K | — | — | 151 TFLOP | 316 TFLOP | 422 TFLOP |
| Qwen3.8 Flash Next | 1M | — | — | 302 TFLOP | 633 TFLOP | 1.69 PFLOP |
| Qwen3.8 27B | 4K | 3.30 TFLOP | — | 1.57 TFLOP | — | — |
| Qwen3.8 27B | 8K | 13.2 TFLOP | — | 3.14 TFLOP | — | — |
| Qwen3.8 27B | 16K | 52.8 TFLOP | — | 6.29 TFLOP | — | — |
| Qwen3.8 27B | 32K | 211 TFLOP | — | 12.6 TFLOP | — | — |
| Qwen3.8 27B | 64K | 844 TFLOP | — | 25.2 TFLOP | — | — |
| Qwen3.8 27B | 128K | 3.38 PFLOP | — | 50.3 TFLOP | — | — |
| Qwen3.8 27B | 256K | 13.5 PFLOP | — | 101 TFLOP | — | — |
| Qwen3.8 27B | 512K | 54.0 PFLOP | — | 201 TFLOP | — | — |
| Qwen3.8 27B | 1M | 216 PFLOP | — | 402 TFLOP | — | — |

*Takeaways the figure shows as the slider moves*

- 4K–16K: At short prompts, selecting a fixed number of tokens saves little. At 4K tokens, GLM-5.3, Kimi K3.1 Pro and DeepSeek V4 Pro even need more attention FLOPs than the baseline, and GLM-5.3 still does at 8K.
- 32K–128K: The baseline's FLOPs grow with the square of the prompt, while the sparse models grow about twice per doubling. At 128K tokens, Qwen3.8 Flash Next needs 183× less than the baseline.
- 256K–1M: The indexer still scores every earlier token, so its share keeps rising. At 1M tokens, it is 60% to 90% of the attention FLOPs in the sparse models.
