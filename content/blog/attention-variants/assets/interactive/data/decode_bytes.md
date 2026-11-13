# Bytes one request reads per decode step

All the data behind this figure in "Beyond GQA: Characterizing Modern Attention Variants with ServingStudio".

*Total per model by context length*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 496 MiB | 992 MiB | 1.94 GiB | 3.88 GiB | 7.75 GiB | 15.5 GiB | 31.0 GiB | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 57.2 MiB | 102 MiB | 192 MiB | 372 MiB | 732 MiB | 1.42 GiB | 2.82 GiB | 5.64 GiB | 11.3 GiB |
| Inkling | 214 MiB | 302 MiB | 478 MiB | 830 MiB | 1.50 GiB | 2.87 GiB | 5.62 GiB | 11.1 GiB | 22.1 GiB |
| GLM-5.3 | 98.6 MiB | 109 MiB | 131 MiB | 174 MiB | 261 MiB | 434 MiB | 781 MiB | 1.44 GiB | 2.79 GiB |
| MiniMax-M3.1 Flash | 150 MiB | 180 MiB | 240 MiB | 360 MiB | 600 MiB | 1.05 GiB | 1.99 GiB | 3.87 GiB | 7.62 GiB |
| DeepSeek V4 Pro | 61.6 MiB | 66.0 MiB | 74.8 MiB | 92.5 MiB | 128 MiB | 199 MiB | 340 MiB | 623 MiB | 1.16 GiB |
| DeepSeek V4.1 Flash | 9.84 MiB | 11.6 MiB | 15.0 MiB | 21.9 MiB | 35.7 MiB | 63.4 MiB | 119 MiB | 229 MiB | 450 MiB |
| Kimi K3 | 911 MiB | 965 MiB | 1.05 GiB | 1.26 GiB | 1.68 GiB | 2.52 GiB | 4.21 GiB | 7.59 GiB | 14.3 GiB |
| Kimi K3.1 Pro | 900 MiB | 913 MiB | 937 MiB | 987 MiB | 1.06 GiB | 1.25 GiB | 1.64 GiB | 2.41 GiB | 3.96 GiB |
| GLM-5.3-Flash | 294 MiB | 295 MiB | 298 MiB | 304 MiB | 315 MiB | 338 MiB | 383 MiB | 474 MiB | 656 MiB |
| Qwen3.8 Flash Next | 246 MiB | 248 MiB | 251 MiB | 257 MiB | 269 MiB | 293 MiB | 341 MiB | 437 MiB | 629 MiB |
| Qwen3.8 27B | 422 MiB | 550 MiB | 806 MiB | 1.29 GiB | 2.29 GiB | 4.29 GiB | 8.29 GiB | 16.3 GiB | 32.3 GiB |

*Breakdown by kind*

| Model | Context | Whole-context cache | Index | Window | Fixed state |
|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 4K | 496 MiB | — | — | — |
| Qwen3-Coder-480B-A35B | 8K | 992 MiB | — | — | — |
| Qwen3-Coder-480B-A35B | 16K | 1.94 GiB | — | — | — |
| Qwen3-Coder-480B-A35B | 32K | 3.88 GiB | — | — | — |
| Qwen3-Coder-480B-A35B | 64K | 7.75 GiB | — | — | — |
| Qwen3-Coder-480B-A35B | 128K | 15.5 GiB | — | — | — |
| Qwen3-Coder-480B-A35B | 256K | 31.0 GiB | — | — | — |
| MiMo V2.6 Flash | 4K | 45.0 MiB | — | 12.2 MiB | — |
| MiMo V2.6 Flash | 8K | 90.0 MiB | — | 12.2 MiB | — |
| MiMo V2.6 Flash | 16K | 180 MiB | — | 12.2 MiB | — |
| MiMo V2.6 Flash | 32K | 360 MiB | — | 12.2 MiB | — |
| MiMo V2.6 Flash | 64K | 720 MiB | — | 12.2 MiB | — |
| MiMo V2.6 Flash | 128K | 1.41 GiB | — | 12.2 MiB | — |
| MiMo V2.6 Flash | 256K | 2.81 GiB | — | 12.2 MiB | — |
| MiMo V2.6 Flash | 512K | 5.63 GiB | — | 12.2 MiB | — |
| MiMo V2.6 Flash | 1M | 11.3 GiB | — | 12.2 MiB | — |
| Inkling | 4K | 88.0 MiB | — | 110 MiB | 16.2 MiB |
| Inkling | 8K | 176 MiB | — | 110 MiB | 16.2 MiB |
| Inkling | 16K | 352 MiB | — | 110 MiB | 16.2 MiB |
| Inkling | 32K | 704 MiB | — | 110 MiB | 16.2 MiB |
| Inkling | 64K | 1.38 GiB | — | 110 MiB | 16.2 MiB |
| Inkling | 128K | 2.75 GiB | — | 110 MiB | 16.2 MiB |
| Inkling | 256K | 5.50 GiB | — | 110 MiB | 16.2 MiB |
| Inkling | 512K | 11.0 GiB | — | 110 MiB | 16.2 MiB |
| Inkling | 1M | 22.0 GiB | — | 110 MiB | 16.2 MiB |
| GLM-5.3 | 4K | 87.8 MiB | 10.8 MiB | — | — |
| GLM-5.3 | 8K | 87.8 MiB | 21.7 MiB | — | — |
| GLM-5.3 | 16K | 87.8 MiB | 43.3 MiB | — | — |
| GLM-5.3 | 32K | 87.8 MiB | 86.6 MiB | — | — |
| GLM-5.3 | 64K | 87.8 MiB | 173 MiB | — | — |
| GLM-5.3 | 128K | 87.8 MiB | 347 MiB | — | — |
| GLM-5.3 | 256K | 87.8 MiB | 693 MiB | — | — |
| GLM-5.3 | 512K | 87.8 MiB | 1.35 GiB | — | — |
| GLM-5.3 | 1M | 87.8 MiB | 2.71 GiB | — | — |
| MiniMax-M3.1 Flash | 4K | 120 MiB | 30.0 MiB | — | — |
| MiniMax-M3.1 Flash | 8K | 120 MiB | 60.0 MiB | — | — |
| MiniMax-M3.1 Flash | 16K | 120 MiB | 120 MiB | — | — |
| MiniMax-M3.1 Flash | 32K | 120 MiB | 240 MiB | — | — |
| MiniMax-M3.1 Flash | 64K | 120 MiB | 480 MiB | — | — |
| MiniMax-M3.1 Flash | 128K | 120 MiB | 960 MiB | — | — |
| MiniMax-M3.1 Flash | 256K | 120 MiB | 1.88 GiB | — | — |
| MiniMax-M3.1 Flash | 512K | 120 MiB | 3.75 GiB | — | — |
| MiniMax-M3.1 Flash | 1M | 120 MiB | 7.50 GiB | — | — |
| DeepSeek V4 Pro | 4K | 17.7 MiB | 3.87 MiB | 4.35 MiB | 35.7 MiB |
| DeepSeek V4 Pro | 8K | 18.2 MiB | 7.73 MiB | 4.35 MiB | 35.7 MiB |
| DeepSeek V4 Pro | 16K | 19.3 MiB | 15.5 MiB | 4.35 MiB | 35.7 MiB |
| DeepSeek V4 Pro | 32K | 21.5 MiB | 30.9 MiB | 4.35 MiB | 35.7 MiB |
| DeepSeek V4 Pro | 64K | 25.9 MiB | 61.9 MiB | 4.35 MiB | 35.7 MiB |
| DeepSeek V4 Pro | 128K | 34.8 MiB | 124 MiB | 4.35 MiB | 35.7 MiB |
| DeepSeek V4 Pro | 256K | 52.5 MiB | 248 MiB | 4.35 MiB | 35.7 MiB |
| DeepSeek V4 Pro | 512K | 87.8 MiB | 495 MiB | 4.35 MiB | 35.7 MiB |
| DeepSeek V4 Pro | 1M | 159 MiB | 990 MiB | 4.35 MiB | 35.7 MiB |
| DeepSeek V4.1 Flash | 4K | 5.34 MiB | 1.73 MiB | 2.58 MiB | 192 KiB |
| DeepSeek V4.1 Flash | 8K | 5.34 MiB | 3.45 MiB | 2.58 MiB | 192 KiB |
| DeepSeek V4.1 Flash | 16K | 5.34 MiB | 6.91 MiB | 2.58 MiB | 192 KiB |
| DeepSeek V4.1 Flash | 32K | 5.34 MiB | 13.8 MiB | 2.58 MiB | 192 KiB |
| DeepSeek V4.1 Flash | 64K | 5.34 MiB | 27.6 MiB | 2.58 MiB | 192 KiB |
| DeepSeek V4.1 Flash | 128K | 5.34 MiB | 55.3 MiB | 2.58 MiB | 192 KiB |
| DeepSeek V4.1 Flash | 256K | 5.34 MiB | 111 MiB | 2.58 MiB | 192 KiB |
| DeepSeek V4.1 Flash | 512K | 5.34 MiB | 221 MiB | 2.58 MiB | 192 KiB |
| DeepSeek V4.1 Flash | 1M | 5.34 MiB | 442 MiB | 2.58 MiB | 192 KiB |
| Kimi K3 | 4K | 54.0 MiB | — | — | 857 MiB |
| Kimi K3 | 8K | 108 MiB | — | — | 857 MiB |
| Kimi K3 | 16K | 216 MiB | — | — | 857 MiB |
| Kimi K3 | 32K | 432 MiB | — | — | 857 MiB |
| Kimi K3 | 64K | 864 MiB | — | — | 857 MiB |
| Kimi K3 | 128K | 1.69 GiB | — | — | 857 MiB |
| Kimi K3 | 256K | 3.38 GiB | — | — | 857 MiB |
| Kimi K3 | 512K | 6.75 GiB | — | — | 857 MiB |
| Kimi K3 | 1M | 13.5 GiB | — | — | 857 MiB |
| Kimi K3.1 Pro | 4K | 30.8 MiB | 12.4 MiB | — | 857 MiB |
| Kimi K3.1 Pro | 8K | 30.8 MiB | 24.8 MiB | — | 857 MiB |
| Kimi K3.1 Pro | 16K | 30.8 MiB | 49.5 MiB | — | 857 MiB |
| Kimi K3.1 Pro | 32K | 30.8 MiB | 99.0 MiB | — | 857 MiB |
| Kimi K3.1 Pro | 64K | 30.8 MiB | 198 MiB | — | 857 MiB |
| Kimi K3.1 Pro | 128K | 30.8 MiB | 396 MiB | — | 857 MiB |
| Kimi K3.1 Pro | 256K | 30.8 MiB | 792 MiB | — | 857 MiB |
| Kimi K3.1 Pro | 512K | 30.8 MiB | 1.55 GiB | — | 857 MiB |
| Kimi K3.1 Pro | 1M | 30.8 MiB | 3.09 GiB | — | 857 MiB |
| GLM-5.3-Flash | 4K | 11.0 MiB | 1.42 MiB | — | 282 MiB |
| GLM-5.3-Flash | 8K | 11.0 MiB | 2.84 MiB | — | 282 MiB |
| GLM-5.3-Flash | 16K | 11.0 MiB | 5.67 MiB | — | 282 MiB |
| GLM-5.3-Flash | 32K | 11.0 MiB | 11.3 MiB | — | 282 MiB |
| GLM-5.3-Flash | 64K | 11.0 MiB | 22.7 MiB | — | 282 MiB |
| GLM-5.3-Flash | 128K | 11.0 MiB | 45.4 MiB | — | 282 MiB |
| GLM-5.3-Flash | 256K | 11.0 MiB | 90.8 MiB | — | 282 MiB |
| GLM-5.3-Flash | 512K | 11.0 MiB | 182 MiB | — | 282 MiB |
| GLM-5.3-Flash | 1M | 11.0 MiB | 363 MiB | — | 282 MiB |
| Qwen3.8 Flash Next | 4K | 24.0 MiB | 1.50 MiB | — | 221 MiB |
| Qwen3.8 Flash Next | 8K | 24.0 MiB | 3.00 MiB | — | 221 MiB |
| Qwen3.8 Flash Next | 16K | 24.0 MiB | 6.00 MiB | — | 221 MiB |
| Qwen3.8 Flash Next | 32K | 24.0 MiB | 12.0 MiB | — | 221 MiB |
| Qwen3.8 Flash Next | 64K | 24.0 MiB | 24.0 MiB | — | 221 MiB |
| Qwen3.8 Flash Next | 128K | 24.0 MiB | 48.0 MiB | — | 221 MiB |
| Qwen3.8 Flash Next | 256K | 24.0 MiB | 96.0 MiB | — | 221 MiB |
| Qwen3.8 Flash Next | 512K | 24.0 MiB | 192 MiB | — | 221 MiB |
| Qwen3.8 Flash Next | 1M | 24.0 MiB | 384 MiB | — | 221 MiB |
| Qwen3.8 27B | 4K | 128 MiB | — | — | 294 MiB |
| Qwen3.8 27B | 8K | 256 MiB | — | — | 294 MiB |
| Qwen3.8 27B | 16K | 512 MiB | — | — | 294 MiB |
| Qwen3.8 27B | 32K | 1.00 GiB | — | — | 294 MiB |
| Qwen3.8 27B | 64K | 2.00 GiB | — | — | 294 MiB |
| Qwen3.8 27B | 128K | 4.00 GiB | — | — | 294 MiB |
| Qwen3.8 27B | 256K | 8.00 GiB | — | — | 294 MiB |
| Qwen3.8 27B | 512K | 16.0 GiB | — | — | 294 MiB |
| Qwen3.8 27B | 1M | 32.0 GiB | — | — | 294 MiB |

*Takeaways the figure shows as the slider moves*

- 4K–16K: At short contexts, the linear-attention state is most of what a step reads, because each step reads it and writes it back at any context length. In GLM-5.3-Flash, it is over 90%.
- 32K–128K: A sparse layer keeps every token but reads only its index and the selected entries. At 128K tokens, GLM-5.3 keeps 5.8 GiB per request, but a step reads only 434 MiB, 14× less, and 80% of that is the index.
- 256K–1M: The index read keeps growing with the context. At 1M tokens, it is more than half of a step in every sparse model, and 98% in DeepSeek V4.1 Flash and MiniMax-M3.1 Flash. Kimi K3's dense MLA reads its whole cache instead, 3.6× more than Kimi K3.1 Pro.
