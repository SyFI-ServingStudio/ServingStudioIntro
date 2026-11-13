# Attention memory of one request

All the data behind this figure in "Beyond GQA: Characterizing Modern Attention Variants with ServingStudio".

*Total per model by context length*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 496 MiB | 992 MiB | 1.94 GiB | 3.88 GiB | 7.75 GiB | 15.5 GiB | 31.0 GiB | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 57.2 MiB | 102 MiB | 192 MiB | 372 MiB | 732 MiB | 1.42 GiB | 2.82 GiB | 5.64 GiB | 11.3 GiB |
| Inkling | 206 MiB | 294 MiB | 470 MiB | 822 MiB | 1.49 GiB | 2.87 GiB | 5.62 GiB | 11.1 GiB | 22.1 GiB |
| GLM-5.3 | 186 MiB | 373 MiB | 745 MiB | 1.46 GiB | 2.91 GiB | 5.82 GiB | 11.6 GiB | 23.3 GiB | 46.6 GiB |
| MiniMax-M3.1 Flash | 270 MiB | 540 MiB | 1.05 GiB | 2.11 GiB | 4.22 GiB | 8.44 GiB | 16.9 GiB | 33.8 GiB | 67.5 GiB |
| DeepSeek V4 Pro | 43.7 MiB | 65.3 MiB | 108 MiB | 194 MiB | 367 MiB | 711 MiB | 1.37 GiB | 2.71 GiB | 5.40 GiB |
| DeepSeek V4.1 Flash | 6.15 MiB | 9.63 MiB | 16.6 MiB | 30.5 MiB | 58.3 MiB | 114 MiB | 225 MiB | 448 MiB | 893 MiB |
| Kimi K3 | 483 MiB | 537 MiB | 645 MiB | 861 MiB | 1.26 GiB | 2.11 GiB | 3.79 GiB | 7.17 GiB | 13.9 GiB |
| Kimi K3.1 Pro | 502 MiB | 576 MiB | 724 MiB | 1020 MiB | 1.57 GiB | 2.73 GiB | 5.04 GiB | 9.65 GiB | 18.9 GiB |
| GLM-5.3-Flash | 164 MiB | 188 MiB | 234 MiB | 328 MiB | 515 MiB | 890 MiB | 1.60 GiB | 3.06 GiB | 5.99 GiB |
| Qwen3.8 Flash Next | 160 MiB | 209 MiB | 308 MiB | 506 MiB | 902 MiB | 1.65 GiB | 3.20 GiB | 6.30 GiB | 12.5 GiB |
| Qwen3.8 27B | 275 MiB | 403 MiB | 659 MiB | 1.14 GiB | 2.14 GiB | 4.14 GiB | 8.14 GiB | 16.1 GiB | 32.1 GiB |

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
| Inkling | 4K | 88.0 MiB | — | 110 MiB | 8.08 MiB |
| Inkling | 8K | 176 MiB | — | 110 MiB | 8.08 MiB |
| Inkling | 16K | 352 MiB | — | 110 MiB | 8.08 MiB |
| Inkling | 32K | 704 MiB | — | 110 MiB | 8.08 MiB |
| Inkling | 64K | 1.38 GiB | — | 110 MiB | 8.08 MiB |
| Inkling | 128K | 2.75 GiB | — | 110 MiB | 8.08 MiB |
| Inkling | 256K | 5.50 GiB | — | 110 MiB | 8.08 MiB |
| Inkling | 512K | 11.0 GiB | — | 110 MiB | 8.08 MiB |
| Inkling | 1M | 22.0 GiB | — | 110 MiB | 8.08 MiB |
| GLM-5.3 | 4K | 176 MiB | 10.8 MiB | — | — |
| GLM-5.3 | 8K | 351 MiB | 21.7 MiB | — | — |
| GLM-5.3 | 16K | 702 MiB | 43.3 MiB | — | — |
| GLM-5.3 | 32K | 1.37 GiB | 86.6 MiB | — | — |
| GLM-5.3 | 64K | 2.74 GiB | 173 MiB | — | — |
| GLM-5.3 | 128K | 5.48 GiB | 347 MiB | — | — |
| GLM-5.3 | 256K | 11.0 GiB | 693 MiB | — | — |
| GLM-5.3 | 512K | 21.9 GiB | 1.35 GiB | — | — |
| GLM-5.3 | 1M | 43.9 GiB | 2.71 GiB | — | — |
| MiniMax-M3.1 Flash | 4K | 240 MiB | 30.0 MiB | — | — |
| MiniMax-M3.1 Flash | 8K | 480 MiB | 60.0 MiB | — | — |
| MiniMax-M3.1 Flash | 16K | 960 MiB | 120 MiB | — | — |
| MiniMax-M3.1 Flash | 32K | 1.88 GiB | 240 MiB | — | — |
| MiniMax-M3.1 Flash | 64K | 3.75 GiB | 480 MiB | — | — |
| MiniMax-M3.1 Flash | 128K | 7.50 GiB | 960 MiB | — | — |
| MiniMax-M3.1 Flash | 256K | 15.0 GiB | 1.88 GiB | — | — |
| MiniMax-M3.1 Flash | 512K | 30.0 GiB | 3.75 GiB | — | — |
| MiniMax-M3.1 Flash | 1M | 60.0 GiB | 7.50 GiB | — | — |
| DeepSeek V4 Pro | 4K | 17.7 MiB | 3.87 MiB | 4.35 MiB | 17.8 MiB |
| DeepSeek V4 Pro | 8K | 35.3 MiB | 7.73 MiB | 4.35 MiB | 17.8 MiB |
| DeepSeek V4 Pro | 16K | 70.6 MiB | 15.5 MiB | 4.35 MiB | 17.8 MiB |
| DeepSeek V4 Pro | 32K | 141 MiB | 30.9 MiB | 4.35 MiB | 17.8 MiB |
| DeepSeek V4 Pro | 64K | 283 MiB | 61.9 MiB | 4.35 MiB | 17.8 MiB |
| DeepSeek V4 Pro | 128K | 565 MiB | 124 MiB | 4.35 MiB | 17.8 MiB |
| DeepSeek V4 Pro | 256K | 1.10 GiB | 248 MiB | 4.35 MiB | 17.8 MiB |
| DeepSeek V4 Pro | 512K | 2.21 GiB | 495 MiB | 4.35 MiB | 17.8 MiB |
| DeepSeek V4 Pro | 1M | 4.42 GiB | 990 MiB | 4.35 MiB | 17.8 MiB |
| DeepSeek V4.1 Flash | 4K | 2.81 MiB | 680 KiB | 2.58 MiB | 96.0 KiB |
| DeepSeek V4.1 Flash | 8K | 5.63 MiB | 1.33 MiB | 2.58 MiB | 96.0 KiB |
| DeepSeek V4.1 Flash | 16K | 11.3 MiB | 2.66 MiB | 2.58 MiB | 96.0 KiB |
| DeepSeek V4.1 Flash | 32K | 22.5 MiB | 5.31 MiB | 2.58 MiB | 96.0 KiB |
| DeepSeek V4.1 Flash | 64K | 45.0 MiB | 10.6 MiB | 2.58 MiB | 96.0 KiB |
| DeepSeek V4.1 Flash | 128K | 90.0 MiB | 21.3 MiB | 2.58 MiB | 96.0 KiB |
| DeepSeek V4.1 Flash | 256K | 180 MiB | 42.5 MiB | 2.58 MiB | 96.0 KiB |
| DeepSeek V4.1 Flash | 512K | 360 MiB | 85.0 MiB | 2.58 MiB | 96.0 KiB |
| DeepSeek V4.1 Flash | 1M | 720 MiB | 170 MiB | 2.58 MiB | 96.0 KiB |
| Kimi K3 | 4K | 54.0 MiB | — | — | 429 MiB |
| Kimi K3 | 8K | 108 MiB | — | — | 429 MiB |
| Kimi K3 | 16K | 216 MiB | — | — | 429 MiB |
| Kimi K3 | 32K | 432 MiB | — | — | 429 MiB |
| Kimi K3 | 64K | 864 MiB | — | — | 429 MiB |
| Kimi K3 | 128K | 1.69 GiB | — | — | 429 MiB |
| Kimi K3 | 256K | 3.38 GiB | — | — | 429 MiB |
| Kimi K3 | 512K | 6.75 GiB | — | — | 429 MiB |
| Kimi K3 | 1M | 13.5 GiB | — | — | 429 MiB |
| Kimi K3.1 Pro | 4K | 61.5 MiB | 12.4 MiB | — | 429 MiB |
| Kimi K3.1 Pro | 8K | 123 MiB | 24.8 MiB | — | 429 MiB |
| Kimi K3.1 Pro | 16K | 246 MiB | 49.5 MiB | — | 429 MiB |
| Kimi K3.1 Pro | 32K | 492 MiB | 99.0 MiB | — | 429 MiB |
| Kimi K3.1 Pro | 64K | 984 MiB | 198 MiB | — | 429 MiB |
| Kimi K3.1 Pro | 128K | 1.92 GiB | 396 MiB | — | 429 MiB |
| Kimi K3.1 Pro | 256K | 3.84 GiB | 792 MiB | — | 429 MiB |
| Kimi K3.1 Pro | 512K | 7.69 GiB | 1.55 GiB | — | 429 MiB |
| Kimi K3.1 Pro | 1M | 15.4 GiB | 3.09 GiB | — | 429 MiB |
| GLM-5.3-Flash | 4K | 22.0 MiB | 1.42 MiB | — | 141 MiB |
| GLM-5.3-Flash | 8K | 44.0 MiB | 2.84 MiB | — | 141 MiB |
| GLM-5.3-Flash | 16K | 88.0 MiB | 5.67 MiB | — | 141 MiB |
| GLM-5.3-Flash | 32K | 176 MiB | 11.3 MiB | — | 141 MiB |
| GLM-5.3-Flash | 64K | 352 MiB | 22.7 MiB | — | 141 MiB |
| GLM-5.3-Flash | 128K | 704 MiB | 45.4 MiB | — | 141 MiB |
| GLM-5.3-Flash | 256K | 1.38 GiB | 90.8 MiB | — | 141 MiB |
| GLM-5.3-Flash | 512K | 2.75 GiB | 182 MiB | — | 141 MiB |
| GLM-5.3-Flash | 1M | 5.50 GiB | 363 MiB | — | 141 MiB |
| Qwen3.8 Flash Next | 4K | 48.0 MiB | 1.50 MiB | — | 110 MiB |
| Qwen3.8 Flash Next | 8K | 96.0 MiB | 3.00 MiB | — | 110 MiB |
| Qwen3.8 Flash Next | 16K | 192 MiB | 6.00 MiB | — | 110 MiB |
| Qwen3.8 Flash Next | 32K | 384 MiB | 12.0 MiB | — | 110 MiB |
| Qwen3.8 Flash Next | 64K | 768 MiB | 24.0 MiB | — | 110 MiB |
| Qwen3.8 Flash Next | 128K | 1.50 GiB | 48.0 MiB | — | 110 MiB |
| Qwen3.8 Flash Next | 256K | 3.00 GiB | 96.0 MiB | — | 110 MiB |
| Qwen3.8 Flash Next | 512K | 6.00 GiB | 192 MiB | — | 110 MiB |
| Qwen3.8 Flash Next | 1M | 12.0 GiB | 384 MiB | — | 110 MiB |
| Qwen3.8 27B | 4K | 128 MiB | — | — | 147 MiB |
| Qwen3.8 27B | 8K | 256 MiB | — | — | 147 MiB |
| Qwen3.8 27B | 16K | 512 MiB | — | — | 147 MiB |
| Qwen3.8 27B | 32K | 1.00 GiB | — | — | 147 MiB |
| Qwen3.8 27B | 64K | 2.00 GiB | — | — | 147 MiB |
| Qwen3.8 27B | 128K | 4.00 GiB | — | — | 147 MiB |
| Qwen3.8 27B | 256K | 8.00 GiB | — | — | 147 MiB |
| Qwen3.8 27B | 512K | 16.0 GiB | — | — | 147 MiB |
| Qwen3.8 27B | 1M | 32.0 GiB | — | — | 147 MiB |

*Takeaways the figure shows as the slider moves*

- 4K–16K: At short contexts, the fixed linear-attention states dominate. At 4K tokens, Kimi K3 and Kimi K3.1 Pro keep 483 MiB and 502 MiB per request, about as much as the baseline's 496 MiB.
- 32K–64K: As the context grows, the per-token caches catch up with the fixed states, and the bytes per token start to decide.
- 128K–256K: At 128K tokens, DeepSeek V4.1 Flash keeps 139× less than the baseline's 15.5 GiB. Even MiniMax-M3.1 Flash, which keeps the most of the new models, keeps 1.8× less.
- 512K–1M: Past 256K tokens, only the new models go on. At 1M tokens, they keep from 893 MiB in DeepSeek V4.1 Flash to 67.5 GiB in MiniMax-M3.1 Flash.
