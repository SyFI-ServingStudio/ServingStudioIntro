# Measured attention time to prefill one prompt as the prompt grows

All the data behind this figure in "Beyond GQA: Characterizing Modern Attention Variants with ServingStudio".

*Per model by prompt length (Attention only)*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 31.5 ms | 74.6 ms | 271 ms | 1.02 s | 3.95 s | 15.6 s | 62.2 s | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 5.37 ms | 13.8 ms | 41.6 ms | 137 ms | 486 ms | 1.83 s | 7.05 s | 27.5 s | 110 s |
| Inkling | 22.8 ms | 49.4 ms | 120 ms | 318 ms | 955 ms | 3.21 s | 11.6 s | 43.9 s | 170 s |
| GLM-5.3 | 33.7 ms | 119 ms | 248 ms | 536 ms | 1.24 s | 3.09 s | 8.41 s | 25.5 s | 86.8 s |
| MiniMax-M3.1 Flash | 31.8 ms | 69.4 ms | 151 ms | 321 ms | 698 ms | 1.58 s | 3.95 s | 11.2 s | 35.5 s |
| DeepSeek V4 Pro | 33.9 ms | 87.2 ms | 176 ms | 358 ms | 850 ms | 2.30 s | 6.96 s | 23.8 s | 88.6 s |
| DeepSeek V4.1 Flash | 7.32 ms | 15.0 ms | 30.5 ms | 63.9 ms | 141 ms | 339 ms | 904 ms | 2.82 s | 10.2 s |
| Kimi K3 | 60.4 ms | 133 ms | 329 ms | 905 ms | 2.79 s | 9.51 s | 34.8 s | 133 s | 517 s |
| Kimi K3.1 Pro | 85.7 ms | 230 ms | 470 ms | 984 ms | 2.15 s | 4.98 s | 12.5 s | 34.9 s | 111 s |
| GLM-5.3-Flash | 27.2 ms | 56.6 ms | 115 ms | 233 ms | 482 ms | 1.03 s | 2.36 s | 5.96 s | 17.7 s |
| Qwen3.8 Flash Next | 27.9 ms | 60.6 ms | 129 ms | 266 ms | 551 ms | 1.16 s | 2.57 s | 5.88 s | 13.9 s |
| Qwen3.8 27B | 8.99 ms | 20.1 ms | 53.8 ms | 161 ms | 533 ms | 1.89 s | 7.31 s | 29.0 s | 114 s |

*Per model by prompt length (With other ops)*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 70.1 ms | 151 ms | 425 ms | 1.33 s | 4.57 s | 16.8 s | 64.6 s | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 29.6 ms | 62.2 ms | 138 ms | 330 ms | 873 ms | 2.60 s | 8.59 s | 30.6 s | 116 s |
| Inkling | 82.3 ms | 167 ms | 355 ms | 788 ms | 1.89 s | 5.09 s | 15.4 s | 51.4 s | 185 s |
| GLM-5.3 | 148 ms | 347 ms | 704 ms | 1.45 s | 3.06 s | 6.74 s | 15.7 s | 40.1 s | 116 s |
| MiniMax-M3.1 Flash | 55.0 ms | 118 ms | 249 ms | 518 ms | 1.09 s | 2.37 s | 5.53 s | 14.3 s | 41.9 s |
| DeepSeek V4 Pro | 126 ms | 271 ms | 544 ms | 1.09 s | 2.25 s | 4.97 s | 12.2 s | 34.1 s | 109 s |
| DeepSeek V4.1 Flash | 17.2 ms | 33.4 ms | 67.4 ms | 138 ms | 289 ms | 634 ms | 1.50 s | 4.01 s | 12.6 s |
| Kimi K3 | 296 ms | 596 ms | 1.26 s | 2.78 s | 6.58 s | 17.3 s | 51.0 s | 168 s | 600 s |
| Kimi K3.1 Pro | 327 ms | 703 ms | 1.42 s | 2.88 s | 5.94 s | 12.6 s | 27.7 s | 65.4 s | 172 s |
| GLM-5.3-Flash | 86.1 ms | 173 ms | 348 ms | 699 ms | 1.41 s | 2.90 s | 6.09 s | 13.4 s | 32.6 s |
| Qwen3.8 Flash Next | 54.6 ms | 113 ms | 233 ms | 474 ms | 967 ms | 1.99 s | 4.24 s | 9.21 s | 20.5 s |
| Qwen3.8 27B | 43.8 ms | 89.1 ms | 192 ms | 437 ms | 1.09 s | 3.00 s | 9.51 s | 33.4 s | 123 s |
