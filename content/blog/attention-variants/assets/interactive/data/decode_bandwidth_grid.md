# Achieved memory bandwidth of decode attention

All the data behind this figure in "Beyond GQA: Characterizing Modern Attention Variants with ServingStudio".

*Per model by context length (Batch 1)*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 0.68 TB/s | 1.22 TB/s | 2.01 TB/s | 3.12 TB/s | 4.35 TB/s | 5.43 TB/s | 6.15 TB/s | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 0.07 TB/s | 0.13 TB/s | 0.24 TB/s | 0.44 TB/s | 0.78 TB/s | 1.30 TB/s | 2.01 TB/s | 2.79 TB/s | 3.42 TB/s |
| Inkling | 0.16 TB/s | 0.21 TB/s | 0.32 TB/s | 0.50 TB/s | 0.81 TB/s | 1.30 TB/s | 1.73 TB/s | 2.09 TB/s | 2.33 TB/s |
| GLM-5.3 | 0.16 TB/s | 0.17 TB/s | 0.19 TB/s | 0.20 TB/s | 0.30 TB/s | 0.47 TB/s | 0.71 TB/s | 1.11 TB/s | 1.69 TB/s |
| MiniMax-M3.1 Flash | 0.15 TB/s | 0.16 TB/s | 0.22 TB/s | 0.31 TB/s | 0.48 TB/s | 0.76 TB/s | 1.17 TB/s | 1.68 TB/s | 2.16 TB/s |
| DeepSeek V4 Pro | 0.06 TB/s | 0.06 TB/s | 0.07 TB/s | 0.08 TB/s | 0.09 TB/s | 0.12 TB/s | 0.19 TB/s | 0.29 TB/s | 0.53 TB/s |
| DeepSeek V4.1 Flash | 0.01 TB/s | 0.01 TB/s | 0.01 TB/s | 0.02 TB/s | 0.03 TB/s | 0.05 TB/s | 0.10 TB/s | 0.18 TB/s | 0.31 TB/s |
| Kimi K3 | 1.22 TB/s | 1.26 TB/s | 1.27 TB/s | 1.44 TB/s | 1.79 TB/s | 2.18 TB/s | 2.99 TB/s | 3.61 TB/s | 3.83 TB/s |
| Kimi K3.1 Pro | 0.57 TB/s | 0.56 TB/s | 0.56 TB/s | 0.53 TB/s | 0.58 TB/s | 0.66 TB/s | 0.79 TB/s | 1.03 TB/s | 1.45 TB/s |
| GLM-5.3-Flash | 1.09 TB/s | 1.08 TB/s | 1.08 TB/s | 1.06 TB/s | 1.03 TB/s | 0.98 TB/s | 1.09 TB/s | 1.26 TB/s | 1.52 TB/s |
| Qwen3.8 Flash Next | 0.61 TB/s | 0.61 TB/s | 0.61 TB/s | 0.60 TB/s | 0.60 TB/s | 0.59 TB/s | 0.66 TB/s | 0.79 TB/s | 1.03 TB/s |
| Qwen3.8 27B | 0.94 TB/s | 1.13 TB/s | 1.44 TB/s | 2.09 TB/s | 3.04 TB/s | 4.19 TB/s | 5.25 TB/s | 6.00 TB/s | 6.56 TB/s |

*Per model by context length (Batch 32)*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 5.81 TB/s | 6.37 TB/s | 6.68 TB/s | 6.83 TB/s | 6.99 TB/s | 7.02 TB/s | 7.03 TB/s | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 1.93 TB/s | 2.61 TB/s | 3.11 TB/s | 3.47 TB/s | 3.70 TB/s | 3.83 TB/s | 3.92 TB/s | 3.97 TB/s | 4.10 TB/s |
| Inkling | 1.43 TB/s | 1.62 TB/s | 1.82 TB/s | 1.99 TB/s | 2.18 TB/s | 2.33 TB/s | 2.42 TB/s | 2.47 TB/s | 2.51 TB/s |
| GLM-5.3 | 2.53 TB/s | 2.65 TB/s | 2.61 TB/s | 2.97 TB/s | 3.50 TB/s | 4.19 TB/s | 4.87 TB/s | 5.38 TB/s | 5.73 TB/s |
| MiniMax-M3.1 Flash | 2.34 TB/s | 2.66 TB/s | 3.18 TB/s | 3.77 TB/s | 4.84 TB/s | 5.63 TB/s | 6.20 TB/s | 6.54 TB/s | 6.69 TB/s |
| DeepSeek V4 Pro | 1.50 TB/s | 1.61 TB/s | 1.79 TB/s | 1.45 TB/s | 1.74 TB/s | 2.16 TB/s | 2.69 TB/s | 3.25 TB/s | 3.63 TB/s |
| DeepSeek V4.1 Flash | 0.31 TB/s | 0.35 TB/s | 0.43 TB/s | 0.58 TB/s | 0.84 TB/s | 1.23 TB/s | 1.71 TB/s | 2.19 TB/s | 2.58 TB/s |
| Kimi K3 | 5.53 TB/s | 5.56 TB/s | 5.36 TB/s | 5.10 TB/s | 4.71 TB/s | 4.84 TB/s | 4.06 TB/s | 3.94 TB/s | 3.86 TB/s |
| Kimi K3.1 Pro | 4.39 TB/s | 4.38 TB/s | 4.31 TB/s | 4.33 TB/s | 4.42 TB/s | 4.59 TB/s | 4.89 TB/s | 5.22 TB/s | 5.55 TB/s |
| GLM-5.3-Flash | 4.01 TB/s | 4.04 TB/s | 4.03 TB/s | 4.06 TB/s | 4.03 TB/s | 4.06 TB/s | 4.21 TB/s | 4.48 TB/s | 4.78 TB/s |
| Qwen3.8 Flash Next | 4.01 TB/s | 4.03 TB/s | 4.06 TB/s | 4.09 TB/s | 4.15 TB/s | 4.22 TB/s | 4.34 TB/s | 4.71 TB/s | 5.16 TB/s |
| Qwen3.8 27B | 5.81 TB/s | 6.08 TB/s | 6.41 TB/s | 6.74 TB/s | 6.97 TB/s | 7.13 TB/s | 7.21 TB/s | 7.26 TB/s | 7.28 TB/s |

Color scale: 0 TB/s to 8 TB/s, the B200's peak.
