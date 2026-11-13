# Measured attention time of one decode step as the context grows

All the data behind this figure in "Beyond GQA: Characterizing Modern Attention Variants with ServingStudio".

*Per model by context length (Batch 1, Attention only)*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 0.76 ms | 0.85 ms | 1.04 ms | 1.33 ms | 1.92 ms | 3.06 ms | 5.41 ms | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 0.80 ms | 0.82 ms | 0.84 ms | 0.89 ms | 0.98 ms | 1.17 ms | 1.51 ms | 2.17 ms | 3.54 ms |
| Inkling | 1.43 ms | 1.50 ms | 1.58 ms | 1.73 ms | 1.97 ms | 2.38 ms | 3.49 ms | 5.72 ms | 10.2 ms |
| GLM-5.3 | 0.66 ms | 0.69 ms | 0.74 ms | 0.92 ms | 0.92 ms | 0.97 ms | 1.15 ms | 1.40 ms | 1.77 ms |
| MiniMax-M3.1 Flash | 1.08 ms | 1.15 ms | 1.16 ms | 1.24 ms | 1.32 ms | 1.50 ms | 1.83 ms | 2.47 ms | 3.79 ms |
| DeepSeek V4 Pro | 1.11 ms | 1.10 ms | 1.11 ms | 1.15 ms | 1.57 ms | 1.76 ms | 1.92 ms | 2.23 ms | 2.37 ms |
| DeepSeek V4.1 Flash | 1.17 ms | 1.17 ms | 1.18 ms | 1.21 ms | 1.22 ms | 1.24 ms | 1.27 ms | 1.33 ms | 1.51 ms |
| Kimi K3 | 0.78 ms | 0.80 ms | 0.89 ms | 0.94 ms | 1.01 ms | 1.24 ms | 1.51 ms | 2.25 ms | 4.02 ms |
| Kimi K3.1 Pro | 1.66 ms | 1.71 ms | 1.76 ms | 1.96 ms | 1.96 ms | 2.04 ms | 2.23 ms | 2.51 ms | 2.93 ms |
| GLM-5.3-Flash | 0.28 ms | 0.29 ms | 0.29 ms | 0.30 ms | 0.32 ms | 0.36 ms | 0.37 ms | 0.40 ms | 0.45 ms |
| Qwen3.8 Flash Next | 0.43 ms | 0.43 ms | 0.43 ms | 0.45 ms | 0.47 ms | 0.52 ms | 0.54 ms | 0.58 ms | 0.64 ms |
| Qwen3.8 27B | 0.47 ms | 0.51 ms | 0.59 ms | 0.66 ms | 0.81 ms | 1.10 ms | 1.70 ms | 2.91 ms | 5.29 ms |

*Per model by context length (Batch 1, With other ops)*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 3.60 ms | 3.69 ms | 3.87 ms | 4.17 ms | 4.75 ms | 5.90 ms | 8.25 ms | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 2.53 ms | 2.55 ms | 2.58 ms | 2.62 ms | 2.72 ms | 2.91 ms | 3.24 ms | 3.90 ms | 5.27 ms |
| Inkling | 5.10 ms | 5.17 ms | 5.25 ms | 5.40 ms | 5.64 ms | 6.05 ms | 7.16 ms | 9.39 ms | 13.9 ms |
| GLM-5.3 | 7.82 ms | 7.85 ms | 7.90 ms | 8.08 ms | 8.09 ms | 8.14 ms | 8.31 ms | 8.56 ms | 8.94 ms |
| MiniMax-M3.1 Flash | 3.16 ms | 3.23 ms | 3.24 ms | 3.32 ms | 3.40 ms | 3.58 ms | 3.91 ms | 4.55 ms | 5.87 ms |
| DeepSeek V4 Pro | 5.99 ms | 5.98 ms | 5.99 ms | 6.03 ms | 6.06 ms | 6.25 ms | 6.41 ms | 6.72 ms | 6.86 ms |
| DeepSeek V4.1 Flash | 3.33 ms | 3.34 ms | 3.35 ms | 3.37 ms | 3.39 ms | 3.40 ms | 3.44 ms | 3.51 ms | 3.68 ms |
| Kimi K3 | 13.7 ms | 13.7 ms | 13.8 ms | 13.9 ms | 13.9 ms | 14.2 ms | 14.4 ms | 15.2 ms | 16.9 ms |
| Kimi K3.1 Pro | 15.3 ms | 15.3 ms | 15.4 ms | 15.6 ms | 15.6 ms | 15.6 ms | 15.8 ms | 16.1 ms | 16.5 ms |
| GLM-5.3-Flash | 3.69 ms | 3.69 ms | 3.70 ms | 3.71 ms | 3.73 ms | 3.77 ms | 3.78 ms | 3.80 ms | 3.86 ms |
| Qwen3.8 Flash Next | 2.86 ms | 2.86 ms | 2.86 ms | 2.88 ms | 2.90 ms | 2.95 ms | 2.98 ms | 3.01 ms | 3.07 ms |
| Qwen3.8 27B | 3.20 ms | 3.24 ms | 3.32 ms | 3.39 ms | 3.54 ms | 3.83 ms | 4.43 ms | 5.64 ms | 8.02 ms |

*Per model by context length (Batch 32, Attention only)*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 2.86 ms | 5.23 ms | 9.96 ms | 19.5 ms | 38.1 ms | 75.9 ms | 152 ms | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 0.99 ms | 1.31 ms | 2.08 ms | 3.60 ms | 6.64 ms | 12.7 ms | 24.7 ms | 48.8 ms | 94.5 ms |
| Inkling | 5.04 ms | 6.25 ms | 8.83 ms | 14.0 ms | 23.6 ms | 42.3 ms | 79.9 ms | 155 ms | 303 ms |
| GLM-5.3 | 1.31 ms | 1.38 ms | 1.69 ms | 1.97 ms | 2.50 ms | 3.47 ms | 5.38 ms | 9.19 ms | 16.8 ms |
| MiniMax-M3.1 Flash | 2.15 ms | 2.27 ms | 2.54 ms | 3.20 ms | 4.16 ms | 6.44 ms | 11.0 ms | 20.3 ms | 39.1 ms |
| DeepSeek V4 Pro | 1.38 ms | 1.38 ms | 1.40 ms | 2.14 ms | 2.46 ms | 3.09 ms | 4.24 ms | 6.44 ms | 11.0 ms |
| DeepSeek V4.1 Flash | 1.08 ms | 1.10 ms | 1.16 ms | 1.27 ms | 1.43 ms | 1.72 ms | 2.33 ms | 3.52 ms | 5.85 ms |
| Kimi K3 | 5.53 ms | 5.83 ms | 6.71 ms | 8.47 ms | 12.3 ms | 17.9 ms | 35.6 ms | 66.2 ms | 128 ms |
| Kimi K3.1 Pro | 6.88 ms | 6.99 ms | 7.30 ms | 7.64 ms | 8.24 ms | 9.38 ms | 11.5 ms | 15.9 ms | 24.5 ms |
| GLM-5.3-Flash | 2.46 ms | 2.46 ms | 2.49 ms | 2.51 ms | 2.62 ms | 2.79 ms | 3.05 ms | 3.55 ms | 4.60 ms |
| Qwen3.8 Flash Next | 2.06 ms | 2.06 ms | 2.07 ms | 2.10 ms | 2.17 ms | 2.33 ms | 2.64 ms | 3.11 ms | 4.09 ms |
| Qwen3.8 27B | 2.44 ms | 3.03 ms | 4.22 ms | 6.56 ms | 11.3 ms | 20.7 ms | 39.5 ms | 77.1 ms | 152 ms |

*Per model by context length (Batch 32, With other ops)*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 5.82 ms | 8.19 ms | 12.9 ms | 22.5 ms | 41.1 ms | 78.9 ms | 154 ms | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 2.81 ms | 3.13 ms | 3.89 ms | 5.42 ms | 8.46 ms | 14.5 ms | 26.6 ms | 50.6 ms | 96.3 ms |
| Inkling | 8.77 ms | 9.98 ms | 12.6 ms | 17.7 ms | 27.3 ms | 46.1 ms | 83.6 ms | 159 ms | 307 ms |
| GLM-5.3 | 8.30 ms | 8.38 ms | 8.68 ms | 8.96 ms | 9.49 ms | 10.5 ms | 12.4 ms | 16.2 ms | 23.7 ms |
| MiniMax-M3.1 Flash | 4.32 ms | 4.44 ms | 4.71 ms | 5.38 ms | 6.33 ms | 8.62 ms | 13.2 ms | 22.5 ms | 41.3 ms |
| DeepSeek V4 Pro | 6.57 ms | 6.57 ms | 6.60 ms | 6.92 ms | 7.24 ms | 7.87 ms | 9.02 ms | 11.2 ms | 15.8 ms |
| DeepSeek V4.1 Flash | 3.17 ms | 3.19 ms | 3.26 ms | 3.36 ms | 3.52 ms | 3.83 ms | 4.45 ms | 5.67 ms | 8.08 ms |
| Kimi K3 | 18.7 ms | 19.0 ms | 19.9 ms | 21.7 ms | 25.5 ms | 31.1 ms | 48.8 ms | 79.4 ms | 141 ms |
| Kimi K3.1 Pro | 20.8 ms | 20.9 ms | 21.2 ms | 21.5 ms | 22.1 ms | 23.3 ms | 25.4 ms | 29.8 ms | 38.4 ms |
| GLM-5.3-Flash | 6.01 ms | 6.01 ms | 6.04 ms | 6.07 ms | 6.18 ms | 6.35 ms | 6.61 ms | 7.10 ms | 8.16 ms |
| Qwen3.8 Flash Next | 4.51 ms | 4.52 ms | 4.52 ms | 4.56 ms | 4.63 ms | 4.78 ms | 5.09 ms | 5.56 ms | 6.54 ms |
| Qwen3.8 27B | 5.27 ms | 5.87 ms | 7.05 ms | 9.40 ms | 14.1 ms | 23.5 ms | 42.3 ms | 79.9 ms | 155 ms |
