# Measured attention time of one decode step

All the data behind this figure in "Beyond GQA: Characterizing Modern Attention Variants with ServingStudio".

ms per decode step, whole model · B200, DP attention (all heads on each GPU). Hover a bar for the backend.

*Total per model by context length (Batch 1, Attention only)*

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

*Breakdown by kind (Batch 1, Attention only)*

| Model | Context | Global attention | Window attention | Linear attention | Sparse attention | Indexer |
|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 4K | 0.76 ms | — | — | — | — |
| Qwen3-Coder-480B-A35B | 8K | 0.85 ms | — | — | — | — |
| Qwen3-Coder-480B-A35B | 16K | 1.04 ms | — | — | — | — |
| Qwen3-Coder-480B-A35B | 32K | 1.33 ms | — | — | — | — |
| Qwen3-Coder-480B-A35B | 64K | 1.92 ms | — | — | — | — |
| Qwen3-Coder-480B-A35B | 128K | 3.06 ms | — | — | — | — |
| Qwen3-Coder-480B-A35B | 256K | 5.41 ms | — | — | — | — |
| MiMo V2.6 Flash | 4K | 0.18 ms | 0.62 ms | — | — | — |
| MiMo V2.6 Flash | 8K | 0.20 ms | 0.62 ms | — | — | — |
| MiMo V2.6 Flash | 16K | 0.22 ms | 0.62 ms | — | — | — |
| MiMo V2.6 Flash | 32K | 0.27 ms | 0.62 ms | — | — | — |
| MiMo V2.6 Flash | 64K | 0.36 ms | 0.62 ms | — | — | — |
| MiMo V2.6 Flash | 128K | 0.55 ms | 0.62 ms | — | — | — |
| MiMo V2.6 Flash | 256K | 0.89 ms | 0.62 ms | — | — | — |
| MiMo V2.6 Flash | 512K | 1.55 ms | 0.62 ms | — | — | — |
| MiMo V2.6 Flash | 1M | 2.92 ms | 0.62 ms | — | — | — |
| Inkling | 4K | 0.28 ms | 1.16 ms | — | — | — |
| Inkling | 8K | 0.34 ms | 1.16 ms | — | — | — |
| Inkling | 16K | 0.42 ms | 1.16 ms | — | — | — |
| Inkling | 32K | 0.57 ms | 1.16 ms | — | — | — |
| Inkling | 64K | 0.81 ms | 1.17 ms | — | — | — |
| Inkling | 128K | 1.21 ms | 1.17 ms | — | — | — |
| Inkling | 256K | 2.33 ms | 1.17 ms | — | — | — |
| Inkling | 512K | 4.55 ms | 1.17 ms | — | — | — |
| Inkling | 1M | 9.02 ms | 1.17 ms | — | — | — |
| GLM-5.3 | 4K | — | — | — | 0.50 ms | 0.16 ms |
| GLM-5.3 | 8K | — | — | — | 0.50 ms | 0.19 ms |
| GLM-5.3 | 16K | — | — | — | 0.49 ms | 0.25 ms |
| GLM-5.3 | 32K | — | — | — | 0.51 ms | 0.41 ms |
| GLM-5.3 | 64K | — | — | — | 0.50 ms | 0.42 ms |
| GLM-5.3 | 128K | — | — | — | 0.49 ms | 0.48 ms |
| GLM-5.3 | 256K | — | — | — | 0.49 ms | 0.66 ms |
| GLM-5.3 | 512K | — | — | — | 0.50 ms | 0.90 ms |
| GLM-5.3 | 1M | — | — | — | 0.50 ms | 1.27 ms |
| MiniMax-M3.1 Flash | 4K | — | — | — | 0.54 ms | 0.54 ms |
| MiniMax-M3.1 Flash | 8K | — | — | — | 0.54 ms | 0.61 ms |
| MiniMax-M3.1 Flash | 16K | — | — | — | 0.54 ms | 0.62 ms |
| MiniMax-M3.1 Flash | 32K | — | — | — | 0.54 ms | 0.69 ms |
| MiniMax-M3.1 Flash | 64K | — | — | — | 0.54 ms | 0.78 ms |
| MiniMax-M3.1 Flash | 128K | — | — | — | 0.54 ms | 0.96 ms |
| MiniMax-M3.1 Flash | 256K | — | — | — | 0.55 ms | 1.28 ms |
| MiniMax-M3.1 Flash | 512K | — | — | — | 0.54 ms | 1.93 ms |
| MiniMax-M3.1 Flash | 1M | — | — | — | 0.54 ms | 3.25 ms |
| DeepSeek V4 Pro | 4K | 0.51 ms | — | — | 0.60 ms | — |
| DeepSeek V4 Pro | 8K | 0.50 ms | — | — | 0.60 ms | — |
| DeepSeek V4 Pro | 16K | 0.51 ms | — | — | 0.60 ms | — |
| DeepSeek V4 Pro | 32K | 0.55 ms | — | — | 0.60 ms | — |
| DeepSeek V4 Pro | 64K | 0.56 ms | — | — | 0.60 ms | 0.41 ms |
| DeepSeek V4 Pro | 128K | 0.63 ms | — | — | 0.60 ms | 0.52 ms |
| DeepSeek V4 Pro | 256K | 0.75 ms | — | — | 0.60 ms | 0.57 ms |
| DeepSeek V4 Pro | 512K | 1.00 ms | — | — | 0.60 ms | 0.63 ms |
| DeepSeek V4 Pro | 1M | 1.01 ms | — | — | 0.60 ms | 0.76 ms |
| DeepSeek V4.1 Flash | 4K | — | 0.02 ms | — | 1.08 ms | 0.06 ms |
| DeepSeek V4.1 Flash | 8K | — | 0.02 ms | — | 1.08 ms | 0.07 ms |
| DeepSeek V4.1 Flash | 16K | — | 0.02 ms | — | 1.08 ms | 0.08 ms |
| DeepSeek V4.1 Flash | 32K | — | 0.02 ms | — | 1.08 ms | 0.10 ms |
| DeepSeek V4.1 Flash | 64K | — | 0.02 ms | — | 1.08 ms | 0.12 ms |
| DeepSeek V4.1 Flash | 128K | — | 0.02 ms | — | 1.08 ms | 0.13 ms |
| DeepSeek V4.1 Flash | 256K | — | 0.02 ms | — | 1.08 ms | 0.17 ms |
| DeepSeek V4.1 Flash | 512K | — | 0.02 ms | — | 1.08 ms | 0.23 ms |
| DeepSeek V4.1 Flash | 1M | — | 0.02 ms | — | 1.08 ms | 0.40 ms |
| Kimi K3 | 4K | 0.27 ms | — | 0.51 ms | — | — |
| Kimi K3 | 8K | 0.29 ms | — | 0.51 ms | — | — |
| Kimi K3 | 16K | 0.37 ms | — | 0.51 ms | — | — |
| Kimi K3 | 32K | 0.43 ms | — | 0.51 ms | — | — |
| Kimi K3 | 64K | 0.50 ms | — | 0.51 ms | — | — |
| Kimi K3 | 128K | 0.73 ms | — | 0.51 ms | — | — |
| Kimi K3 | 256K | 1.00 ms | — | 0.51 ms | — | — |
| Kimi K3 | 512K | 1.74 ms | — | 0.51 ms | — | — |
| Kimi K3 | 1M | 3.50 ms | — | 0.51 ms | — | — |
| Kimi K3.1 Pro | 4K | — | — | 0.51 ms | 0.96 ms | 0.18 ms |
| Kimi K3.1 Pro | 8K | — | — | 0.51 ms | 0.98 ms | 0.22 ms |
| Kimi K3.1 Pro | 16K | — | — | 0.51 ms | 0.96 ms | 0.28 ms |
| Kimi K3.1 Pro | 32K | — | — | 0.51 ms | 0.98 ms | 0.47 ms |
| Kimi K3.1 Pro | 64K | — | — | 0.51 ms | 0.96 ms | 0.49 ms |
| Kimi K3.1 Pro | 128K | — | — | 0.51 ms | 0.97 ms | 0.55 ms |
| Kimi K3.1 Pro | 256K | — | — | 0.51 ms | 0.96 ms | 0.75 ms |
| Kimi K3.1 Pro | 512K | — | — | 0.51 ms | 0.97 ms | 1.03 ms |
| Kimi K3.1 Pro | 1M | — | — | 0.51 ms | 0.96 ms | 1.45 ms |
| GLM-5.3-Flash | 4K | — | — | 0.12 ms | 0.08 ms | 0.08 ms |
| GLM-5.3-Flash | 8K | — | — | 0.12 ms | 0.08 ms | 0.08 ms |
| GLM-5.3-Flash | 16K | — | — | 0.12 ms | 0.08 ms | 0.09 ms |
| GLM-5.3-Flash | 32K | — | — | 0.12 ms | 0.08 ms | 0.10 ms |
| GLM-5.3-Flash | 64K | — | — | 0.12 ms | 0.08 ms | 0.12 ms |
| GLM-5.3-Flash | 128K | — | — | 0.12 ms | 0.08 ms | 0.16 ms |
| GLM-5.3-Flash | 256K | — | — | 0.12 ms | 0.08 ms | 0.17 ms |
| GLM-5.3-Flash | 512K | — | — | 0.12 ms | 0.08 ms | 0.19 ms |
| GLM-5.3-Flash | 1M | — | — | 0.12 ms | 0.08 ms | 0.25 ms |
| Qwen3.8 Flash Next | 4K | — | — | 0.21 ms | 0.13 ms | 0.09 ms |
| Qwen3.8 Flash Next | 8K | — | — | 0.21 ms | 0.12 ms | 0.09 ms |
| Qwen3.8 Flash Next | 16K | — | — | 0.21 ms | 0.13 ms | 0.10 ms |
| Qwen3.8 Flash Next | 32K | — | — | 0.21 ms | 0.13 ms | 0.11 ms |
| Qwen3.8 Flash Next | 64K | — | — | 0.21 ms | 0.13 ms | 0.13 ms |
| Qwen3.8 Flash Next | 128K | — | — | 0.21 ms | 0.13 ms | 0.18 ms |
| Qwen3.8 Flash Next | 256K | — | — | 0.21 ms | 0.13 ms | 0.21 ms |
| Qwen3.8 Flash Next | 512K | — | — | 0.21 ms | 0.13 ms | 0.24 ms |
| Qwen3.8 Flash Next | 1M | — | — | 0.21 ms | 0.13 ms | 0.30 ms |
| Qwen3.8 27B | 4K | 0.19 ms | — | 0.28 ms | — | — |
| Qwen3.8 27B | 8K | 0.23 ms | — | 0.28 ms | — | — |
| Qwen3.8 27B | 16K | 0.31 ms | — | 0.28 ms | — | — |
| Qwen3.8 27B | 32K | 0.38 ms | — | 0.28 ms | — | — |
| Qwen3.8 27B | 64K | 0.53 ms | — | 0.28 ms | — | — |
| Qwen3.8 27B | 128K | 0.82 ms | — | 0.28 ms | — | — |
| Qwen3.8 27B | 256K | 1.42 ms | — | 0.28 ms | — | — |
| Qwen3.8 27B | 512K | 2.63 ms | — | 0.28 ms | — | — |
| Qwen3.8 27B | 1M | 5.01 ms | — | 0.28 ms | — | — |

*Total per model by context length (Batch 1, With other ops)*

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

*Breakdown by kind (Batch 1, With other ops)*

| Model | Context | Global attention | Window attention | Linear attention | Sparse attention | Indexer | QKV and output projections | Norms, RoPE and other ops |
|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 4K | 0.76 ms | — | — | — | — | 2.61 ms | 0.23 ms |
| Qwen3-Coder-480B-A35B | 8K | 0.85 ms | — | — | — | — | 2.61 ms | 0.23 ms |
| Qwen3-Coder-480B-A35B | 16K | 1.04 ms | — | — | — | — | 2.61 ms | 0.23 ms |
| Qwen3-Coder-480B-A35B | 32K | 1.33 ms | — | — | — | — | 2.61 ms | 0.23 ms |
| Qwen3-Coder-480B-A35B | 64K | 1.92 ms | — | — | — | — | 2.61 ms | 0.23 ms |
| Qwen3-Coder-480B-A35B | 128K | 3.06 ms | — | — | — | — | 2.61 ms | 0.23 ms |
| Qwen3-Coder-480B-A35B | 256K | 5.41 ms | — | — | — | — | 2.61 ms | 0.23 ms |
| MiMo V2.6 Flash | 4K | 0.18 ms | 0.62 ms | — | — | — | 1.52 ms | 0.22 ms |
| MiMo V2.6 Flash | 8K | 0.20 ms | 0.62 ms | — | — | — | 1.52 ms | 0.22 ms |
| MiMo V2.6 Flash | 16K | 0.22 ms | 0.62 ms | — | — | — | 1.52 ms | 0.22 ms |
| MiMo V2.6 Flash | 32K | 0.27 ms | 0.62 ms | — | — | — | 1.52 ms | 0.22 ms |
| MiMo V2.6 Flash | 64K | 0.36 ms | 0.62 ms | — | — | — | 1.52 ms | 0.22 ms |
| MiMo V2.6 Flash | 128K | 0.55 ms | 0.62 ms | — | — | — | 1.52 ms | 0.22 ms |
| MiMo V2.6 Flash | 256K | 0.89 ms | 0.62 ms | — | — | — | 1.52 ms | 0.22 ms |
| MiMo V2.6 Flash | 512K | 1.55 ms | 0.62 ms | — | — | — | 1.52 ms | 0.22 ms |
| MiMo V2.6 Flash | 1M | 2.92 ms | 0.62 ms | — | — | — | 1.52 ms | 0.22 ms |
| Inkling | 4K | 0.28 ms | 1.16 ms | — | — | — | 3.33 ms | 0.34 ms |
| Inkling | 8K | 0.34 ms | 1.16 ms | — | — | — | 3.33 ms | 0.34 ms |
| Inkling | 16K | 0.42 ms | 1.16 ms | — | — | — | 3.33 ms | 0.34 ms |
| Inkling | 32K | 0.57 ms | 1.16 ms | — | — | — | 3.33 ms | 0.34 ms |
| Inkling | 64K | 0.81 ms | 1.17 ms | — | — | — | 3.33 ms | 0.34 ms |
| Inkling | 128K | 1.21 ms | 1.17 ms | — | — | — | 3.33 ms | 0.34 ms |
| Inkling | 256K | 2.33 ms | 1.17 ms | — | — | — | 3.33 ms | 0.34 ms |
| Inkling | 512K | 4.55 ms | 1.17 ms | — | — | — | 3.33 ms | 0.34 ms |
| Inkling | 1M | 9.02 ms | 1.17 ms | — | — | — | 3.33 ms | 0.34 ms |
| GLM-5.3 | 4K | — | — | — | 0.50 ms | 0.16 ms | 5.91 ms | 1.26 ms |
| GLM-5.3 | 8K | — | — | — | 0.50 ms | 0.19 ms | 5.91 ms | 1.26 ms |
| GLM-5.3 | 16K | — | — | — | 0.49 ms | 0.25 ms | 5.91 ms | 1.26 ms |
| GLM-5.3 | 32K | — | — | — | 0.51 ms | 0.41 ms | 5.91 ms | 1.26 ms |
| GLM-5.3 | 64K | — | — | — | 0.50 ms | 0.42 ms | 5.91 ms | 1.26 ms |
| GLM-5.3 | 128K | — | — | — | 0.49 ms | 0.48 ms | 5.91 ms | 1.26 ms |
| GLM-5.3 | 256K | — | — | — | 0.49 ms | 0.66 ms | 5.91 ms | 1.26 ms |
| GLM-5.3 | 512K | — | — | — | 0.50 ms | 0.90 ms | 5.91 ms | 1.26 ms |
| GLM-5.3 | 1M | — | — | — | 0.50 ms | 1.27 ms | 5.91 ms | 1.26 ms |
| MiniMax-M3.1 Flash | 4K | — | — | — | 0.54 ms | 0.54 ms | 1.61 ms | 0.47 ms |
| MiniMax-M3.1 Flash | 8K | — | — | — | 0.54 ms | 0.61 ms | 1.61 ms | 0.47 ms |
| MiniMax-M3.1 Flash | 16K | — | — | — | 0.54 ms | 0.62 ms | 1.61 ms | 0.47 ms |
| MiniMax-M3.1 Flash | 32K | — | — | — | 0.54 ms | 0.69 ms | 1.61 ms | 0.47 ms |
| MiniMax-M3.1 Flash | 64K | — | — | — | 0.54 ms | 0.78 ms | 1.61 ms | 0.47 ms |
| MiniMax-M3.1 Flash | 128K | — | — | — | 0.54 ms | 0.96 ms | 1.61 ms | 0.47 ms |
| MiniMax-M3.1 Flash | 256K | — | — | — | 0.55 ms | 1.28 ms | 1.61 ms | 0.47 ms |
| MiniMax-M3.1 Flash | 512K | — | — | — | 0.54 ms | 1.93 ms | 1.61 ms | 0.47 ms |
| MiniMax-M3.1 Flash | 1M | — | — | — | 0.54 ms | 3.25 ms | 1.61 ms | 0.47 ms |
| DeepSeek V4 Pro | 4K | 0.51 ms | — | — | 0.60 ms | — | 4.35 ms | 0.53 ms |
| DeepSeek V4 Pro | 8K | 0.50 ms | — | — | 0.60 ms | — | 4.35 ms | 0.53 ms |
| DeepSeek V4 Pro | 16K | 0.51 ms | — | — | 0.60 ms | — | 4.35 ms | 0.53 ms |
| DeepSeek V4 Pro | 32K | 0.55 ms | — | — | 0.60 ms | — | 4.35 ms | 0.53 ms |
| DeepSeek V4 Pro | 64K | 0.56 ms | — | — | 0.60 ms | 0.41 ms | 3.98 ms | 0.51 ms |
| DeepSeek V4 Pro | 128K | 0.63 ms | — | — | 0.60 ms | 0.52 ms | 3.98 ms | 0.51 ms |
| DeepSeek V4 Pro | 256K | 0.75 ms | — | — | 0.60 ms | 0.57 ms | 3.98 ms | 0.51 ms |
| DeepSeek V4 Pro | 512K | 1.00 ms | — | — | 0.60 ms | 0.63 ms | 3.98 ms | 0.51 ms |
| DeepSeek V4 Pro | 1M | 1.01 ms | — | — | 0.60 ms | 0.76 ms | 3.98 ms | 0.51 ms |
| DeepSeek V4.1 Flash | 4K | — | 0.02 ms | — | 1.08 ms | 0.06 ms | 1.57 ms | 0.59 ms |
| DeepSeek V4.1 Flash | 8K | — | 0.02 ms | — | 1.08 ms | 0.07 ms | 1.57 ms | 0.59 ms |
| DeepSeek V4.1 Flash | 16K | — | 0.02 ms | — | 1.08 ms | 0.08 ms | 1.57 ms | 0.59 ms |
| DeepSeek V4.1 Flash | 32K | — | 0.02 ms | — | 1.08 ms | 0.10 ms | 1.57 ms | 0.59 ms |
| DeepSeek V4.1 Flash | 64K | — | 0.02 ms | — | 1.08 ms | 0.12 ms | 1.57 ms | 0.59 ms |
| DeepSeek V4.1 Flash | 128K | — | 0.02 ms | — | 1.08 ms | 0.13 ms | 1.57 ms | 0.59 ms |
| DeepSeek V4.1 Flash | 256K | — | 0.02 ms | — | 1.08 ms | 0.17 ms | 1.57 ms | 0.59 ms |
| DeepSeek V4.1 Flash | 512K | — | 0.02 ms | — | 1.08 ms | 0.23 ms | 1.57 ms | 0.60 ms |
| DeepSeek V4.1 Flash | 1M | — | 0.02 ms | — | 1.08 ms | 0.40 ms | 1.57 ms | 0.60 ms |
| Kimi K3 | 4K | 0.27 ms | — | 0.51 ms | — | — | 12.6 ms | 0.31 ms |
| Kimi K3 | 8K | 0.29 ms | — | 0.51 ms | — | — | 12.6 ms | 0.31 ms |
| Kimi K3 | 16K | 0.37 ms | — | 0.51 ms | — | — | 12.6 ms | 0.31 ms |
| Kimi K3 | 32K | 0.43 ms | — | 0.51 ms | — | — | 12.6 ms | 0.31 ms |
| Kimi K3 | 64K | 0.50 ms | — | 0.51 ms | — | — | 12.6 ms | 0.31 ms |
| Kimi K3 | 128K | 0.73 ms | — | 0.51 ms | — | — | 12.6 ms | 0.31 ms |
| Kimi K3 | 256K | 1.00 ms | — | 0.51 ms | — | — | 12.6 ms | 0.31 ms |
| Kimi K3 | 512K | 1.74 ms | — | 0.51 ms | — | — | 12.6 ms | 0.31 ms |
| Kimi K3 | 1M | 3.50 ms | — | 0.51 ms | — | — | 12.6 ms | 0.31 ms |
| Kimi K3.1 Pro | 4K | — | — | 0.51 ms | 0.96 ms | 0.18 ms | 13.1 ms | 0.54 ms |
| Kimi K3.1 Pro | 8K | — | — | 0.51 ms | 0.98 ms | 0.22 ms | 13.1 ms | 0.54 ms |
| Kimi K3.1 Pro | 16K | — | — | 0.51 ms | 0.96 ms | 0.28 ms | 13.1 ms | 0.54 ms |
| Kimi K3.1 Pro | 32K | — | — | 0.51 ms | 0.98 ms | 0.47 ms | 13.1 ms | 0.54 ms |
| Kimi K3.1 Pro | 64K | — | — | 0.51 ms | 0.96 ms | 0.49 ms | 13.1 ms | 0.54 ms |
| Kimi K3.1 Pro | 128K | — | — | 0.51 ms | 0.97 ms | 0.55 ms | 13.1 ms | 0.54 ms |
| Kimi K3.1 Pro | 256K | — | — | 0.51 ms | 0.96 ms | 0.75 ms | 13.1 ms | 0.54 ms |
| Kimi K3.1 Pro | 512K | — | — | 0.51 ms | 0.97 ms | 1.03 ms | 13.1 ms | 0.54 ms |
| Kimi K3.1 Pro | 1M | — | — | 0.51 ms | 0.96 ms | 1.45 ms | 13.1 ms | 0.54 ms |
| GLM-5.3-Flash | 4K | — | — | 0.12 ms | 0.08 ms | 0.08 ms | 2.82 ms | 0.59 ms |
| GLM-5.3-Flash | 8K | — | — | 0.12 ms | 0.08 ms | 0.08 ms | 2.82 ms | 0.59 ms |
| GLM-5.3-Flash | 16K | — | — | 0.12 ms | 0.08 ms | 0.09 ms | 2.82 ms | 0.59 ms |
| GLM-5.3-Flash | 32K | — | — | 0.12 ms | 0.08 ms | 0.10 ms | 2.82 ms | 0.59 ms |
| GLM-5.3-Flash | 64K | — | — | 0.12 ms | 0.08 ms | 0.12 ms | 2.82 ms | 0.59 ms |
| GLM-5.3-Flash | 128K | — | — | 0.12 ms | 0.08 ms | 0.16 ms | 2.82 ms | 0.59 ms |
| GLM-5.3-Flash | 256K | — | — | 0.12 ms | 0.08 ms | 0.17 ms | 2.82 ms | 0.59 ms |
| GLM-5.3-Flash | 512K | — | — | 0.12 ms | 0.08 ms | 0.19 ms | 2.82 ms | 0.59 ms |
| GLM-5.3-Flash | 1M | — | — | 0.12 ms | 0.08 ms | 0.25 ms | 2.82 ms | 0.59 ms |
| Qwen3.8 Flash Next | 4K | — | — | 0.21 ms | 0.13 ms | 0.09 ms | 1.52 ms | 0.91 ms |
| Qwen3.8 Flash Next | 8K | — | — | 0.21 ms | 0.12 ms | 0.09 ms | 1.52 ms | 0.91 ms |
| Qwen3.8 Flash Next | 16K | — | — | 0.21 ms | 0.13 ms | 0.10 ms | 1.52 ms | 0.91 ms |
| Qwen3.8 Flash Next | 32K | — | — | 0.21 ms | 0.13 ms | 0.11 ms | 1.52 ms | 0.91 ms |
| Qwen3.8 Flash Next | 64K | — | — | 0.21 ms | 0.13 ms | 0.13 ms | 1.52 ms | 0.91 ms |
| Qwen3.8 Flash Next | 128K | — | — | 0.21 ms | 0.13 ms | 0.18 ms | 1.52 ms | 0.91 ms |
| Qwen3.8 Flash Next | 256K | — | — | 0.21 ms | 0.13 ms | 0.21 ms | 1.52 ms | 0.91 ms |
| Qwen3.8 Flash Next | 512K | — | — | 0.21 ms | 0.13 ms | 0.24 ms | 1.52 ms | 0.91 ms |
| Qwen3.8 Flash Next | 1M | — | — | 0.21 ms | 0.13 ms | 0.30 ms | 1.52 ms | 0.91 ms |
| Qwen3.8 27B | 4K | 0.19 ms | — | 0.28 ms | — | — | 1.70 ms | 1.03 ms |
| Qwen3.8 27B | 8K | 0.23 ms | — | 0.28 ms | — | — | 1.70 ms | 1.03 ms |
| Qwen3.8 27B | 16K | 0.31 ms | — | 0.28 ms | — | — | 1.70 ms | 1.03 ms |
| Qwen3.8 27B | 32K | 0.38 ms | — | 0.28 ms | — | — | 1.70 ms | 1.03 ms |
| Qwen3.8 27B | 64K | 0.53 ms | — | 0.28 ms | — | — | 1.70 ms | 1.03 ms |
| Qwen3.8 27B | 128K | 0.82 ms | — | 0.28 ms | — | — | 1.70 ms | 1.03 ms |
| Qwen3.8 27B | 256K | 1.42 ms | — | 0.28 ms | — | — | 1.70 ms | 1.03 ms |
| Qwen3.8 27B | 512K | 2.63 ms | — | 0.28 ms | — | — | 1.70 ms | 1.03 ms |
| Qwen3.8 27B | 1M | 5.01 ms | — | 0.28 ms | — | — | 1.70 ms | 1.03 ms |

*Total per model by context length (Batch 32, Attention only)*

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

*Breakdown by kind (Batch 32, Attention only)*

| Model | Context | Global attention | Window attention | Linear attention | Sparse attention | Indexer |
|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 4K | 2.86 ms | — | — | — | — |
| Qwen3-Coder-480B-A35B | 8K | 5.23 ms | — | — | — | — |
| Qwen3-Coder-480B-A35B | 16K | 9.96 ms | — | — | — | — |
| Qwen3-Coder-480B-A35B | 32K | 19.5 ms | — | — | — | — |
| Qwen3-Coder-480B-A35B | 64K | 38.1 ms | — | — | — | — |
| Qwen3-Coder-480B-A35B | 128K | 75.9 ms | — | — | — | — |
| Qwen3-Coder-480B-A35B | 256K | 152 ms | — | — | — | — |
| MiMo V2.6 Flash | 4K | 0.40 ms | 0.59 ms | — | — | — |
| MiMo V2.6 Flash | 8K | 0.72 ms | 0.59 ms | — | — | — |
| MiMo V2.6 Flash | 16K | 1.49 ms | 0.59 ms | — | — | — |
| MiMo V2.6 Flash | 32K | 3.01 ms | 0.59 ms | — | — | — |
| MiMo V2.6 Flash | 64K | 6.05 ms | 0.59 ms | — | — | — |
| MiMo V2.6 Flash | 128K | 12.1 ms | 0.59 ms | — | — | — |
| MiMo V2.6 Flash | 256K | 24.2 ms | 0.59 ms | — | — | — |
| MiMo V2.6 Flash | 512K | 48.2 ms | 0.59 ms | — | — | — |
| MiMo V2.6 Flash | 1M | 93.9 ms | 0.59 ms | — | — | — |
| Inkling | 4K | 1.54 ms | 3.50 ms | — | — | — |
| Inkling | 8K | 2.74 ms | 3.50 ms | — | — | — |
| Inkling | 16K | 5.33 ms | 3.50 ms | — | — | — |
| Inkling | 32K | 10.5 ms | 3.50 ms | — | — | — |
| Inkling | 64K | 19.2 ms | 4.43 ms | — | — | — |
| Inkling | 128K | 37.9 ms | 4.43 ms | — | — | — |
| Inkling | 256K | 75.4 ms | 4.43 ms | — | — | — |
| Inkling | 512K | 151 ms | 4.43 ms | — | — | — |
| Inkling | 1M | 299 ms | 4.43 ms | — | — | — |
| GLM-5.3 | 4K | — | — | — | 1.10 ms | 0.21 ms |
| GLM-5.3 | 8K | — | — | — | 1.09 ms | 0.29 ms |
| GLM-5.3 | 16K | — | — | — | 1.12 ms | 0.57 ms |
| GLM-5.3 | 32K | — | — | — | 1.11 ms | 0.86 ms |
| GLM-5.3 | 64K | — | — | — | 1.10 ms | 1.40 ms |
| GLM-5.3 | 128K | — | — | — | 1.09 ms | 2.38 ms |
| GLM-5.3 | 256K | — | — | — | 1.11 ms | 4.28 ms |
| GLM-5.3 | 512K | — | — | — | 1.12 ms | 8.07 ms |
| GLM-5.3 | 1M | — | — | — | 1.11 ms | 15.6 ms |
| MiniMax-M3.1 Flash | 4K | — | — | — | 1.26 ms | 0.89 ms |
| MiniMax-M3.1 Flash | 8K | — | — | — | 1.24 ms | 1.03 ms |
| MiniMax-M3.1 Flash | 16K | — | — | — | 1.24 ms | 1.30 ms |
| MiniMax-M3.1 Flash | 32K | — | — | — | 1.24 ms | 1.97 ms |
| MiniMax-M3.1 Flash | 64K | — | — | — | 1.24 ms | 2.92 ms |
| MiniMax-M3.1 Flash | 128K | — | — | — | 1.24 ms | 5.20 ms |
| MiniMax-M3.1 Flash | 256K | — | — | — | 1.26 ms | 9.77 ms |
| MiniMax-M3.1 Flash | 512K | — | — | — | 1.28 ms | 19.0 ms |
| MiniMax-M3.1 Flash | 1M | — | — | — | 1.28 ms | 37.8 ms |
| DeepSeek V4 Pro | 4K | 0.44 ms | — | — | 0.94 ms | — |
| DeepSeek V4 Pro | 8K | 0.44 ms | — | — | 0.94 ms | — |
| DeepSeek V4 Pro | 16K | 0.47 ms | — | — | 0.94 ms | — |
| DeepSeek V4 Pro | 32K | 0.67 ms | — | — | 0.94 ms | 0.53 ms |
| DeepSeek V4 Pro | 64K | 0.76 ms | — | — | 0.94 ms | 0.76 ms |
| DeepSeek V4 Pro | 128K | 0.96 ms | — | — | 0.94 ms | 1.19 ms |
| DeepSeek V4 Pro | 256K | 1.41 ms | — | — | 0.94 ms | 1.89 ms |
| DeepSeek V4 Pro | 512K | 2.23 ms | — | — | 0.94 ms | 3.27 ms |
| DeepSeek V4 Pro | 1M | 3.87 ms | — | — | 0.94 ms | 6.18 ms |
| DeepSeek V4.1 Flash | 4K | — | 0.02 ms | — | 0.98 ms | 0.08 ms |
| DeepSeek V4.1 Flash | 8K | — | 0.02 ms | — | 0.98 ms | 0.10 ms |
| DeepSeek V4.1 Flash | 16K | — | 0.02 ms | — | 0.98 ms | 0.16 ms |
| DeepSeek V4.1 Flash | 32K | — | 0.02 ms | — | 0.98 ms | 0.26 ms |
| DeepSeek V4.1 Flash | 64K | — | 0.02 ms | — | 0.98 ms | 0.42 ms |
| DeepSeek V4.1 Flash | 128K | — | 0.02 ms | — | 0.98 ms | 0.72 ms |
| DeepSeek V4.1 Flash | 256K | — | 0.02 ms | — | 0.98 ms | 1.33 ms |
| DeepSeek V4.1 Flash | 512K | — | 0.02 ms | — | 0.98 ms | 2.51 ms |
| DeepSeek V4.1 Flash | 1M | — | 0.02 ms | — | 0.98 ms | 4.85 ms |
| Kimi K3 | 4K | 0.67 ms | — | 4.86 ms | — | — |
| Kimi K3 | 8K | 0.97 ms | — | 4.86 ms | — | — |
| Kimi K3 | 16K | 1.86 ms | — | 4.86 ms | — | — |
| Kimi K3 | 32K | 3.62 ms | — | 4.86 ms | — | — |
| Kimi K3 | 64K | 7.42 ms | — | 4.86 ms | — | — |
| Kimi K3 | 128K | 13.1 ms | — | 4.86 ms | — | — |
| Kimi K3 | 256K | 30.8 ms | — | 4.86 ms | — | — |
| Kimi K3 | 512K | 61.3 ms | — | 4.86 ms | — | — |
| Kimi K3 | 1M | 123 ms | — | 4.86 ms | — | — |
| Kimi K3.1 Pro | 4K | — | — | 4.86 ms | 1.79 ms | 0.24 ms |
| Kimi K3.1 Pro | 8K | — | — | 4.86 ms | 1.80 ms | 0.34 ms |
| Kimi K3.1 Pro | 16K | — | — | 4.86 ms | 1.79 ms | 0.65 ms |
| Kimi K3.1 Pro | 32K | — | — | 4.86 ms | 1.80 ms | 0.98 ms |
| Kimi K3.1 Pro | 64K | — | — | 4.86 ms | 1.79 ms | 1.60 ms |
| Kimi K3.1 Pro | 128K | — | — | 4.86 ms | 1.80 ms | 2.72 ms |
| Kimi K3.1 Pro | 256K | — | — | 4.86 ms | 1.79 ms | 4.89 ms |
| Kimi K3.1 Pro | 512K | — | — | 4.86 ms | 1.80 ms | 9.23 ms |
| Kimi K3.1 Pro | 1M | — | — | 4.86 ms | 1.79 ms | 17.9 ms |
| GLM-5.3-Flash | 4K | — | — | 2.12 ms | 0.25 ms | 0.09 ms |
| GLM-5.3-Flash | 8K | — | — | 2.12 ms | 0.24 ms | 0.10 ms |
| GLM-5.3-Flash | 16K | — | — | 2.12 ms | 0.25 ms | 0.11 ms |
| GLM-5.3-Flash | 32K | — | — | 2.12 ms | 0.24 ms | 0.15 ms |
| GLM-5.3-Flash | 64K | — | — | 2.12 ms | 0.24 ms | 0.27 ms |
| GLM-5.3-Flash | 128K | — | — | 2.12 ms | 0.26 ms | 0.42 ms |
| GLM-5.3-Flash | 256K | — | — | 2.12 ms | 0.25 ms | 0.68 ms |
| GLM-5.3-Flash | 512K | — | — | 2.12 ms | 0.26 ms | 1.18 ms |
| GLM-5.3-Flash | 1M | — | — | 2.12 ms | 0.26 ms | 2.22 ms |
| Qwen3.8 Flash Next | 4K | — | — | 1.31 ms | 0.53 ms | 0.22 ms |
| Qwen3.8 Flash Next | 8K | — | — | 1.31 ms | 0.53 ms | 0.22 ms |
| Qwen3.8 Flash Next | 16K | — | — | 1.31 ms | 0.53 ms | 0.23 ms |
| Qwen3.8 Flash Next | 32K | — | — | 1.31 ms | 0.52 ms | 0.27 ms |
| Qwen3.8 Flash Next | 64K | — | — | 1.31 ms | 0.53 ms | 0.33 ms |
| Qwen3.8 Flash Next | 128K | — | — | 1.31 ms | 0.53 ms | 0.49 ms |
| Qwen3.8 Flash Next | 256K | — | — | 1.31 ms | 0.52 ms | 0.80 ms |
| Qwen3.8 Flash Next | 512K | — | — | 1.31 ms | 0.52 ms | 1.27 ms |
| Qwen3.8 Flash Next | 1M | — | — | 1.31 ms | 0.52 ms | 2.25 ms |
| Qwen3.8 27B | 4K | 0.69 ms | — | 1.75 ms | — | — |
| Qwen3.8 27B | 8K | 1.28 ms | — | 1.75 ms | — | — |
| Qwen3.8 27B | 16K | 2.47 ms | — | 1.75 ms | — | — |
| Qwen3.8 27B | 32K | 4.81 ms | — | 1.75 ms | — | — |
| Qwen3.8 27B | 64K | 9.52 ms | — | 1.75 ms | — | — |
| Qwen3.8 27B | 128K | 18.9 ms | — | 1.75 ms | — | — |
| Qwen3.8 27B | 256K | 37.7 ms | — | 1.75 ms | — | — |
| Qwen3.8 27B | 512K | 75.3 ms | — | 1.75 ms | — | — |
| Qwen3.8 27B | 1M | 151 ms | — | 1.75 ms | — | — |

*Total per model by context length (Batch 32, With other ops)*

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

*Breakdown by kind (Batch 32, With other ops)*

| Model | Context | Global attention | Window attention | Linear attention | Sparse attention | Indexer | QKV and output projections | Norms, RoPE and other ops |
|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 4K | 2.86 ms | — | — | — | — | 2.70 ms | 0.26 ms |
| Qwen3-Coder-480B-A35B | 8K | 5.23 ms | — | — | — | — | 2.70 ms | 0.26 ms |
| Qwen3-Coder-480B-A35B | 16K | 9.96 ms | — | — | — | — | 2.70 ms | 0.26 ms |
| Qwen3-Coder-480B-A35B | 32K | 19.5 ms | — | — | — | — | 2.70 ms | 0.26 ms |
| Qwen3-Coder-480B-A35B | 64K | 38.1 ms | — | — | — | — | 2.70 ms | 0.26 ms |
| Qwen3-Coder-480B-A35B | 128K | 75.9 ms | — | — | — | — | 2.70 ms | 0.26 ms |
| Qwen3-Coder-480B-A35B | 256K | 152 ms | — | — | — | — | 2.70 ms | 0.26 ms |
| MiMo V2.6 Flash | 4K | 0.40 ms | 0.59 ms | — | — | — | 1.57 ms | 0.24 ms |
| MiMo V2.6 Flash | 8K | 0.72 ms | 0.59 ms | — | — | — | 1.57 ms | 0.24 ms |
| MiMo V2.6 Flash | 16K | 1.49 ms | 0.59 ms | — | — | — | 1.57 ms | 0.24 ms |
| MiMo V2.6 Flash | 32K | 3.01 ms | 0.59 ms | — | — | — | 1.57 ms | 0.24 ms |
| MiMo V2.6 Flash | 64K | 6.05 ms | 0.59 ms | — | — | — | 1.57 ms | 0.24 ms |
| MiMo V2.6 Flash | 128K | 12.1 ms | 0.59 ms | — | — | — | 1.57 ms | 0.24 ms |
| MiMo V2.6 Flash | 256K | 24.2 ms | 0.59 ms | — | — | — | 1.57 ms | 0.24 ms |
| MiMo V2.6 Flash | 512K | 48.2 ms | 0.59 ms | — | — | — | 1.57 ms | 0.24 ms |
| MiMo V2.6 Flash | 1M | 93.9 ms | 0.59 ms | — | — | — | 1.57 ms | 0.24 ms |
| Inkling | 4K | 1.54 ms | 3.50 ms | — | — | — | 3.16 ms | 0.57 ms |
| Inkling | 8K | 2.74 ms | 3.50 ms | — | — | — | 3.16 ms | 0.57 ms |
| Inkling | 16K | 5.33 ms | 3.50 ms | — | — | — | 3.16 ms | 0.57 ms |
| Inkling | 32K | 10.5 ms | 3.50 ms | — | — | — | 3.16 ms | 0.57 ms |
| Inkling | 64K | 19.2 ms | 4.43 ms | — | — | — | 3.16 ms | 0.57 ms |
| Inkling | 128K | 37.9 ms | 4.43 ms | — | — | — | 3.16 ms | 0.57 ms |
| Inkling | 256K | 75.4 ms | 4.43 ms | — | — | — | 3.16 ms | 0.57 ms |
| Inkling | 512K | 151 ms | 4.43 ms | — | — | — | 3.16 ms | 0.57 ms |
| Inkling | 1M | 299 ms | 4.43 ms | — | — | — | 3.16 ms | 0.57 ms |
| GLM-5.3 | 4K | — | — | — | 1.10 ms | 0.21 ms | 5.48 ms | 1.51 ms |
| GLM-5.3 | 8K | — | — | — | 1.09 ms | 0.29 ms | 5.48 ms | 1.51 ms |
| GLM-5.3 | 16K | — | — | — | 1.12 ms | 0.57 ms | 5.48 ms | 1.51 ms |
| GLM-5.3 | 32K | — | — | — | 1.11 ms | 0.86 ms | 5.48 ms | 1.51 ms |
| GLM-5.3 | 64K | — | — | — | 1.10 ms | 1.40 ms | 5.48 ms | 1.51 ms |
| GLM-5.3 | 128K | — | — | — | 1.09 ms | 2.38 ms | 5.48 ms | 1.51 ms |
| GLM-5.3 | 256K | — | — | — | 1.11 ms | 4.28 ms | 5.48 ms | 1.51 ms |
| GLM-5.3 | 512K | — | — | — | 1.12 ms | 8.07 ms | 5.48 ms | 1.51 ms |
| GLM-5.3 | 1M | — | — | — | 1.11 ms | 15.6 ms | 5.48 ms | 1.51 ms |
| MiniMax-M3.1 Flash | 4K | — | — | — | 1.26 ms | 0.89 ms | 1.83 ms | 0.35 ms |
| MiniMax-M3.1 Flash | 8K | — | — | — | 1.24 ms | 1.03 ms | 1.83 ms | 0.34 ms |
| MiniMax-M3.1 Flash | 16K | — | — | — | 1.24 ms | 1.30 ms | 1.83 ms | 0.35 ms |
| MiniMax-M3.1 Flash | 32K | — | — | — | 1.24 ms | 1.97 ms | 1.83 ms | 0.35 ms |
| MiniMax-M3.1 Flash | 64K | — | — | — | 1.24 ms | 2.92 ms | 1.83 ms | 0.34 ms |
| MiniMax-M3.1 Flash | 128K | — | — | — | 1.24 ms | 5.20 ms | 1.83 ms | 0.35 ms |
| MiniMax-M3.1 Flash | 256K | — | — | — | 1.26 ms | 9.77 ms | 1.83 ms | 0.35 ms |
| MiniMax-M3.1 Flash | 512K | — | — | — | 1.28 ms | 19.0 ms | 1.83 ms | 0.35 ms |
| MiniMax-M3.1 Flash | 1M | — | — | — | 1.28 ms | 37.8 ms | 1.83 ms | 0.35 ms |
| DeepSeek V4 Pro | 4K | 0.44 ms | — | — | 0.94 ms | — | 4.52 ms | 0.68 ms |
| DeepSeek V4 Pro | 8K | 0.44 ms | — | — | 0.94 ms | — | 4.52 ms | 0.68 ms |
| DeepSeek V4 Pro | 16K | 0.47 ms | — | — | 0.94 ms | — | 4.52 ms | 0.68 ms |
| DeepSeek V4 Pro | 32K | 0.67 ms | — | — | 0.94 ms | 0.53 ms | 4.15 ms | 0.64 ms |
| DeepSeek V4 Pro | 64K | 0.76 ms | — | — | 0.94 ms | 0.76 ms | 4.15 ms | 0.64 ms |
| DeepSeek V4 Pro | 128K | 0.96 ms | — | — | 0.94 ms | 1.19 ms | 4.15 ms | 0.64 ms |
| DeepSeek V4 Pro | 256K | 1.41 ms | — | — | 0.94 ms | 1.89 ms | 4.15 ms | 0.64 ms |
| DeepSeek V4 Pro | 512K | 2.23 ms | — | — | 0.94 ms | 3.27 ms | 4.15 ms | 0.64 ms |
| DeepSeek V4 Pro | 1M | 3.87 ms | — | — | 0.94 ms | 6.18 ms | 4.15 ms | 0.64 ms |
| DeepSeek V4.1 Flash | 4K | — | 0.02 ms | — | 0.98 ms | 0.08 ms | 1.72 ms | 0.37 ms |
| DeepSeek V4.1 Flash | 8K | — | 0.02 ms | — | 0.98 ms | 0.10 ms | 1.72 ms | 0.37 ms |
| DeepSeek V4.1 Flash | 16K | — | 0.02 ms | — | 0.98 ms | 0.16 ms | 1.72 ms | 0.37 ms |
| DeepSeek V4.1 Flash | 32K | — | 0.02 ms | — | 0.98 ms | 0.26 ms | 1.72 ms | 0.37 ms |
| DeepSeek V4.1 Flash | 64K | — | 0.02 ms | — | 0.98 ms | 0.42 ms | 1.72 ms | 0.38 ms |
| DeepSeek V4.1 Flash | 128K | — | 0.02 ms | — | 0.98 ms | 0.72 ms | 1.72 ms | 0.38 ms |
| DeepSeek V4.1 Flash | 256K | — | 0.02 ms | — | 0.98 ms | 1.33 ms | 1.72 ms | 0.40 ms |
| DeepSeek V4.1 Flash | 512K | — | 0.02 ms | — | 0.98 ms | 2.51 ms | 1.72 ms | 0.43 ms |
| DeepSeek V4.1 Flash | 1M | — | 0.02 ms | — | 0.98 ms | 4.85 ms | 1.72 ms | 0.51 ms |
| Kimi K3 | 4K | 0.67 ms | — | 4.86 ms | — | — | 12.9 ms | 0.31 ms |
| Kimi K3 | 8K | 0.97 ms | — | 4.86 ms | — | — | 12.9 ms | 0.31 ms |
| Kimi K3 | 16K | 1.86 ms | — | 4.86 ms | — | — | 12.9 ms | 0.31 ms |
| Kimi K3 | 32K | 3.62 ms | — | 4.86 ms | — | — | 12.9 ms | 0.31 ms |
| Kimi K3 | 64K | 7.42 ms | — | 4.86 ms | — | — | 12.9 ms | 0.31 ms |
| Kimi K3 | 128K | 13.1 ms | — | 4.86 ms | — | — | 12.9 ms | 0.31 ms |
| Kimi K3 | 256K | 30.8 ms | — | 4.86 ms | — | — | 12.9 ms | 0.31 ms |
| Kimi K3 | 512K | 61.3 ms | — | 4.86 ms | — | — | 12.9 ms | 0.31 ms |
| Kimi K3 | 1M | 123 ms | — | 4.86 ms | — | — | 12.9 ms | 0.31 ms |
| Kimi K3.1 Pro | 4K | — | — | 4.86 ms | 1.79 ms | 0.24 ms | 13.3 ms | 0.58 ms |
| Kimi K3.1 Pro | 8K | — | — | 4.86 ms | 1.80 ms | 0.34 ms | 13.3 ms | 0.58 ms |
| Kimi K3.1 Pro | 16K | — | — | 4.86 ms | 1.79 ms | 0.65 ms | 13.3 ms | 0.58 ms |
| Kimi K3.1 Pro | 32K | — | — | 4.86 ms | 1.80 ms | 0.98 ms | 13.3 ms | 0.58 ms |
| Kimi K3.1 Pro | 64K | — | — | 4.86 ms | 1.79 ms | 1.60 ms | 13.3 ms | 0.58 ms |
| Kimi K3.1 Pro | 128K | — | — | 4.86 ms | 1.80 ms | 2.72 ms | 13.3 ms | 0.58 ms |
| Kimi K3.1 Pro | 256K | — | — | 4.86 ms | 1.79 ms | 4.89 ms | 13.3 ms | 0.58 ms |
| Kimi K3.1 Pro | 512K | — | — | 4.86 ms | 1.80 ms | 9.23 ms | 13.3 ms | 0.58 ms |
| Kimi K3.1 Pro | 1M | — | — | 4.86 ms | 1.79 ms | 17.9 ms | 13.3 ms | 0.58 ms |
| GLM-5.3-Flash | 4K | — | — | 2.12 ms | 0.25 ms | 0.09 ms | 2.75 ms | 0.80 ms |
| GLM-5.3-Flash | 8K | — | — | 2.12 ms | 0.24 ms | 0.10 ms | 2.75 ms | 0.80 ms |
| GLM-5.3-Flash | 16K | — | — | 2.12 ms | 0.25 ms | 0.11 ms | 2.75 ms | 0.80 ms |
| GLM-5.3-Flash | 32K | — | — | 2.12 ms | 0.24 ms | 0.15 ms | 2.75 ms | 0.80 ms |
| GLM-5.3-Flash | 64K | — | — | 2.12 ms | 0.24 ms | 0.27 ms | 2.75 ms | 0.80 ms |
| GLM-5.3-Flash | 128K | — | — | 2.12 ms | 0.26 ms | 0.42 ms | 2.75 ms | 0.80 ms |
| GLM-5.3-Flash | 256K | — | — | 2.12 ms | 0.25 ms | 0.68 ms | 2.75 ms | 0.80 ms |
| GLM-5.3-Flash | 512K | — | — | 2.12 ms | 0.26 ms | 1.18 ms | 2.75 ms | 0.80 ms |
| GLM-5.3-Flash | 1M | — | — | 2.12 ms | 0.26 ms | 2.22 ms | 2.75 ms | 0.80 ms |
| Qwen3.8 Flash Next | 4K | — | — | 1.31 ms | 0.53 ms | 0.22 ms | 1.49 ms | 0.97 ms |
| Qwen3.8 Flash Next | 8K | — | — | 1.31 ms | 0.53 ms | 0.22 ms | 1.49 ms | 0.97 ms |
| Qwen3.8 Flash Next | 16K | — | — | 1.31 ms | 0.53 ms | 0.23 ms | 1.49 ms | 0.97 ms |
| Qwen3.8 Flash Next | 32K | — | — | 1.31 ms | 0.52 ms | 0.27 ms | 1.49 ms | 0.97 ms |
| Qwen3.8 Flash Next | 64K | — | — | 1.31 ms | 0.53 ms | 0.33 ms | 1.49 ms | 0.97 ms |
| Qwen3.8 Flash Next | 128K | — | — | 1.31 ms | 0.53 ms | 0.49 ms | 1.49 ms | 0.97 ms |
| Qwen3.8 Flash Next | 256K | — | — | 1.31 ms | 0.52 ms | 0.80 ms | 1.49 ms | 0.97 ms |
| Qwen3.8 Flash Next | 512K | — | — | 1.31 ms | 0.52 ms | 1.27 ms | 1.49 ms | 0.97 ms |
| Qwen3.8 Flash Next | 1M | — | — | 1.31 ms | 0.52 ms | 2.25 ms | 1.49 ms | 0.97 ms |
| Qwen3.8 27B | 4K | 0.69 ms | — | 1.75 ms | — | — | 1.72 ms | 1.11 ms |
| Qwen3.8 27B | 8K | 1.28 ms | — | 1.75 ms | — | — | 1.72 ms | 1.11 ms |
| Qwen3.8 27B | 16K | 2.47 ms | — | 1.75 ms | — | — | 1.72 ms | 1.11 ms |
| Qwen3.8 27B | 32K | 4.81 ms | — | 1.75 ms | — | — | 1.72 ms | 1.11 ms |
| Qwen3.8 27B | 64K | 9.52 ms | — | 1.75 ms | — | — | 1.72 ms | 1.11 ms |
| Qwen3.8 27B | 128K | 18.9 ms | — | 1.75 ms | — | — | 1.72 ms | 1.11 ms |
| Qwen3.8 27B | 256K | 37.7 ms | — | 1.75 ms | — | — | 1.72 ms | 1.11 ms |
| Qwen3.8 27B | 512K | 75.3 ms | — | 1.75 ms | — | — | 1.72 ms | 1.11 ms |
| Qwen3.8 27B | 1M | 151 ms | — | 1.75 ms | — | — | 1.72 ms | 1.11 ms |

*Backend per model*

| Model | Backend |
|---|---|
| Qwen3-Coder-480B-A35B | FlashInfer trtllm-gen paged prefill/decode, FP8 Q/KV |
| MiMo V2.6 Flash | vLLM FlashAttention 4, FP8 KV, with flash-attention PR #2918 (unpatched FA4 FP8 split-KV decode is wrong) |
| Inkling | per cell the faster of vLLM Inkling fork tml_fa4 with a BF16 cache and SGLang FA4 with MXFP8 Q/K/V (block-scaled QK, in-kernel V dequant), both with the sheared relative bias; Triton fused_qkvr_prep (short conv + q/k norm + cache write) |
| GLM-5.3 | FlashInfer trtllm-gen FP8 sparse MLA, DeepGEMM FP8 indexer, vLLM top-k (cooperative <= 64 rows) |
| MiniMax-M3.1 Flash | vLLM MiniMax-M3 MSA with FP8 K/V and index keys; decode indexer: CuTe DSL index scoring + FlashInfer top-k; decode attention: Triton sparse GQA below 16 requests, CUTLASS fmha_sm100 from 16; prefill indexer: fmha_sm100 score-only (score + block max pool) + FlashInfer top-k; prefill attention: CuTe DSL KV-outer sparse attention + combine |
| DeepSeek V4 Pro | vLLM FlashMLA sparse MLA on the fp8_ds_mla cache, all 128 heads: BF16 prefill (flash_mla_sparse_fwd), FP8 decode under CUDA graphs (Sm100 Head128 split-KV); CSA (C4) layers attend the 128-token window + top-1024 compressed entries, HCA (C128) layers the window + every compressed entry, one call per layer; DeepGEMM FP8 indexer logits, vLLM top-k (cooperative <= 64 rows, persistent above) |
| DeepSeek V4.1 Flash | FlashMLA (compressed sparse MLA; CuTe DSL swap-AB decode for steps of <= 16 requests), DeepGEMM FP8 indexer, vLLM top-k; prefill with decoder bounded replay, as vLLM's V4.1 fork runs it by default |
| Kimi K3 | FlashInfer cute-dsl MLA decode (FP8 q/KV, all 96 heads, no padding); TRT-LLM-gen ragged prefill (vLLM TRTLLM_RAGGED, FP8 Q/K/V); FlashKDA prefill and fused CUDA KDA decode as Kimi K3.1 Pro |
| Kimi K3.1 Pro | vLLM FlashMLA sparse MLA on the fp8_ds_mla cache: FP8 decode (BF16 q padded to 128 heads); prefill upconverts the context to BF16 (BF16 sparse attention at 128 heads); DeepGEMM FP8 indexer, vLLM top-k (cooperative <= 64 rows); FlashKDA prefill, vLLM fused CUDA KDA decode (fused_kda_decode) |
| GLM-5.3-Flash | FlashInfer trtllm-gen FP8 sparse MLA, DeepGEMM FP8 indexer, vLLM top-k (cooperative <= 64 rows); FlashKDA prefill, FLA Triton KDA decode |
| Qwen3.8 Flash Next | vLLM Triton sparse GQA, vLLM top-k (cooperative <= 64 rows); FlashInfer / Triton GDN |
| Qwen3.8 27B | FlashInfer trtllm-gen paged prefill/decode, FP8 Q/KV (FA4 has no FP8-KV path at head dim 256); FlashInfer CUTLASS GDN chunked prefill, vLLM Triton GDN decode |

*Takeaways the figure shows as the slider moves*

- 4K–16K, batch 32: At short contexts, fixed work decides. Kimi K3 and Kimi K3.1 Pro read and write their large linear-attention states at every step, so up to 8K tokens they are slower than the baseline. Inkling is slower too, because its 55 window layers take about 3.5 ms per step at any context.
- 32K–128K, batch 32: The baseline's time grows with the bytes it reads, to 75.9 ms at 128K tokens. DeepSeek V4.1 Flash takes 1.7 ms there, 44× less. Inkling takes 42.3 ms: it reads less than Qwen3.8 27B but takes twice as long.
- 256K–1M, batch 32: At long contexts, the indexer takes most of the time in GLM-5.3 and MiniMax-M3.1 Flash, and the models with full-attention layers slow down the most. At 1M tokens, Qwen3.8 Flash Next takes 4.1 ms per step and Inkling 303 ms.
- 4K–32K, batch 1: With one request, a step reads too little to keep the GPU busy. Up to 32K tokens, every model takes 0.3 to 2.0 ms in attention kernels.
- 64K–256K, batch 1: The baseline's time grows with the context, to 5.4 ms at 256K tokens. The other new models stay at or below 2.3 ms, but Inkling reaches 3.5 ms, twice the time of Qwen3.8 27B, which reads more bytes per step.
- 512K–1M, batch 1: Even at 1M tokens, GLM-5.3-Flash and Qwen3.8 Flash Next take less than 0.7 ms per step.
