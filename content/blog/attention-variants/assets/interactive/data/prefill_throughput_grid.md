# Achieved compute throughput of prefill attention

All the data behind this figure in "Beyond GQA: Characterizing Modern Attention Variants with ServingStudio".

*Per model by prompt length*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 0.81 PFLOP/s | 1.37 PFLOP/s | 1.51 PFLOP/s | 1.60 PFLOP/s | 1.65 PFLOP/s | 1.68 PFLOP/s | 1.68 PFLOP/s | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 0.73 PFLOP/s | 1.02 PFLOP/s | 1.27 PFLOP/s | 1.50 PFLOP/s | 1.66 PFLOP/s | 1.75 PFLOP/s | 1.81 PFLOP/s | 1.85 PFLOP/s | 1.85 PFLOP/s |
| Inkling | 0.29 PFLOP/s | 0.39 PFLOP/s | 0.53 PFLOP/s | 0.70 PFLOP/s | 0.87 PFLOP/s | 1.00 PFLOP/s | 1.09 PFLOP/s | 1.14 PFLOP/s | 1.17 PFLOP/s |
| GLM-5.3 | 2.07 PFLOP/s | 1.39 PFLOP/s | 1.47 PFLOP/s | 1.49 PFLOP/s | 1.46 PFLOP/s | 1.42 PFLOP/s | 1.39 PFLOP/s | 1.38 PFLOP/s | 1.36 PFLOP/s |
| MiniMax-M3.1 Flash | 0.41 PFLOP/s | 0.45 PFLOP/s | 0.47 PFLOP/s | 0.50 PFLOP/s | 0.56 PFLOP/s | 0.66 PFLOP/s | 0.80 PFLOP/s | 0.94 PFLOP/s | 1.07 PFLOP/s |
| DeepSeek V4 Pro | 0.77 PFLOP/s | 0.83 PFLOP/s | 0.99 PFLOP/s | 1.16 PFLOP/s | 1.23 PFLOP/s | 1.26 PFLOP/s | 1.30 PFLOP/s | 1.30 PFLOP/s | 1.28 PFLOP/s |
| DeepSeek V4.1 Flash | 0.87 PFLOP/s | 0.91 PFLOP/s | 0.95 PFLOP/s | 1.00 PFLOP/s | 1.07 PFLOP/s | 1.15 PFLOP/s | 1.25 PFLOP/s | 1.30 PFLOP/s | 1.27 PFLOP/s |
| Kimi K3 | 0.28 PFLOP/s | 0.44 PFLOP/s | 0.66 PFLOP/s | 0.91 PFLOP/s | 1.16 PFLOP/s | 1.35 PFLOP/s | 1.46 PFLOP/s | 1.53 PFLOP/s | 1.57 PFLOP/s |
| Kimi K3.1 Pro | 0.44 PFLOP/s | 0.39 PFLOP/s | 0.43 PFLOP/s | 0.48 PFLOP/s | 0.54 PFLOP/s | 0.64 PFLOP/s | 0.78 PFLOP/s | 0.94 PFLOP/s | 1.08 PFLOP/s |
| GLM-5.3-Flash | 0.39 PFLOP/s | 0.44 PFLOP/s | 0.47 PFLOP/s | 0.51 PFLOP/s | 0.54 PFLOP/s | 0.61 PFLOP/s | 0.69 PFLOP/s | 0.81 PFLOP/s | 0.90 PFLOP/s |
| Qwen3.8 Flash Next | 0.11 PFLOP/s | 0.11 PFLOP/s | 0.11 PFLOP/s | 0.11 PFLOP/s | 0.12 PFLOP/s | 0.12 PFLOP/s | 0.13 PFLOP/s | 0.15 PFLOP/s | 0.19 PFLOP/s |
| Qwen3.8 27B | 0.54 PFLOP/s | 0.81 PFLOP/s | 1.10 PFLOP/s | 1.39 PFLOP/s | 1.63 PFLOP/s | 1.81 PFLOP/s | 1.86 PFLOP/s | 1.87 PFLOP/s | 1.89 PFLOP/s |

Color scale: 0 PFLOP/s to 4.5 PFLOP/s, the B200's FP8 peak.
