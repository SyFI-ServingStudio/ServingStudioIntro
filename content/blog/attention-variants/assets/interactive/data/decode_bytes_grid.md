# Bytes one request reads per decode step as the context grows

All the data behind this figure in "Beyond GQA: Characterizing Modern Attention Variants with ServingStudio".

*Per model by context length*

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
