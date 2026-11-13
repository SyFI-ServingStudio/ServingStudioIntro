# Attention memory of one request as the context grows

All the data behind this figure in "Beyond GQA: Characterizing Modern Attention Variants with ServingStudio".

*Per model by context length*

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
