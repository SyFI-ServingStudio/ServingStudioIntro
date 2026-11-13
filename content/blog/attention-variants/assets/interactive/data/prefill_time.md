# Measured attention time to prefill one prompt

All the data behind this figure in "Beyond GQA: Characterizing Modern Attention Variants with ServingStudio".

ms to prefill the prompt in 8K chunks, whole model · B200, DP attention (all heads on each GPU). Hover a bar for the backend.

*Total per model by prompt length (Attention only)*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 31.5 ms | 74.6 ms | 271 ms | 1,021 ms | 3,955 ms | 15,595 ms | 62,179 ms | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 5.37 ms | 13.8 ms | 41.6 ms | 137 ms | 486 ms | 1,830 ms | 7,046 ms | 27,476 ms | 109,744 ms |
| Inkling | 22.8 ms | 49.4 ms | 120 ms | 318 ms | 955 ms | 3,213 ms | 11,623 ms | 43,891 ms | 169,949 ms |
| GLM-5.3 | 33.7 ms | 119 ms | 248 ms | 536 ms | 1,239 ms | 3,086 ms | 8,415 ms | 25,539 ms | 86,761 ms |
| MiniMax-M3.1 Flash | 31.8 ms | 69.4 ms | 151 ms | 321 ms | 698 ms | 1,584 ms | 3,947 ms | 11,167 ms | 35,460 ms |
| DeepSeek V4 Pro | 33.9 ms | 87.2 ms | 176 ms | 358 ms | 850 ms | 2,299 ms | 6,961 ms | 23,829 ms | 88,635 ms |
| DeepSeek V4.1 Flash | 7.32 ms | 15.0 ms | 30.5 ms | 63.9 ms | 141 ms | 339 ms | 904 ms | 2,816 ms | 10,207 ms |
| Kimi K3 | 60.4 ms | 133 ms | 329 ms | 905 ms | 2,788 ms | 9,515 ms | 34,787 ms | 132,600 ms | 517,306 ms |
| Kimi K3.1 Pro | 85.7 ms | 230 ms | 470 ms | 984 ms | 2,149 ms | 4,975 ms | 12,495 ms | 34,927 ms | 110,615 ms |
| GLM-5.3-Flash | 27.2 ms | 56.6 ms | 115 ms | 233 ms | 482 ms | 1,033 ms | 2,364 ms | 5,955 ms | 17,696 ms |
| Qwen3.8 Flash Next | 27.9 ms | 60.6 ms | 129 ms | 266 ms | 551 ms | 1,160 ms | 2,571 ms | 5,881 ms | 13,885 ms |
| Qwen3.8 27B | 8.99 ms | 20.1 ms | 53.8 ms | 161 ms | 533 ms | 1,892 ms | 7,306 ms | 29,010 ms | 114,343 ms |

*Breakdown by kind (Attention only)*

| Model | Prompt | Global attention | Window attention | Linear attention | Sparse attention | Indexer |
|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 4K | 31.5 ms | — | — | — | — |
| Qwen3-Coder-480B-A35B | 8K | 74.6 ms | — | — | — | — |
| Qwen3-Coder-480B-A35B | 16K | 271 ms | — | — | — | — |
| Qwen3-Coder-480B-A35B | 32K | 1,021 ms | — | — | — | — |
| Qwen3-Coder-480B-A35B | 64K | 3,955 ms | — | — | — | — |
| Qwen3-Coder-480B-A35B | 128K | 15,595 ms | — | — | — | — |
| Qwen3-Coder-480B-A35B | 256K | 62,179 ms | — | — | — | — |
| MiMo V2.6 Flash | 4K | 2.16 ms | 3.21 ms | — | — | — |
| MiMo V2.6 Flash | 8K | 7.79 ms | 5.97 ms | — | — | — |
| MiMo V2.6 Flash | 16K | 29.6 ms | 12.0 ms | — | — | — |
| MiMo V2.6 Flash | 32K | 113 ms | 24.0 ms | — | — | — |
| MiMo V2.6 Flash | 64K | 438 ms | 47.9 ms | — | — | — |
| MiMo V2.6 Flash | 128K | 1,734 ms | 95.9 ms | — | — | — |
| MiMo V2.6 Flash | 256K | 6,854 ms | 192 ms | — | — | — |
| MiMo V2.6 Flash | 512K | 27,092 ms | 384 ms | — | — | — |
| MiMo V2.6 Flash | 1M | 108,976 ms | 767 ms | — | — | — |
| Inkling | 4K | 6.32 ms | 16.4 ms | — | — | — |
| Inkling | 8K | 17.9 ms | 31.6 ms | — | — | — |
| Inkling | 16K | 56.0 ms | 63.8 ms | — | — | — |
| Inkling | 32K | 190 ms | 128 ms | — | — | — |
| Inkling | 64K | 697 ms | 258 ms | — | — | — |
| Inkling | 128K | 2,697 ms | 516 ms | — | — | — |
| Inkling | 256K | 10,591 ms | 1,033 ms | — | — | — |
| Inkling | 512K | 41,825 ms | 2,066 ms | — | — | — |
| Inkling | 1M | 165,816 ms | 4,133 ms | — | — | — |
| GLM-5.3 | 4K | — | — | — | 32.0 ms | 1.71 ms |
| GLM-5.3 | 8K | — | — | — | 113 ms | 5.60 ms |
| GLM-5.3 | 16K | — | — | — | 228 ms | 19.3 ms |
| GLM-5.3 | 32K | — | — | — | 463 ms | 72.8 ms |
| GLM-5.3 | 64K | — | — | — | 953 ms | 286 ms |
| GLM-5.3 | 128K | — | — | — | 1,951 ms | 1,135 ms |
| GLM-5.3 | 256K | — | — | — | 3,948 ms | 4,466 ms |
| GLM-5.3 | 512K | — | — | — | 7,941 ms | 17,598 ms |
| GLM-5.3 | 1M | — | — | — | 15,921 ms | 70,840 ms |
| MiniMax-M3.1 Flash | 4K | — | — | — | 24.3 ms | 7.48 ms |
| MiniMax-M3.1 Flash | 8K | — | — | — | 51.6 ms | 17.8 ms |
| MiniMax-M3.1 Flash | 16K | — | — | — | 109 ms | 41.4 ms |
| MiniMax-M3.1 Flash | 32K | — | — | — | 227 ms | 94.4 ms |
| MiniMax-M3.1 Flash | 64K | — | — | — | 473 ms | 225 ms |
| MiniMax-M3.1 Flash | 128K | — | — | — | 991 ms | 593 ms |
| MiniMax-M3.1 Flash | 256K | — | — | — | 2,144 ms | 1,803 ms |
| MiniMax-M3.1 Flash | 512K | — | — | — | 5,007 ms | 6,160 ms |
| MiniMax-M3.1 Flash | 1M | — | — | — | 12,843 ms | 22,617 ms |
| DeepSeek V4 Pro | 4K | 14.3 ms | — | — | 19.6 ms | — |
| DeepSeek V4 Pro | 8K | 29.0 ms | — | — | 58.2 ms | — |
| DeepSeek V4 Pro | 16K | 59.3 ms | — | — | 117 ms | — |
| DeepSeek V4 Pro | 32K | 124 ms | — | — | 235 ms | — |
| DeepSeek V4 Pro | 64K | 276 ms | — | — | 473 ms | 101 ms |
| DeepSeek V4 Pro | 128K | 705 ms | — | — | 955 ms | 639 ms |
| DeepSeek V4 Pro | 256K | 2,088 ms | — | — | 1,933 ms | 2,940 ms |
| DeepSeek V4 Pro | 512K | 6,909 ms | — | — | 3,896 ms | 13,024 ms |
| DeepSeek V4 Pro | 1M | 24,701 ms | — | — | 7,758 ms | 56,176 ms |
| DeepSeek V4.1 Flash | 4K | — | 0.39 ms | — | 6.72 ms | 0.22 ms |
| DeepSeek V4.1 Flash | 8K | — | 0.76 ms | — | 13.2 ms | 0.97 ms |
| DeepSeek V4.1 Flash | 16K | — | 1.51 ms | — | 27.0 ms | 2.04 ms |
| DeepSeek V4.1 Flash | 32K | — | 3.03 ms | — | 54.7 ms | 6.22 ms |
| DeepSeek V4.1 Flash | 64K | — | 6.06 ms | — | 111 ms | 23.7 ms |
| DeepSeek V4.1 Flash | 128K | — | 12.1 ms | — | 231 ms | 96.2 ms |
| DeepSeek V4.1 Flash | 256K | — | 24.2 ms | — | 491 ms | 389 ms |
| DeepSeek V4.1 Flash | 512K | — | 48.5 ms | — | 1,090 ms | 1,677 ms |
| DeepSeek V4.1 Flash | 1M | — | 96.9 ms | — | 2,523 ms | 7,587 ms |
| Kimi K3 | 4K | 9.47 ms | — | 50.9 ms | — | — |
| Kimi K3 | 8K | 33.2 ms | — | 99.5 ms | — | — |
| Kimi K3 | 16K | 130 ms | — | 199 ms | — | — |
| Kimi K3 | 32K | 507 ms | — | 398 ms | — | — |
| Kimi K3 | 64K | 1,992 ms | — | 796 ms | — | — |
| Kimi K3 | 128K | 7,923 ms | — | 1,592 ms | — | — |
| Kimi K3 | 256K | 31,603 ms | — | 3,184 ms | — | — |
| Kimi K3 | 512K | 126,233 ms | — | 6,367 ms | — | — |
| Kimi K3 | 1M | 504,572 ms | — | 12,734 ms | — | — |
| Kimi K3.1 Pro | 4K | — | — | 50.9 ms | 32.9 ms | 1.95 ms |
| Kimi K3.1 Pro | 8K | — | — | 99.5 ms | 124 ms | 6.40 ms |
| Kimi K3.1 Pro | 16K | — | — | 199 ms | 249 ms | 22.1 ms |
| Kimi K3.1 Pro | 32K | — | — | 398 ms | 502 ms | 83.2 ms |
| Kimi K3.1 Pro | 64K | — | — | 796 ms | 1,025 ms | 327 ms |
| Kimi K3.1 Pro | 128K | — | — | 1,592 ms | 2,086 ms | 1,297 ms |
| Kimi K3.1 Pro | 256K | — | — | 3,184 ms | 4,207 ms | 5,104 ms |
| Kimi K3.1 Pro | 512K | — | — | 6,367 ms | 8,448 ms | 20,112 ms |
| Kimi K3.1 Pro | 1M | — | — | 12,734 ms | 16,921 ms | 80,960 ms |
| GLM-5.3-Flash | 4K | — | — | 18.3 ms | 7.88 ms | 1.00 ms |
| GLM-5.3-Flash | 8K | — | — | 35.7 ms | 17.8 ms | 3.04 ms |
| GLM-5.3-Flash | 16K | — | — | 71.4 ms | 37.1 ms | 6.08 ms |
| GLM-5.3-Flash | 32K | — | — | 143 ms | 75.8 ms | 14.3 ms |
| GLM-5.3-Flash | 64K | — | — | 286 ms | 153 ms | 43.6 ms |
| GLM-5.3-Flash | 128K | — | — | 571 ms | 308 ms | 154 ms |
| GLM-5.3-Flash | 256K | — | — | 1,143 ms | 617 ms | 604 ms |
| GLM-5.3-Flash | 512K | — | — | 2,286 ms | 1,235 ms | 2,435 ms |
| GLM-5.3-Flash | 1M | — | — | 4,571 ms | 2,471 ms | 10,653 ms |
| Qwen3.8 Flash Next | 4K | — | — | 4.84 ms | 22.2 ms | 0.87 ms |
| Qwen3.8 Flash Next | 8K | — | — | 8.93 ms | 49.8 ms | 1.91 ms |
| Qwen3.8 Flash Next | 16K | — | — | 17.9 ms | 106 ms | 4.73 ms |
| Qwen3.8 Flash Next | 32K | — | — | 35.7 ms | 218 ms | 12.4 ms |
| Qwen3.8 Flash Next | 64K | — | — | 71.5 ms | 444 ms | 35.7 ms |
| Qwen3.8 Flash Next | 128K | — | — | 143 ms | 903 ms | 114 ms |
| Qwen3.8 Flash Next | 256K | — | — | 286 ms | 1,841 ms | 445 ms |
| Qwen3.8 Flash Next | 512K | — | — | 572 ms | 3,738 ms | 1,571 ms |
| Qwen3.8 Flash Next | 1M | — | — | 1,144 ms | 7,533 ms | 5,208 ms |
| Qwen3.8 27B | 4K | 2.54 ms | — | 6.45 ms | — | — |
| Qwen3.8 27B | 8K | 8.17 ms | — | 11.9 ms | — | — |
| Qwen3.8 27B | 16K | 30.0 ms | — | 23.8 ms | — | — |
| Qwen3.8 27B | 32K | 114 ms | — | 47.6 ms | — | — |
| Qwen3.8 27B | 64K | 438 ms | — | 95.3 ms | — | — |
| Qwen3.8 27B | 128K | 1,701 ms | — | 191 ms | — | — |
| Qwen3.8 27B | 256K | 6,925 ms | — | 381 ms | — | — |
| Qwen3.8 27B | 512K | 28,248 ms | — | 762 ms | — | — |
| Qwen3.8 27B | 1M | 112,819 ms | — | 1,525 ms | — | — |

*Total per model by prompt length (With other ops)*

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 70.1 ms | 151 ms | 425 ms | 1,329 ms | 4,570 ms | 16,826 ms | 64,641 ms | not supported beyond 256K | not supported beyond 256K |
| MiMo V2.6 Flash | 29.6 ms | 62.2 ms | 138 ms | 330 ms | 873 ms | 2,604 ms | 8,594 ms | 30,573 ms | 115,938 ms |
| Inkling | 82.3 ms | 167 ms | 355 ms | 788 ms | 1,893 ms | 5,091 ms | 15,378 ms | 51,402 ms | 184,970 ms |
| GLM-5.3 | 148 ms | 347 ms | 704 ms | 1,448 ms | 3,064 ms | 6,736 ms | 15,716 ms | 40,148 ms | 116,001 ms |
| MiniMax-M3.1 Flash | 55.0 ms | 118 ms | 249 ms | 518 ms | 1,091 ms | 2,371 ms | 5,525 ms | 14,344 ms | 41,901 ms |
| DeepSeek V4 Pro | 126 ms | 271 ms | 544 ms | 1,095 ms | 2,247 ms | 4,966 ms | 12,167 ms | 34,113 ms | 109,076 ms |
| DeepSeek V4.1 Flash | 17.2 ms | 33.4 ms | 67.4 ms | 138 ms | 289 ms | 634 ms | 1,497 ms | 4,010 ms | 12,629 ms |
| Kimi K3 | 296 ms | 596 ms | 1,259 ms | 2,778 ms | 6,578 ms | 17,277 ms | 51,040 ms | 168,021 ms | 599,805 ms |
| Kimi K3.1 Pro | 327 ms | 703 ms | 1,418 ms | 2,880 ms | 5,941 ms | 12,565 ms | 27,690 ms | 65,380 ms | 171,781 ms |
| GLM-5.3-Flash | 86.1 ms | 173 ms | 348 ms | 699 ms | 1,414 ms | 2,896 ms | 6,091 ms | 13,409 ms | 32,604 ms |
| Qwen3.8 Flash Next | 54.6 ms | 113 ms | 233 ms | 474 ms | 967 ms | 1,992 ms | 4,236 ms | 9,210 ms | 20,542 ms |
| Qwen3.8 27B | 43.8 ms | 89.1 ms | 192 ms | 437 ms | 1,085 ms | 2,996 ms | 9,514 ms | 33,426 ms | 123,175 ms |

*Breakdown by kind (With other ops)*

| Model | Prompt | Global attention | Window attention | Linear attention | Sparse attention | Indexer | QKV and output projections | Norms, RoPE and other ops |
|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 4K | 31.5 ms | — | — | — | — | 36.1 ms | 2.47 ms |
| Qwen3-Coder-480B-A35B | 8K | 74.6 ms | — | — | — | — | 72.0 ms | 4.91 ms |
| Qwen3-Coder-480B-A35B | 16K | 271 ms | — | — | — | — | 144 ms | 9.82 ms |
| Qwen3-Coder-480B-A35B | 32K | 1,021 ms | — | — | — | — | 288 ms | 19.6 ms |
| Qwen3-Coder-480B-A35B | 64K | 3,955 ms | — | — | — | — | 576 ms | 39.3 ms |
| Qwen3-Coder-480B-A35B | 128K | 15,595 ms | — | — | — | — | 1,152 ms | 78.6 ms |
| Qwen3-Coder-480B-A35B | 256K | 62,179 ms | — | — | — | — | 2,304 ms | 157 ms |
| MiMo V2.6 Flash | 4K | 2.16 ms | 3.21 ms | — | — | — | 19.5 ms | 4.70 ms |
| MiMo V2.6 Flash | 8K | 7.79 ms | 5.97 ms | — | — | — | 38.9 ms | 9.46 ms |
| MiMo V2.6 Flash | 16K | 29.6 ms | 12.0 ms | — | — | — | 77.9 ms | 18.9 ms |
| MiMo V2.6 Flash | 32K | 113 ms | 24.0 ms | — | — | — | 156 ms | 37.8 ms |
| MiMo V2.6 Flash | 64K | 438 ms | 47.9 ms | — | — | — | 311 ms | 75.7 ms |
| MiMo V2.6 Flash | 128K | 1,734 ms | 95.9 ms | — | — | — | 623 ms | 151 ms |
| MiMo V2.6 Flash | 256K | 6,854 ms | 192 ms | — | — | — | 1,246 ms | 303 ms |
| MiMo V2.6 Flash | 512K | 27,092 ms | 384 ms | — | — | — | 2,492 ms | 605 ms |
| MiMo V2.6 Flash | 1M | 108,976 ms | 767 ms | — | — | — | 4,984 ms | 1,210 ms |
| Inkling | 4K | 6.32 ms | 16.4 ms | — | — | — | 49.8 ms | 9.80 ms |
| Inkling | 8K | 17.9 ms | 31.6 ms | — | — | — | 97.8 ms | 19.6 ms |
| Inkling | 16K | 56.0 ms | 63.8 ms | — | — | — | 196 ms | 39.2 ms |
| Inkling | 32K | 190 ms | 128 ms | — | — | — | 391 ms | 78.4 ms |
| Inkling | 64K | 697 ms | 258 ms | — | — | — | 782 ms | 157 ms |
| Inkling | 128K | 2,697 ms | 516 ms | — | — | — | 1,564 ms | 313 ms |
| Inkling | 256K | 10,591 ms | 1,033 ms | — | — | — | 3,128 ms | 627 ms |
| Inkling | 512K | 41,825 ms | 2,066 ms | — | — | — | 6,257 ms | 1,254 ms |
| Inkling | 1M | 165,816 ms | 4,133 ms | — | — | — | 12,513 ms | 2,508 ms |
| GLM-5.3 | 4K | — | — | — | 32.0 ms | 1.71 ms | 83.3 ms | 30.8 ms |
| GLM-5.3 | 8K | — | — | — | 113 ms | 5.60 ms | 167 ms | 61.0 ms |
| GLM-5.3 | 16K | — | — | — | 228 ms | 19.3 ms | 334 ms | 122 ms |
| GLM-5.3 | 32K | — | — | — | 463 ms | 72.8 ms | 668 ms | 244 ms |
| GLM-5.3 | 64K | — | — | — | 953 ms | 286 ms | 1,337 ms | 488 ms |
| GLM-5.3 | 128K | — | — | — | 1,951 ms | 1,135 ms | 2,673 ms | 977 ms |
| GLM-5.3 | 256K | — | — | — | 3,948 ms | 4,466 ms | 5,346 ms | 1,955 ms |
| GLM-5.3 | 512K | — | — | — | 7,941 ms | 17,598 ms | 10,692 ms | 3,916 ms |
| GLM-5.3 | 1M | — | — | — | 15,921 ms | 70,840 ms | 21,384 ms | 7,855 ms |
| MiniMax-M3.1 Flash | 4K | — | — | — | 24.3 ms | 7.48 ms | 20.3 ms | 2.90 ms |
| MiniMax-M3.1 Flash | 8K | — | — | — | 51.6 ms | 17.8 ms | 44.5 ms | 4.47 ms |
| MiniMax-M3.1 Flash | 16K | — | — | — | 109 ms | 41.4 ms | 89.0 ms | 9.01 ms |
| MiniMax-M3.1 Flash | 32K | — | — | — | 227 ms | 94.4 ms | 178 ms | 18.3 ms |
| MiniMax-M3.1 Flash | 64K | — | — | — | 473 ms | 225 ms | 356 ms | 37.3 ms |
| MiniMax-M3.1 Flash | 128K | — | — | — | 991 ms | 593 ms | 712 ms | 75.5 ms |
| MiniMax-M3.1 Flash | 256K | — | — | — | 2,144 ms | 1,803 ms | 1,423 ms | 155 ms |
| MiniMax-M3.1 Flash | 512K | — | — | — | 5,007 ms | 6,160 ms | 2,847 ms | 331 ms |
| MiniMax-M3.1 Flash | 1M | — | — | — | 12,843 ms | 22,617 ms | 5,694 ms | 748 ms |
| DeepSeek V4 Pro | 4K | 14.3 ms | — | — | 19.6 ms | — | 64.0 ms | 27.6 ms |
| DeepSeek V4 Pro | 8K | 29.0 ms | — | — | 58.2 ms | — | 129 ms | 54.7 ms |
| DeepSeek V4 Pro | 16K | 59.3 ms | — | — | 117 ms | — | 259 ms | 109 ms |
| DeepSeek V4 Pro | 32K | 124 ms | — | — | 235 ms | — | 518 ms | 219 ms |
| DeepSeek V4 Pro | 64K | 276 ms | — | — | 473 ms | 101 ms | 985 ms | 411 ms |
| DeepSeek V4 Pro | 128K | 705 ms | — | — | 955 ms | 639 ms | 1,886 ms | 780 ms |
| DeepSeek V4 Pro | 256K | 2,088 ms | — | — | 1,933 ms | 2,940 ms | 3,689 ms | 1,516 ms |
| DeepSeek V4 Pro | 512K | 6,909 ms | — | — | 3,896 ms | 13,024 ms | 7,294 ms | 2,990 ms |
| DeepSeek V4 Pro | 1M | 24,701 ms | — | — | 7,758 ms | 56,176 ms | 14,505 ms | 5,936 ms |
| DeepSeek V4.1 Flash | 4K | — | 0.39 ms | — | 6.72 ms | 0.22 ms | 9.31 ms | 0.55 ms |
| DeepSeek V4.1 Flash | 8K | — | 0.76 ms | — | 13.2 ms | 0.97 ms | 17.7 ms | 0.78 ms |
| DeepSeek V4.1 Flash | 16K | — | 1.51 ms | — | 27.0 ms | 2.04 ms | 35.3 ms | 1.56 ms |
| DeepSeek V4.1 Flash | 32K | — | 3.03 ms | — | 54.7 ms | 6.22 ms | 70.6 ms | 3.15 ms |
| DeepSeek V4.1 Flash | 64K | — | 6.06 ms | — | 111 ms | 23.7 ms | 141 ms | 6.39 ms |
| DeepSeek V4.1 Flash | 128K | — | 12.1 ms | — | 231 ms | 96.2 ms | 282 ms | 13.2 ms |
| DeepSeek V4.1 Flash | 256K | — | 24.2 ms | — | 491 ms | 389 ms | 565 ms | 28.2 ms |
| DeepSeek V4.1 Flash | 512K | — | 48.5 ms | — | 1,090 ms | 1,677 ms | 1,130 ms | 64.6 ms |
| DeepSeek V4.1 Flash | 1M | — | 96.9 ms | — | 2,523 ms | 7,587 ms | 2,259 ms | 163 ms |
| Kimi K3 | 4K | 9.47 ms | — | 50.9 ms | — | — | 214 ms | 21.2 ms |
| Kimi K3 | 8K | 33.2 ms | — | 99.5 ms | — | — | 421 ms | 41.9 ms |
| Kimi K3 | 16K | 130 ms | — | 199 ms | — | — | 842 ms | 88.5 ms |
| Kimi K3 | 32K | 507 ms | — | 398 ms | — | — | 1,684 ms | 189 ms |
| Kimi K3 | 64K | 1,992 ms | — | 796 ms | — | — | 3,367 ms | 422 ms |
| Kimi K3 | 128K | 7,923 ms | — | 1,592 ms | — | — | 6,735 ms | 1,027 ms |
| Kimi K3 | 256K | 31,603 ms | — | 3,184 ms | — | — | 13,470 ms | 2,784 ms |
| Kimi K3 | 512K | 126,233 ms | — | 6,367 ms | — | — | 26,939 ms | 8,482 ms |
| Kimi K3 | 1M | 504,572 ms | — | 12,734 ms | — | — | 53,878 ms | 28,620 ms |
| Kimi K3.1 Pro | 4K | — | — | 50.9 ms | 32.9 ms | 1.95 ms | 218 ms | 24.0 ms |
| Kimi K3.1 Pro | 8K | — | — | 99.5 ms | 124 ms | 6.40 ms | 427 ms | 47.0 ms |
| Kimi K3.1 Pro | 16K | — | — | 199 ms | 249 ms | 22.1 ms | 854 ms | 94.1 ms |
| Kimi K3.1 Pro | 32K | — | — | 398 ms | 502 ms | 83.2 ms | 1,708 ms | 188 ms |
| Kimi K3.1 Pro | 64K | — | — | 796 ms | 1,025 ms | 327 ms | 3,415 ms | 378 ms |
| Kimi K3.1 Pro | 128K | — | — | 1,592 ms | 2,086 ms | 1,297 ms | 6,830 ms | 759 ms |
| Kimi K3.1 Pro | 256K | — | — | 3,184 ms | 4,207 ms | 5,104 ms | 13,660 ms | 1,534 ms |
| Kimi K3.1 Pro | 512K | — | — | 6,367 ms | 8,448 ms | 20,112 ms | 27,321 ms | 3,131 ms |
| Kimi K3.1 Pro | 1M | — | — | 12,734 ms | 16,921 ms | 80,960 ms | 54,641 ms | 6,524 ms |
| GLM-5.3-Flash | 4K | — | — | 18.3 ms | 7.88 ms | 1.00 ms | 38.0 ms | 20.9 ms |
| GLM-5.3-Flash | 8K | — | — | 35.7 ms | 17.8 ms | 3.04 ms | 75.7 ms | 40.7 ms |
| GLM-5.3-Flash | 16K | — | — | 71.4 ms | 37.1 ms | 6.08 ms | 151 ms | 81.5 ms |
| GLM-5.3-Flash | 32K | — | — | 143 ms | 75.8 ms | 14.3 ms | 303 ms | 163 ms |
| GLM-5.3-Flash | 64K | — | — | 286 ms | 153 ms | 43.6 ms | 606 ms | 326 ms |
| GLM-5.3-Flash | 128K | — | — | 571 ms | 308 ms | 154 ms | 1,212 ms | 652 ms |
| GLM-5.3-Flash | 256K | — | — | 1,143 ms | 617 ms | 604 ms | 2,423 ms | 1,304 ms |
| GLM-5.3-Flash | 512K | — | — | 2,286 ms | 1,235 ms | 2,435 ms | 4,846 ms | 2,608 ms |
| GLM-5.3-Flash | 1M | — | — | 4,571 ms | 2,471 ms | 10,653 ms | 9,693 ms | 5,215 ms |
| Qwen3.8 Flash Next | 4K | — | — | 4.84 ms | 22.2 ms | 0.87 ms | 15.0 ms | 11.7 ms |
| Qwen3.8 Flash Next | 8K | — | — | 8.93 ms | 49.8 ms | 1.91 ms | 30.7 ms | 21.3 ms |
| Qwen3.8 Flash Next | 16K | — | — | 17.9 ms | 106 ms | 4.73 ms | 61.3 ms | 42.7 ms |
| Qwen3.8 Flash Next | 32K | — | — | 35.7 ms | 218 ms | 12.4 ms | 123 ms | 85.4 ms |
| Qwen3.8 Flash Next | 64K | — | — | 71.5 ms | 444 ms | 35.7 ms | 245 ms | 171 ms |
| Qwen3.8 Flash Next | 128K | — | — | 143 ms | 903 ms | 114 ms | 491 ms | 342 ms |
| Qwen3.8 Flash Next | 256K | — | — | 286 ms | 1,841 ms | 445 ms | 981 ms | 683 ms |
| Qwen3.8 Flash Next | 512K | — | — | 572 ms | 3,738 ms | 1,571 ms | 1,962 ms | 1,366 ms |
| Qwen3.8 Flash Next | 1M | — | — | 1,144 ms | 7,533 ms | 5,208 ms | 3,925 ms | 2,733 ms |
| Qwen3.8 27B | 4K | 2.54 ms | — | 6.45 ms | — | — | 20.2 ms | 14.7 ms |
| Qwen3.8 27B | 8K | 8.17 ms | — | 11.9 ms | — | — | 42.0 ms | 27.0 ms |
| Qwen3.8 27B | 16K | 30.0 ms | — | 23.8 ms | — | — | 84.0 ms | 54.0 ms |
| Qwen3.8 27B | 32K | 114 ms | — | 47.6 ms | — | — | 168 ms | 108 ms |
| Qwen3.8 27B | 64K | 438 ms | — | 95.3 ms | — | — | 336 ms | 216 ms |
| Qwen3.8 27B | 128K | 1,701 ms | — | 191 ms | — | — | 672 ms | 432 ms |
| Qwen3.8 27B | 256K | 6,925 ms | — | 381 ms | — | — | 1,344 ms | 864 ms |
| Qwen3.8 27B | 512K | 28,248 ms | — | 762 ms | — | — | 2,688 ms | 1,728 ms |
| Qwen3.8 27B | 1M | 112,819 ms | — | 1,525 ms | — | — | 5,375 ms | 3,456 ms |

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

- 4K–16K: At short prompts, sparse attention saves little. At 8K tokens, the baseline's dense attention is still faster than the attention of GLM-5.3, DeepSeek V4 Pro, Kimi K3 and Kimi K3.1 Pro.
- 32K–128K: The baseline's time grows with the square of the prompt, and from 32K tokens on it is slower than every new model. At 128K tokens, GLM-5.3 needs 1.4× the attention FLOPs of MiMo V2.6 Flash and takes 1.7× as long.
- 256K–1M: At long prompts, the indexer takes a growing share of the time. At 1M tokens, it is most of the time in every sparse model except Qwen3.8 Flash Next, which takes 13.9 s in total against 86.8 s for GLM-5.3.
