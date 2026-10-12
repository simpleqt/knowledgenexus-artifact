# 跨域泛化主表（10×690 冻结重放样本，mean；sd 见 scan_agg.json）

| Domain | Arm | Terms | Entities | C10 | Conc/doc | P_silver | R_silver | F1 | judged-valid | A-B agree |
|---|---|---|---|---|---|---|---|---|---|---|
| Computer science | batch schema (pre) | 95 | 18.4 | 0.899 | 1.144 | 0.0 | 0.0 | 0.0 | 0.565 | 0.903 |
| Computer science | frequency top-K | 251 | 312.4 | 0.227 | 30.677 | 0.101 | 0.097 | 0.099 | 0.293 | 0.775 |
| Computer science | C-value | 251 | 273.4 | 0.457 | 7.15 | 0.3 | 0.043 | 0.075 | 0.267 | 0.763 |
| Computer science | LLM direct | 1198 | 344.5 | 0.501 | 4.634 | 0.663 | 0.132 | 0.22 | 0.787 | 0.883 |
| Computer science | three-source (ours) | 251 | 202.1 | 0.404 | 3.602 | 0.274 | 0.05 | 0.085 | 0.838 | 0.785 |
| Biomedicine | batch schema (pre) | 130 | 100.5 | 0.428 | 2.808 | 0.089 | 0.009 | 0.016 | 0.906 | 0.836 |
| Biomedicine | frequency top-K | 513 | 642.7 | 0.114 | 29.701 | 0.038 | 0.039 | 0.038 | 0.362 | 0.772 |
| Biomedicine | C-value | 513 | 495.4 | 0.249 | 8.108 | 0.073 | 0.022 | 0.034 | 0.35 | 0.883 |
| Biomedicine | LLM direct | 2003 | 591.5 | 0.309 | 4.908 | 0.343 | 0.085 | 0.136 | 0.838 | 0.835 |
| Biomedicine | three-source (ours) | 513 | 408.7 | 0.281 | 5.136 | 0.273 | 0.06 | 0.098 | 0.85 | 0.922 |
| Materials | batch schema (pre) | 152 | 58.5 | 0.829 | 2.327 | 0.054 | 0.004 | 0.007 | 0.688 | 0.831 |
| Materials | frequency top-K | 384 | 481.4 | 0.131 | 31.781 | 0.029 | 0.035 | 0.032 | 0.4 | 0.722 |
| Materials | C-value | 384 | 411.2 | 0.238 | 7.48 | 0.155 | 0.044 | 0.069 | 0.367 | 0.85 |
| Materials | LLM direct | 2045 | 713.8 | 0.198 | 6.919 | 0.336 | 0.093 | 0.146 | 0.925 | 0.937 |
| Materials | three-source (ours) | 384 | 337.4 | 0.268 | 5.375 | 0.223 | 0.046 | 0.076 | 0.912 | 0.899 |
