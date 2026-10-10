| system | set | slice | n | R@10(any) | **cov@10** | cov@20 | MRR@10 | p50(s) | 建库 |
|---|---|---|---|---|---|---|---|---|---|
| kaa-alias-off | nea | L | 48 | 0.9167 | **0.584** | 0.7101 | 0.7861 | 1.629 | 0s |
| kaa-alias-off | ner | L | 39 | 0.9231 | **0.6744** | 0.7615 | 0.7552 | 1.589 | 0s |
| kaa-alias-on | nea | L | 48 | 0.9375 | **0.5993** | 0.7125 | 0.7346 | 3.519 | 0s |
| kaa-alias-on | ner | L | 39 | 0.9487 | **0.6752** | 0.7645 | 0.772 | 3.108 | 0s |

**配对差（系统 − kaa-alias-on，nea，bootstrap 95% CI，n=48）**

| system | metric | mean_diff | ci95 | 显著 |
|---|---|---|---|---|
| kaa-alias-off | recall@10 | -0.0208 | [-0.0625, 0.0] | 否 |
| kaa-alias-off | cov@10 | -0.0153 | [-0.0392, 0.0062] | 否 |
| kaa-alias-off | mrr@10 | 0.0515 | [-0.0256, 0.1267] | 否 |
| kaa-alias-on | recall@10 | 0.0 | [0.0, 0.0] | 否 |
| kaa-alias-on | cov@10 | 0.0 | [0.0, 0.0] | 否 |
| kaa-alias-on | mrr@10 | 0.0 | [0.0, 0.0] | 否 |
