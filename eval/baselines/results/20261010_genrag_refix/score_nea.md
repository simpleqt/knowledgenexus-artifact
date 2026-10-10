| system | set | slice | n | R@10(any) | **cov@10** | cov@20 | MRR@10 | p50(s) | 建库 |
|---|---|---|---|---|---|---|---|---|---|
| genrag | nea | L | 48 | 0.8958 | **0.6344** | 0.642 | 0.7353 | 83.12 | 1576s |
| knc-scoped | nea | L | 48 | 0.9375 | **0.5993** | 0.7125 | 0.7346 | 1.457 | 0s |

**配对差（系统 − knc-scoped，nea，bootstrap 95% CI，n=48）**

| system | metric | mean_diff | ci95 | 显著 |
|---|---|---|---|---|
| genrag | recall@10 | -0.0417 | [-0.1458, 0.0625] | 否 |
| genrag | cov@10 | 0.0351 | [-0.0483, 0.1194] | 否 |
| genrag | mrr@10 | 0.0007 | [-0.1076, 0.1042] | 否 |
| knc-scoped | recall@10 | 0.0 | [0.0, 0.0] | 否 |
| knc-scoped | cov@10 | 0.0 | [0.0, 0.0] | 否 |
| knc-scoped | mrr@10 | 0.0 | [0.0, 0.0] | 否 |
