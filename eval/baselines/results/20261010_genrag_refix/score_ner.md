| system | set | slice | n | R@10(any) | **cov@10** | cov@20 | MRR@10 | p50(s) | 建库 |
|---|---|---|---|---|---|---|---|---|---|
| genrag | ner | L | 39 | 0.8718 | **0.5432** | 0.5526 | 0.7804 | 79.31 | 536s |
| knc-scoped | ner | L | 39 | 0.9487 | **0.6752** | 0.7645 | 0.772 | 1.838 | 0s |

**配对差（系统 − knc-scoped，ner，bootstrap 95% CI，n=39）**

| system | metric | mean_diff | ci95 | 显著 |
|---|---|---|---|---|
| genrag | recall@10 | -0.0769 | [-0.1795, 0.0256] | 否 |
| genrag | cov@10 | -0.132 | [-0.2073, -0.0551] | 是 |
| genrag | mrr@10 | 0.0083 | [-0.0528, 0.0827] | 否 |
| knc-scoped | recall@10 | 0.0 | [0.0, 0.0] | 否 |
| knc-scoped | cov@10 | 0.0 | [0.0, 0.0] | 否 |
| knc-scoped | mrr@10 | 0.0 | [0.0, 0.0] | 否 |
