| system | set | slice | n | R@10(any) | **cov@10** | cov@20 | MRR@10 | p50(s) | 建库 |
|---|---|---|---|---|---|---|---|---|---|
| chab-all | ner | L | 39 | 0.9231 | **0.6701** | 0.7645 | 0.7695 | 1.541 | 0s |
| chab-noentity | ner | L | 39 | 0.9231 | **0.6701** | 0.7645 | 0.7701 | 0.51 | 0s |

**配对差（系统 − chab-all，ner，bootstrap 95% CI，n=39）**

| system | metric | mean_diff | ci95 | 显著 |
|---|---|---|---|---|
| chab-all | recall@10 | 0.0 | [0.0, 0.0] | 否 |
| chab-all | cov@10 | 0.0 | [0.0, 0.0] | 否 |
| chab-all | mrr@10 | 0.0 | [0.0, 0.0] | 否 |
| chab-noentity | recall@10 | 0.0 | [0.0, 0.0] | 否 |
| chab-noentity | cov@10 | 0.0 | [0.0, 0.0] | 否 |
| chab-noentity | mrr@10 | 0.0006 | [0.0, 0.0018] | 否 |
