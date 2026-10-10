| system | set | slice | n | R@10(any) | **cov@10** | cov@20 | MRR@10 | p50(s) | 建库 |
|---|---|---|---|---|---|---|---|---|---|
| chab-all | nea | L | 48 | 0.9375 | **0.5993** | 0.7125 | 0.7346 | 1.463 | 0s |
| chab-noentity | nea | L | 48 | 0.9375 | **0.5993** | 0.7021 | 0.7346 | 0.519 | 0s |

**配对差（系统 − chab-all，nea，bootstrap 95% CI，n=48）**

| system | metric | mean_diff | ci95 | 显著 |
|---|---|---|---|---|
| chab-all | recall@10 | 0.0 | [0.0, 0.0] | 否 |
| chab-all | cov@10 | 0.0 | [0.0, 0.0] | 否 |
| chab-all | mrr@10 | 0.0 | [0.0, 0.0] | 否 |
| chab-noentity | recall@10 | 0.0 | [0.0, 0.0] | 否 |
| chab-noentity | cov@10 | 0.0 | [0.0, 0.0] | 否 |
| chab-noentity | mrr@10 | 0.0 | [0.0, 0.0] | 否 |
