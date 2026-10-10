#!/usr/bin/env python3
"""图谱治理消融统计（2026-09-30）：逐题配对 bootstrap + 符号检验。

用法：.venv/bin/python graph_gov_stats.py [结果json] [基线臂] [对照臂...]（缺省
graph_gov_ablation.json，基线 prod，对照全部其余臂）。
口径：hit@10/hit@20/MRR@10 逐题 0/1（MRR 取倒数贡献），配对 bootstrap
10,000 重采样 95% CI；n<5 提示集群口径不适用。纯本地计算，不触库。
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT = Path(REPO) / "eval/baselines/results/20260930_graphgov/graph_gov_ablation.json"


def per_q(rows: list[dict]) -> dict[str, tuple[float, float, float]]:
    out = {}
    for r in rows:
        rank = r["rank"]
        h10 = 1.0 if (rank or 99) <= 10 else 0.0
        h20 = 1.0 if (rank or 99) <= 20 else 0.0
        mrr = 1.0 / rank if (rank and rank <= 10) else 0.0
        out[r["query"]] = (h10, h20, mrr)
    return out


def paired_bootstrap(a: list[float], b: list[float], n: int = 10000, seed: int = 20260930) -> tuple[float, float, float]:
    """返回 (Δ均值, CI_lo, CI_hi)，Δ=对照-基线。"""
    rng = random.Random(seed)
    diffs = [y - x for x, y in zip(a, b)]
    if not diffs:
        return 0.0, 0.0, 0.0
    m = sum(diffs) / len(diffs)
    boots = []
    for _ in range(n):
        s = sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs)
        boots.append(s)
    boots.sort()
    return m, boots[int(0.025 * n)], boots[int(0.975 * n)]


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT)
    base_arm = sys.argv[2] if len(sys.argv) > 2 else "prod"
    data = json.load(open(path, encoding="utf-8"))
    keys = [k for k in data if "|" in k]
    sets = sorted({k.split("|")[1] for k in keys})
    arms = sorted({k.split("|")[0] for k in keys})
    others = [a for a in arms if sys.argv[3:]] if len(sys.argv) > 3 else \
             [a for a in arms if a != base_arm]
    for s in sets:
        bk = f"{base_arm}|{s}"
        if bk not in data:
            continue
        bp = per_q(data[bk]["rows"])
        print(f"\n## {s}（n={data[bk]['n']}，基线={base_arm}）")
        print(f"| 臂 | hit@10 | hit@20 | MRR@10 | Δhit@10[95%CI] | ΔMRR[95%CI] | p50 |")
        print("|---|---|---|---|---|---|---|")
        print(f"| {base_arm} | {data[bk]['hit@10']:.3f} | {data[bk]['hit@20']:.3f} "
              f"| {data[bk]['mrr@10']:.3f} | — | — | {data[bk]['p50_s']}s |")
        for a in others:
            ak = f"{a}|{s}"
            if ak not in data:
                continue
            ap = per_q(data[ak]["rows"])
            qs = [q for q in bp if q in ap]
            b_h10 = [bp[q][0] for q in qs]
            a_h10 = [ap[q][0] for q in qs]
            b_mrr = [bp[q][2] for q in qs]
            a_mrr = [ap[q][2] for q in qs]
            d1, lo1, hi1 = paired_bootstrap(b_h10, a_h10)
            d2, lo2, hi2 = paired_bootstrap(b_mrr, a_mrr)
            w = sum(1 for q in qs if ap[q][0] > bp[q][0])
            l = sum(1 for q in qs if ap[q][0] < bp[q][0])
            print(f"| {a} | {data[ak]['hit@10']:.3f} | {data[ak]['hit@20']:.3f} "
                  f"| {data[ak]['mrr@10']:.3f} | {d1:+.3f}[{lo1:+.3f},{hi1:+.3f}] "
                  f"({'胜/负 %d/%d' % (w, l) if qs else ''}) | {d2:+.3f}[{lo2:+.3f},{hi2:+.3f}] "
              f"| {data[ak]['p50_s']}s |")


if __name__ == "__main__":
    main()
