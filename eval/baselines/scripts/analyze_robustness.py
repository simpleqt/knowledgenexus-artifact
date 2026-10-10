#!/usr/bin/env python3
"""改写鲁棒性分析：从多查询扇出产物 paraphrase_{set}.jsonl 统计改写敏感性。

指标（每题 3 变体）：
  var_hit10      变体首金标进前 10 的比例（原查询同口径并列）
  var_mrr        变体首金标 MRR@10（原查询并列）
  worse_rate     变体名次劣于原查询（rank 变 > rank 原，含原命中而变体未命中）的比例
  frozen_set     逐变体展开为独立题目行，冻结为 data/<set>_para.jsonl
用法：analyze_robustness.py --set nea --set ner [--freeze]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import BASE, DATA, load_questions  # noqa: E402

RESULT_DIR = os.path.join(BASE, "results", "20260928_query_enhance")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()

    lines = ["# 改写鲁棒性（查询侧语义增强的压力测试维度）", ""]
    for set_name in args.set:
        rows = [json.loads(l) for l in
                open(os.path.join(RESULT_DIR, f"paraphrase_{set_name}.jsonl"),
                     encoding="utf-8") if l.strip()]
        orig_ranks, var_ranks = [], []
        frozen = []
        for r in rows:
            gold = set(r["gold_doc_ids"])
            orank = r["first_gold_rank"]["original"]
            orig_ranks.append(orank)
            for v, vrank in zip(r["variants"], r["first_gold_rank"]["variants"]):
                var_ranks.append(vrank)
                if args.freeze and v:
                    frozen.append({"query": v, "gold_doc_ids": r["gold_doc_ids"],
                                   "mode": "para", "origin_query": r["query"],
                                   "source_set": set_name})

        def hit10(ranks):
            vals = [r for r in ranks if r is not None]
            return sum(1 for r in ranks if r is not None and r <= 10) / len(ranks)

        def mrr(ranks):
            return sum(1.0 / r for r in ranks if r is not None and r <= 10) / len(ranks)

        def worse_rate():
            n = worse = 0
            for r in rows:
                o = r["first_gold_rank"]["original"]
                for v in r["first_gold_rank"]["variants"]:
                    n += 1
                    if (v is None and o is not None) or (
                            v is not None and o is not None and v > o):
                        worse += 1
            return worse / n if n else 0.0

        lines += [
            f"## {set_name}（{len(rows)} 题 × {sum(len(r['variants']) for r in rows)} 变体）",
            "",
            "| 口径 | hit@10 | MRR@10 |", "|---|---|---|",
            f"| 原查询 | {hit10(orig_ranks):.3f} | {mrr(orig_ranks):.3f} |",
            f"| 改写变体 | {hit10(var_ranks):.3f} | {mrr(var_ranks):.3f} |",
            f"| 变体名次劣于原查询比例 | {worse_rate():.3f} | |", ""]
        if args.freeze and frozen:
            out = os.path.join(DATA, f"{set_name}_para.jsonl")
            with open(out, "w", encoding="utf-8") as f:
                for row in frozen:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            lines.append(f"冻结改写题集：`eval/baselines/data/{set_name}_para.jsonl"
                         f"`（{len(frozen)} 题，金标继承原题）")
            lines.append("")
    text = "\n".join(lines)
    print(text)
    open(os.path.join(RESULT_DIR, "robustness.md"), "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
