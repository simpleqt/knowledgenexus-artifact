#!/usr/bin/env python3
"""开源多跳问答基准 → 基线对比语料/题集（HippoRAG 2 复现包的本地数据）。

数据源：eval/baselines/src/HippoRAG/reproduce/dataset/{hotpotqa,2wikimultihopqa,musique}.json
（各 1000 题 + 候选语料；hotpot 式 supporting_facts+context / musique 式 paragraphs 双兼容）。

产物（eval/baselines/data/bench/<name>/）：
  passages.jsonl   池内语料：{stable_id, title, text}；stable_id = md5("<name>:<title>")[:32]
  questions.jsonl  题集（harness 口径）：{query, gold_doc_ids: [stable_id...]}
池构成：抽样题自身的 context/paragraphs（基准自带的强干扰段，保任务难度语义）
+ 随机补齐到 --pool 篇。全语料模式加 --full（入库耗时约 10 倍）。
用法：.venv/bin/python bench_prepare.py --dataset hotpotqa [--n-questions 150] [--pool 2000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import BASE

SRC = os.path.join(BASE, "src", "HippoRAG", "reproduce", "dataset")


def gold_titles(item: dict) -> set[str]:
    if "supporting_facts" in item:                      # hotpot / 2wiki 式
        return {t for t, _ in item["supporting_facts"]}
    return {p["title"] for p in item["paragraphs"] if p.get("is_supporting")}


def cand_titles(item: dict) -> set[str]:
    if "context" in item:
        return {t for t, _ in item["context"]}
    return {p["title"] for p in item["paragraphs"]}


def stable_id(dataset: str, title: str) -> str:
    return hashlib.md5(f"{dataset}:{title}".encode()).hexdigest()[:32]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "2wikimultihopqa", "musique"])
    ap.add_argument("--n-questions", type=int, default=150)
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--full", action="store_true", help="全语料池（忽略 --pool）")
    args = ap.parse_args()

    queries = json.load(open(os.path.join(SRC, f"{args.dataset}.json")))
    corpus = json.load(open(os.path.join(SRC, f"{args.dataset}_corpus.json")))
    by_title = {p["title"]: p for p in corpus}

    rng = random.Random(20260923)
    picked = rng.sample(queries, min(args.n_questions, len(queries)))
    for it in picked:
        missing = gold_titles(it) - set(by_title)
        if missing:
            sys.exit(f"gold 不在语料中（不应发生）: {missing}")

    pool_titles: set[str] = set()
    if args.full:
        pool_titles = set(by_title)
    else:
        # --pool 为硬上限：按随机题序贪心收题（该题全部 gold+干扰段入池），
        # 池满即止——保证"部分导入"规模可控（09-24 纪律）
        used = []
        for it in picked:
            add = cand_titles(it) & set(by_title)
            if len(pool_titles | add) > args.pool:
                continue
            pool_titles |= add
            used.append(it)
            if len(pool_titles) >= args.pool:
                break
        picked = used

    out = os.path.join(BASE, "data", "bench", args.dataset)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "passages.jsonl"), "w", encoding="utf-8") as f:
        for t in sorted(pool_titles):
            f.write(json.dumps({"stable_id": stable_id(args.dataset, t),
                                "title": t, "text": by_title[t]["text"]},
                               ensure_ascii=False) + "\n")
    with open(os.path.join(out, "questions.jsonl"), "w", encoding="utf-8") as f:
        for it in picked:
            f.write(json.dumps({"query": it["question"],
                                "gold_doc_ids": sorted(stable_id(args.dataset, t)
                                                       for t in gold_titles(it))},
                               ensure_ascii=False) + "\n")
    gold_docs = sorted({stable_id(args.dataset, t) for it in picked for t in gold_titles(it)})
    print(f"{args.dataset}: 池 {len(pool_titles)} 篇 / {len(picked)} 题 / gold 文档 {len(gold_docs)} 篇"
          f"（gold 密度 {len(gold_docs)/len(pool_titles):.1%}）")


if __name__ == "__main__":
    main()
