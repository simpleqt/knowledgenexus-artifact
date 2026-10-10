#!/usr/bin/env python3
"""开源基准上的多查询扇出迭代（2026-09-28）：检验域内结论在开源多跳数据上成立。

- 库：测试栈 bench-hotpotqa / bench-musique / bench-2wikimultihopqa
  （"一文档一块" wiki 语料，各自全库检索；与域内 scoped 口径的差别如实记录）
- 题：questions.jsonl 固定种子抽 100 题/库
- 臂：base（原查询单路）vs fanout（原查询 + 3 LLM 变体，RRF k=60 融合取前 10）
- 产物：runs/ob-{arm}_{bench}_fullkb.json + 汇总 metrics
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  .venv/bin/python eval/baselines/scripts/openbench_fanout.py'
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, RUNS, write_predictions  # noqa: E402
from multiquery_fanout import RRF_K, N_VARIANTS, VARIANT_SYSTEM, llm_json, rrf_fuse  # noqa: E402

API = "http://127.0.0.1:28121/api/retrieve"
TOP_K = 10
N_Q = 100
SEED = 20260928
RESULT_DIR = Path(DATA).parent / "results" / "20260928_query_enhance"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

BENCHES = ["hotpotqa", "musique", "2wikimultihopqa"]


def retrieve(kb: str, query: str, k: int = TOP_K) -> list[dict]:
    body = {"kb_id": kb, "query": query, "top_k": k}
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:  # noqa: S310
        return json.loads(r.read())["chunks"]


def main() -> None:
    variants_log: dict[str, list[str]] = {}
    for bench in BENCHES:
        bdir = os.path.join(DATA, "bench", bench)
        kb = json.load(open(os.path.join(bdir, "kb.json")))["kb_id"]
        # gold 与基线同在 stable_id 空间；命中 doc_id 需经 idmap rev 译回
        idmap = json.load(open(os.path.join(bdir, "idmap.json")))
        rev = {v: k for k, v in idmap.items()}
        cache_file = RESULT_DIR / f"ob_variants_{bench}.json"
        if cache_file.exists():  # 变体缓存复用（修复重跑不重花 LLM）
            variants_log.update(json.load(open(cache_file, encoding="utf-8")))
        questions = [json.loads(l) for l in
                     open(os.path.join(bdir, "questions.jsonl"), encoding="utf-8")
                     if l.strip()]
        rng = random.Random(SEED)
        qs = rng.sample(questions, min(N_Q, len(questions)))
        base_rows, fan_rows = [], []
        for qi, q in enumerate(qs):
            gold = q["gold_doc_ids"]
            if isinstance(gold, str):
                import ast
                gold = ast.literal_eval(gold)
            gset = set(gold)

            if q["query"] not in variants_log:
                out = llm_json(VARIANT_SYSTEM, f"查询：{q['query']}")
                variants_log[q["query"]] = [
                    v.strip() for v in (out.get("variants") or [])[:N_VARIANTS]
                    if isinstance(v, str) and v.strip()]
            variants = variants_log[q["query"]]

            t0 = time.time()
            base_chunks = retrieve(kb, q["query"])
            base_rows.append((q["query"], gold, base_chunks, time.time() - t0))

            t0 = time.time()
            lists = [base_chunks] + [retrieve(kb, v) for v in variants]
            fused = rrf_fuse(lists, top_k=TOP_K) if variants else base_chunks
            fan_rows.append((q["query"], gold, fused, time.time() - t0))

            if (qi + 1) % 10 == 0:
                print(f"[{bench} {qi+1}/{len(qs)}]", flush=True)

        for arm, rows in (("base", base_rows), ("fanout", fan_rows)):
            per_query = [{"query": qq, "gold_doc_ids": gold,
                          "hits": [{"rank": i + 1, "doc_id": rev.get(c.get("doc_id"),
                                                                    c.get("doc_id")),
                                    "chunk_id": c.get("chunk_id"),
                                    "score": c.get("_score")}
                                   for i, c in enumerate(chunks)],
                          "latency_s": round(dt, 3)}
                         for qq, gold, chunks, dt in rows]
            write_predictions(f"ob-{arm}", "full-kb", bench,
                              {"wall_s": 0,
                               "note": f"测试栈 bench-{bench} 全库 {TOP_K}；"
                                       f"{arm}"
                                       + (f"（+{N_VARIANTS}变体 RRF k={RRF_K}）"
                                          if arm == "fanout" else "")},
                              per_query, 0)
        json.dump({q: v for q, v in variants_log.items()},
                  open(RESULT_DIR / f"ob_variants_{bench}.json", "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
