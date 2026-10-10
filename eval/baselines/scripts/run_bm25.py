#!/usr/bin/env python3
"""BM25 稀疏检索基线（rank_bm25 + jieba 分词，块级检索 → 文档级映射）。
用法：venvs/lite/bin/python run_bm25.py --slice L --set nea
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import load_chunks, load_questions, load_slice_ids, write_predictions

import jieba
from rank_bm25 import BM25Okapi


def toks(text: str) -> list[str]:
    return [t for t in jieba.lcut(text.lower()) if t.strip()]


def bench_main(args) -> None:
    import json
    import os
    from common import BASE
    d = os.path.join(BASE, "data", "bench", args.bench)
    passages = [json.loads(l) for l in open(os.path.join(d, "passages.jsonl"), encoding="utf-8") if l.strip()]
    questions = [json.loads(l) for l in open(os.path.join(d, "questions.jsonl"), encoding="utf-8") if l.strip()]
    t0 = time.time()
    bm25 = BM25Okapi([toks(p["title"] + "\n" + p["text"]) for p in passages])
    wall = time.time() - t0
    per_query = []
    for q in questions:
        t1 = time.time()
        scores = bm25.get_scores(toks(q["query"]))
        top = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)[:20]
        per_query.append({"query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
                          "hits": [{"rank": i + 1, "doc_id": passages[j]["stable_id"],
                                    "chunk_id": passages[j]["stable_id"],
                                    "score": round(float(scores[j]), 4)} for i, j in enumerate(top)],
                          "latency_s": round(time.time() - t1, 4)})
    write_predictions("bm25", "pool", args.bench,
                      {"wall_s": round(wall, 1), "engine": "rank_bm25 + jieba（passage 级）"},
                      per_query, len(passages))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="L", choices=["L", "S", "full"])
    ap.add_argument("--set", required=True)
    ap.add_argument("--bench", default=None, help="基准模式：data/bench/<name>，语料=passages.jsonl")
    args = ap.parse_args()

    if args.bench:
        bench_main(args)
        return
    if args.slice == "full":                      # 全库模式：语料=data/full_chunks.jsonl
        import json
        import os
        from common import DATA
        chunks = [json.loads(l) for l in open(os.path.join(DATA, "full_chunks.jsonl"),
                                              encoding="utf-8") if l.strip()]
        n_docs = len({c["doc_id"] for c in chunks})
    else:
        slice_ids = set(load_slice_ids(args.slice))
        chunks = [c for c in load_chunks() if c["doc_id"] in slice_ids]
        n_docs = len(slice_ids)
    questions = load_questions(args.set)

    t0 = time.time()
    corpus_tokens = [toks(c["content_text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    index_wall = time.time() - t0
    print(f"BM25 index: {len(chunks)} chunks, {index_wall:.1f}s")

    per_query = []
    for q in questions:
        t1 = time.time()
        scores = bm25.get_scores(toks(q["query"]))
        top = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:20]
        dt = time.time() - t1
        hits = [{"rank": i + 1, "doc_id": chunks[i]["doc_id"],
                 "chunk_id": chunks[i]["chunk_id"], "score": round(float(scores[i]), 4)}
                for i in top]
        per_query.append({"query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
                          "hits": hits, "latency_s": round(dt, 4)})
    write_predictions("bm25", args.slice, os.path.basename(args.set).replace(".jsonl", ""),
                      {"wall_s": round(index_wall, 1), "engine": "rank_bm25 BM25Okapi + jieba"},
                      per_query, n_docs)


if __name__ == "__main__":
    main()
