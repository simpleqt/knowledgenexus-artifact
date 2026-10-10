#!/usr/bin/env python3
"""HippoRAG 2 基线（OSU-NLP-Group，OpenIE+PPR 图检索，OpenAI 兼容端点）。

索引：整篇文档为 Chunk(source_id=doc_id)，内部按 800 tok/128 overlap 重切；
查询：retrieve() 返回 QuerySolution(doc_metadata 携带 doc_id) → 直接映射。
用法：venvs/hipporag/bin/python run_hipporag.py --slice S --set nea_S
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import INDEXES, load_docs, load_questions, load_services_env, load_slice_ids, write_predictions

HIPPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "HippoRAG"))
sys.path.insert(0, HIPPO_SRC)


def chunk_text(text: str, size: int = 3000, overlap: int = 400) -> list[str]:
    """按段落边界切 ~size 字符、overlap 重叠（英文 ≈ 750 tok/段，上限 32k 上下文内）。"""
    paras = text.split("\n\n")
    pieces, buf = [], ""
    for p in paras:
        while len(p) > size:            # 超长单段硬切
            pieces.append(p[:size])
            p = p[size - overlap:]
        if len(buf) + len(p) + 2 <= size or not buf:
            buf = f"{buf}\n\n{p}".strip() if buf else p
        else:
            pieces.append(buf)
            buf = buf[-overlap:] + "\n\n" + p if overlap else p
    if buf:
        pieces.append(buf)
    return pieces


def bench_main(args) -> None:
    """基准模式：语料=data/bench/<name>/passages.jsonl（stable_id 即 doc_id），单段多为 1 块。"""
    import json
    import os

    from common import BASE, load_services_env
    env = load_services_env()
    os.environ.setdefault("OPENAI_API_KEY", env["LLM_API_KEY"])

    from src.hipporag.HippoRAG import HippoRAG
    from src.hipporag.utils.config_utils import BaseConfig
    from src.hipporag.utils.misc_utils import Chunk

    d = os.path.join(BASE, "data", "bench", args.bench)
    docs = [json.loads(l) for l in open(os.path.join(d, "passages.jsonl"), encoding="utf-8") if l.strip()]
    questions = [json.loads(l) for l in open(os.path.join(d, "questions.jsonl"), encoding="utf-8") if l.strip()]

    cfg = BaseConfig(
        save_dir=os.path.join(INDEXES, f"hipporag_bench_{args.bench}"),
        llm_name=env["LLM_MODEL"], llm_base_url=env["LLM_API_BASE"],
        embedding_model_name=env["EMBEDDING_MODEL"],
        embedding_base_url=env["EMBEDDING_API_BASE"],
        embedding_provider="openai",
        openie_mode="online",
        openie_max_workers=70,
        force_index_from_scratch=True,
        preprocess_chunk_max_token_size=800,
        preprocess_chunk_overlap_token_size=128,
        retrieval_top_k=20,
        embedding_batch_size=64,
    )
    hippo = HippoRAG(global_config=cfg)
    chunk_inputs = [Chunk(content=f"{p['title']}\n\n{p['text']}", source_id=p["stable_id"],
                          metadata={"doc_id": p["stable_id"]}) for p in docs]
    print(f"HippoRAG bench 输入：{len(docs)} 篇 → {len(chunk_inputs)} 段")
    t0 = time.time()
    hippo.index(chunk_inputs)
    index_wall = time.time() - t0

    solutions = hippo.retrieve([q["query"] for q in questions], num_to_retrieve=20)
    per_query = []
    for q, sol in zip(questions, solutions):
        hits = []
        for i, md in enumerate(sol.doc_metadata or []):
            doc_id = (md or {}).get("doc_id")
            if doc_id:
                hits.append({"rank": len(hits) + 1, "doc_id": doc_id, "chunk_id": None,
                             "score": round(float(sol.doc_scores[i]), 4) if sol.doc_scores is not None else None})
        per_query.append({"query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
                          "hits": hits, "latency_s": None})
    write_predictions("hipporag", "pool", args.bench,
                      {"wall_s": round(index_wall, 1), "note": "openie=online, passage 级"},
                      per_query, len(docs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="S", choices=["L", "S"])
    ap.add_argument("--set", required=True)
    ap.add_argument("--bench", default=None, help="基准模式：data/bench/<name>")
    args = ap.parse_args()
    if args.bench:
        bench_main(args)
        return

    env = load_services_env()
    os.environ.setdefault("OPENAI_API_KEY", env["LLM_API_KEY"])

    from src.hipporag.HippoRAG import HippoRAG
    from src.hipporag.utils.config_utils import BaseConfig
    from src.hipporag.utils.misc_utils import Chunk

    slice_ids = set(load_slice_ids(args.slice))
    docs = [d for d in load_docs() if d["doc_id"] in slice_ids]
    questions = load_questions(args.set)
    save_dir = os.path.join(INDEXES, f"hipporag_{args.slice}")

    cfg = BaseConfig(
        save_dir=save_dir,
        llm_name=env["LLM_MODEL"], llm_base_url=env["LLM_API_BASE"],
        embedding_model_name=env["EMBEDDING_MODEL"],
        embedding_base_url=env["EMBEDDING_API_BASE"],
        embedding_provider="openai",
        openie_mode="online",
        openie_max_workers=70,
        force_index_from_scratch=True,   # 复用 llm_cache/OpenIE 存储，授权重建图（崩溃续跑）
        preprocess_chunk_max_token_size=800,
        preprocess_chunk_overlap_token_size=128,
        retrieval_top_k=20,
        embedding_batch_size=64,
    )
    hippo = HippoRAG(global_config=cfg)

    chunk_inputs = [Chunk(content=pc, source_id=d["doc_id"],
                          metadata={"doc_id": d["doc_id"], "title": d["title"]})
                    for d in docs for pc in chunk_text(d["text"])]
    print(f"HippoRAG 输入：{len(docs)} 篇 → {len(chunk_inputs)} 段")
    t0 = time.time()
    hippo.index(chunk_inputs)
    index_wall = time.time() - t0
    print(f"HippoRAG index: {len(docs)} docs, {index_wall:.0f}s")

    solutions = hippo.retrieve([q["query"] for q in questions], num_to_retrieve=20)
    per_query = []
    for q, sol in zip(questions, solutions):
        hits = []
        for i, md in enumerate(sol.doc_metadata or []):
            doc_id = (md or {}).get("doc_id")
            if doc_id:
                hits.append({"rank": len(hits) + 1, "doc_id": doc_id, "chunk_id": None,
                             "score": round(float(sol.doc_scores[i]), 4) if sol.doc_scores is not None else None})
        per_query.append({"query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
                          "hits": hits, "latency_s": None})
    write_predictions("hipporag", args.slice,
                      os.path.basename(args.set).replace(".jsonl", ""),
                      {"wall_s": round(index_wall, 1),
                       "note": "openie=online(qwen3.5-35b), chunk=800tok/128ov, "
                       "PPR retrieval_top_k=20, 无 reranker 文件（DSPyFilter 关）"},
                      per_query, len(slice_ids))


if __name__ == "__main__":
    main()
