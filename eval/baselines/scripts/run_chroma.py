#!/usr/bin/env python3
"""ChromaDB 向量 RAG 基线（块级 embed + Chroma 持久集合，检索层 only）。

与 run_llamaindex.py 同口径：同切块（load_chunks）、同嵌入端点（deploy/.env.services）、
块级 top-20；差异仅向量库本体（Chroma 持久化 vs llamaindex 内存索引）——
入库速度与检索质量可分别对照（ChromaDB 是最常用的 MVP 级知识库基线）。
用法：venvs/fast/bin/python run_chroma.py --slice L --set nea [--rebuild]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import BASE, INDEXES, load_chunks, load_questions, load_services_env, load_slice_ids, write_predictions  # noqa: E402


def embed_texts(env: dict, texts: list[str], batch: int = 32) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), batch):
        body = json.dumps({"model": env["EMBEDDING_MODEL"], "input": texts[i:i + batch]}).encode()
        req = urllib.request.Request(env["EMBEDDING_API_BASE"].rstrip("/") + "/embeddings",
                                     data=body, headers={"Content-Type": "application/json",
                                                         "Authorization": f"Bearer {env['EMBEDDING_API_KEY']}"})
        with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310
            data = json.loads(r.read())["data"]
        out.extend(d["embedding"] for d in sorted(data, key=lambda x: x["index"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="L", choices=["L", "S", "full"])
    ap.add_argument("--set", required=True)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    env = load_services_env()

    import chromadb
    persist = os.path.join(INDEXES, f"chroma_{args.slice}")
    client = chromadb.PersistentClient(path=persist)
    col = client.get_or_create_collection("knc", metadata={"hnsw:space": "cosine"})

    if args.slice == "full":                      # 全库模式：语料=data/full_chunks.jsonl
        import json
        from common import DATA
        chunks = [json.loads(l) for l in
                  open(os.path.join(DATA, "full_chunks.jsonl"), encoding="utf-8") if l.strip()]
        n_docs = len({c["doc_id"] for c in chunks})
    else:
        slice_ids = set(load_slice_ids(args.slice))
        chunks = [c for c in load_chunks() if c["doc_id"] in slice_ids]
        n_docs = len(slice_ids)
    have = col.count()
    if have < len(chunks) or args.rebuild:
        if have:
            client.delete_collection("knc")
            col = client.get_or_create_collection("knc", metadata={"hnsw:space": "cosine"})
        t0 = time.time()
        # 全库 196k 块流式嵌入分批写入（嵌入向量的 Python list 峰值 ~25GB，
        # 整库一次性持有会更高——按 5k 块批嵌入+写+释放）
        CH_BATCH = 5000
        for i in range(0, len(chunks), CH_BATCH):
            part = chunks[i:i + CH_BATCH]
            vecs = embed_texts(env, [c["content_text"] for c in part])
            col.add(ids=[c["chunk_id"] for c in part],
                    embeddings=vecs,
                    documents=[c["content_text"] for c in part],
                    metadatas=[{"doc_id": c["doc_id"]} for c in part])
            print(f"  chroma 批 {min(i + CH_BATCH, len(chunks))}/{len(chunks)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        wall = time.time() - t0
        print(f"chroma 入库：{len(chunks)} 块流式，{wall:.0f}s")
    else:
        wall = 0
        print(f"chroma 已有 {have} 块（复用）")

    questions = load_questions(args.set)
    per_query = []
    qtexts = [q["query"] for q in questions]
    qvecs = embed_texts(env, qtexts, batch=16)
    for q, qv in zip(questions, qvecs):
        t0 = time.time()
        res = col.query(query_embeddings=[qv], n_results=min(20, col.count()))
        dt = time.time() - t0
        hits = [{"rank": i + 1, "doc_id": m["doc_id"], "chunk_id": cid, "score": None}
                for i, (m, cid) in enumerate(zip(res["metadatas"][0], res["ids"][0]))]
        per_query.append({"query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
                          "hits": hits, "latency_s": round(dt, 3)})
    write_predictions("chroma", args.slice,
                      os.path.basename(args.set).replace(".jsonl", ""),
                      {"wall_s": round(wall), "note": "embed=共享 Qwen3-8B 端点，块级，Chroma 持久化 cosine"},
                      per_query, n_docs)


if __name__ == "__main__":
    main()
