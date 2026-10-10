#!/usr/bin/env python3
"""LlamaIndex vanilla 向量 RAG 基线（块级 embed + 内存向量索引，检索层 only）。
用法：venvs/lite/bin/python run_llamaindex.py --slice L --set nea [--rebuild]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import INDEXES, load_chunks, load_questions, load_services_env, load_slice_ids, write_predictions

from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai_like import OpenAILikeEmbedding


def build_or_load(slice_name: str, chunk_rows: list[dict], emb) -> VectorStoreIndex:
    persist = os.path.join(INDEXES, f"llamaindex_{slice_name}")
    if os.path.exists(os.path.join(persist, "docstore.json")) and not ARGS_REBUILD:
        sc = StorageContext.from_defaults(persist_dir=persist)
        return load_index_from_storage(sc)
    os.makedirs(persist, exist_ok=True)
    nodes = [TextNode(text=c["content_text"], id_=c["chunk_id"],
                      metadata={"doc_id": c["doc_id"], "title": c.get("title_text") or ""})
             for c in chunk_rows]
    t0 = time.time()
    index = VectorStoreIndex(nodes, show_progress=True)
    print(f"embed+index: {time.time()-t0:.0f}s over {len(nodes)} nodes")
    index.storage_context.persist(persist_dir=persist)
    return index


ARGS_REBUILD = False


def bench_main(args, env) -> None:
    import json
    import os
    from common import BASE
    from llama_index.core import VectorStoreIndex
    from llama_index.embeddings.openai_like import OpenAILikeEmbedding
    emb = OpenAILikeEmbedding(
        model_name=env["EMBEDDING_MODEL"], api_base=env["EMBEDDING_API_BASE"],
        api_key=env["EMBEDDING_API_KEY"], embed_dim=int(env.get("EMBEDDING_DIM", 4096)),
        timeout=120, max_retries=5)
    Settings.embed_model = emb
    Settings.llm = None
    d = os.path.join(BASE, "data", "bench", args.bench)
    passages = [json.loads(l) for l in open(os.path.join(d, "passages.jsonl"), encoding="utf-8") if l.strip()]
    questions = [json.loads(l) for l in open(os.path.join(d, "questions.jsonl"), encoding="utf-8") if l.strip()]
    persist = os.path.join(INDEXES, f"llamaindex_bench_{args.bench}")
    if os.path.exists(os.path.join(persist, "docstore.json")) and not ARGS_REBUILD:
        sc = StorageContext.from_defaults(persist_dir=persist)
        index = load_index_from_storage(sc)
    else:
        os.makedirs(persist, exist_ok=True)
        nodes = [TextNode(text=p["title"] + "\n" + p["text"], id_=p["stable_id"],
                          metadata={"doc_id": p["stable_id"]}) for p in passages]
        t0 = time.time()
        index = VectorStoreIndex(nodes, show_progress=True)
        wall = time.time() - t0
        print(f"embed+index: {wall:.0f}s over {len(nodes)} passages")
        index.storage_context.persist(persist_dir=persist)
    retriever = index.as_retriever(similarity_top_k=20)
    per_query = []
    for q in questions:
        t0 = time.time()
        nodes = retriever.retrieve(q["query"])
        per_query.append({"query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
                          "hits": [{"rank": i + 1, "doc_id": n.metadata["doc_id"],
                                    "chunk_id": n.id_, "score": round(float(n.score), 4)}
                                   for i, n in enumerate(nodes)],
                          "latency_s": round(time.time() - t0, 3)})
    write_predictions("llamaindex", "pool", args.bench,
                      {"wall_s": None, "note": f"embed={env['EMBEDDING_MODEL']}, passage 级"},
                      per_query, len(passages))


def main() -> None:
    global ARGS_REBUILD
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="L", choices=["L", "S", "full"])
    ap.add_argument("--set", default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--bench", default=None, help="基准模式：data/bench/<name>")
    args = ap.parse_args()
    ARGS_REBUILD = args.rebuild

    env = load_services_env()
    if args.bench:
        bench_main(args, env)
        return
    if not args.set:
        ap.error("--set 必填（或用 --bench）")
    emb = OpenAILikeEmbedding(
        model_name=env["EMBEDDING_MODEL"], api_base=env["EMBEDDING_API_BASE"],
        api_key=env["EMBEDDING_API_KEY"], embed_dim=int(env.get("EMBEDDING_DIM", 4096)),
        timeout=120, max_retries=5,
        reuse_client=False,  # 每次显式建连，避免多线程共享 client 状态
    )
    Settings.embed_model = emb
    Settings.llm = None

    if args.slice == "full":                      # 全库模式：语料=data/full_chunks.jsonl
        import json
        import os
        from common import DATA
        chunk_rows = [json.loads(l) for l in
                      open(os.path.join(DATA, "full_chunks.jsonl"), encoding="utf-8") if l.strip()]
        n_docs = len({c["doc_id"] for c in chunk_rows})
    else:
        slice_ids = set(load_slice_ids(args.slice))
        chunk_rows = [c for c in load_chunks() if c["doc_id"] in slice_ids]
        n_docs = len(slice_ids)
    questions = load_questions(args.set)

    index = build_or_load(args.slice, chunk_rows, emb)
    retriever = index.as_retriever(similarity_top_k=20)

    per_query = []
    for q in questions:
        t0 = time.time()
        nodes = retriever.retrieve(q["query"])
        dt = time.time() - t0
        hits = [{"rank": i + 1, "doc_id": n.metadata["doc_id"], "chunk_id": n.id_,
                 "score": round(float(n.score), 4)} for i, n in enumerate(nodes)]
        per_query.append({"query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
                          "hits": hits, "latency_s": round(dt, 3)})
    write_predictions("llamaindex", args.slice,
                      os.path.basename(args.set).replace(".jsonl", ""),
                      {"wall_s": None, "note": f"embed={env['EMBEDDING_MODEL']}, "
                       "SimpleVectorStore(HNSW 无), 检索层 only"},
                      per_query, n_docs)


if __name__ == "__main__":
    main()
