#!/usr/bin/env python3
"""LightRAG 基线（HKUDS lightrag-hku 1.4.13，图+向量混合，OpenAI 兼容端点）。

索引：整篇文档文本 insert（LightRAG 内部按 chunk_token_size=1200 tok 重切）；
查询：hybrid / naive 两模式，only_need_context=True 取 raw_data["data"]["chunks"]，
     按 file_path（=我们的 doc_id）映射回文档，缺 file_path 时用规范化子串匹配兜底。
用法：venvs/lightrag/bin/python run_lightrag.py --slice L --set nea [--modes hybrid,naive]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import INDEXES, load_docs, load_questions, load_services_env, load_slice_ids, write_predictions

_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return _WS.sub("", s).lower()


class DocMapper:
    """LightRAG chunk → 我们的 doc_id（file_path 即 doc_id，子串兜底）。"""

    def __init__(self, docs: list[dict]) -> None:
        self.known: set[str] = {d["doc_id"] for d in docs}
        self.corpus = [(d["doc_id"], norm(d["text"])) for d in docs]
        self.corpus.sort(key=lambda x: len(x[1]))  # 短文优先，减少误配

    def map_chunk(self, chunk: dict) -> str | None:
        fp = (chunk.get("file_path") or "").strip()
        if fp in self.known:
            return fp
        content = norm(chunk.get("content") or "")
        if len(content) < 40:
            return None
        for doc_id, text in self.corpus:
            if content in text:
                return doc_id
        return None


async def run(args: argparse.Namespace) -> None:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import EmbeddingFunc

    env = load_services_env()
    slice_ids = set(load_slice_ids(args.slice))
    docs = [d for d in load_docs() if d["doc_id"] in slice_ids]
    questions = load_questions(args.set)
    ws = os.path.join(INDEXES, f"lightrag_{args.slice}")
    os.makedirs(ws, exist_ok=True)

    from openai import AsyncOpenAI

    aic = AsyncOpenAI(base_url=env["EMBEDDING_API_BASE"], api_key=env["EMBEDDING_API_KEY"])

    async def _embed(texts: list[str]):
        out: list[list[float]] = []
        for i in range(0, len(texts), 32):
            r = await aic.embeddings.create(model=env["EMBEDDING_MODEL"],
                                            input=texts[i:i + 32])
            out.extend(d.embedding for d in r.data)
        import numpy as np
        return np.asarray(out, dtype=np.float32)

    async def _llm(prompt: str, system_prompt: str | None = None,
                   history_messages: list | None = None, **kw):
        kw.pop("model", None)   # 库调用侧可能注入 model=，统一以 env 为准
        return await openai_complete_if_cache(
            env["LLM_MODEL"], prompt, system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=env["LLM_API_BASE"], api_key=env["LLM_API_KEY"])

    rag = LightRAG(
        working_dir=ws,
        llm_model_func=_llm,
        llm_model_name=env["LLM_MODEL"],
        llm_model_max_async=args.max_async,
        max_parallel_insert=args.parallel_insert,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(env.get("EMBEDDING_DIM", 4096)),
            func=_embed,
        ),
        addon_params={"language": "English"},   # 语料主体为英文（NIM-A）
    )
    await rag.initialize_storages()
    try:
        await _pipeline(rag, args, slice_ids, docs, questions)
    finally:
        await rag.finalize_storages()


async def _pipeline(rag, args: argparse.Namespace, slice_ids: set,
                    docs: list[dict], questions: list[dict]) -> None:
    from lightrag import QueryParam

    # 索引：永远走 insert 循环——LightRAG 的 doc_status 自动跳过已处理文档（幂等）。
    # 注意不可用 graphml 文件存在性做跳过：图增量落盘，半建状态也存在该文件（已踩）。
    t0 = time.time()
    for d in docs:
        await rag.ainsert(d["text"], ids=[d["doc_id"]], file_paths=[d["doc_id"]])
    index_wall = time.time() - t0
    print(f"LightRAG index: {len(docs)} docs, {index_wall:.0f}s")

    mapper = DocMapper(docs)

    for mode in args.modes.split(","):
        per_query = []
        for q in questions:
            t0 = time.time()
            res = await rag.aquery_data(q["query"], param=QueryParam(
                mode=mode, only_need_context=True, chunk_top_k=20, top_k=60))
            dt = time.time() - t0
            chunks = (res.get("data", {}).get("chunks") if isinstance(res, dict)
          else ((getattr(res, "raw_data", None) or {}).get("data", {}).get("chunks"))) or []
            hits, seen_rank = [], 0
            unmapped = 0
            for ch in chunks:
                doc_id = mapper.map_chunk(ch)
                if doc_id is None:
                    unmapped += 1
                    continue
                seen_rank += 1
                hits.append({"rank": seen_rank, "doc_id": doc_id,
                             "chunk_id": None, "score": ch.get("score")})
            per_query.append({"query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
                              "hits": hits, "latency_s": round(dt, 3),
                              "unmapped_chunks": unmapped})
        n_un = sum(p["unmapped_chunks"] for p in per_query)
        n_hit = sum(len(p["hits"]) for p in per_query)
        print(f"[{mode}] hits={n_hit} unmapped={n_un}")
        write_predictions(f"lightrag-{mode}", args.slice,
                          os.path.basename(args.set).replace(".jsonl", ""),
                          {"wall_s": index_wall,
                           "note": f"mode={mode}, chunk_token_size=1200(default), "
                           f"unmapped={n_un}/{n_hit+n_un}"},
                          per_query, len(slice_ids))


async def bench_pipeline(rag, args: argparse.Namespace, env: dict) -> None:
    import json
    import os
    from common import BASE
    from lightrag import QueryParam
    d = os.path.join(BASE, "data", "bench", args.bench)
    docs = [json.loads(l) for l in open(os.path.join(d, "passages.jsonl"), encoding="utf-8") if l.strip()]
    questions = [json.loads(l) for l in open(os.path.join(d, "questions.jsonl"), encoding="utf-8") if l.strip()]
    mapper = DocMapper([{"doc_id": p["stable_id"],
                         "text": f"{p['title']}\n\n{p['text']}"} for p in docs])

    t0 = time.time()
    texts = [f"{p['title']}\n\n{p['text']}" for p in docs]
    sids = [p["stable_id"] for p in docs]
    for i in range(0, len(docs), 200):   # 列表式批量 ainsert：max_parallel_insert 才能跨文档并行
        await rag.ainsert(texts[i:i + 200], ids=sids[i:i + 200], file_paths=sids[i:i + 200])
    index_wall = time.time() - t0
    print(f"LightRAG bench index: {len(docs)} passages, {index_wall:.0f}s")

    for mode in args.modes.split(","):
        per_query, unmapped_total = [], 0
        for q in questions:
            t0 = time.time()
            res = await rag.aquery_data(q["query"], param=QueryParam(
                mode=mode, only_need_context=True, chunk_top_k=20, top_k=60))
            dt = time.time() - t0
            chunks = (res.get("data", {}).get("chunks") if isinstance(res, dict)
          else ((getattr(res, "raw_data", None) or {}).get("data", {}).get("chunks"))) or []
            hits, rank, unmapped = [], 0, 0
            for ch in chunks:
                sid = mapper.map_chunk(ch)
                if sid is None:
                    unmapped += 1
                    continue
                rank += 1
                hits.append({"rank": rank, "doc_id": sid, "chunk_id": None,
                             "score": ch.get("score")})
            unmapped_total += unmapped
            per_query.append({"query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
                              "hits": hits, "latency_s": round(dt, 3),
                              "unmapped_chunks": unmapped})
        write_predictions(f"lightrag-{mode}", "pool", args.bench,
                          {"wall_s": round(index_wall, 1),
                           "note": f"mode={mode}, passage 级, unmapped={unmapped_total}"},
                          per_query, len(docs))


async def bench_run(args: argparse.Namespace) -> None:
    """基准模式：语料=data/bench/<name>/passages.jsonl，stable_id 即 doc_id。"""
    import os

    from openai import AsyncOpenAI

    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import EmbeddingFunc

    env = load_services_env()
    os.environ.setdefault("OPENAI_API_KEY", env["LLM_API_KEY"])
    aic = AsyncOpenAI(base_url=env["EMBEDDING_API_BASE"], api_key=env["EMBEDDING_API_KEY"])

    async def _embed(texts: list[str]):
        import numpy as np
        out: list[list[float]] = []
        for i in range(0, len(texts), 32):
            r = await aic.embeddings.create(model=env["EMBEDDING_MODEL"],
                                            input=texts[i:i + 32])
            out.extend(d.embedding for d in r.data)
        return np.asarray(out, dtype=np.float32)

    async def _llm(prompt: str, system_prompt: str | None = None,
                   history_messages: list | None = None, **kw):
        kw.pop("model", None)
        return await openai_complete_if_cache(
            env["LLM_MODEL"], prompt, system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=env["LLM_API_BASE"], api_key=env["LLM_API_KEY"])

    ws = os.path.join(INDEXES, f"lightrag_bench_{args.bench}")
    os.makedirs(ws, exist_ok=True)
    rag = LightRAG(
        working_dir=ws,
        llm_model_func=_llm, llm_model_name=env["LLM_MODEL"],
        llm_model_max_async=args.max_async, max_parallel_insert=args.parallel_insert,
        embedding_func=EmbeddingFunc(embedding_dim=int(env.get("EMBEDDING_DIM", 4096)),
                                     func=_embed),
        addon_params={"language": "English"},
    )
    await rag.initialize_storages()
    try:
        await bench_pipeline(rag, args, env)
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="L", choices=["L", "S"])
    ap.add_argument("--set", default=None)
    ap.add_argument("--modes", default="hybrid,naive")
    ap.add_argument("--max-async", type=int, default=70, help="LLM 并发（默认 70）")
    ap.add_argument("--parallel-insert", type=int, default=4, help="文档级并行插入（默认 4）")
    ap.add_argument("--bench", default=None, help="基准模式：data/bench/<name>")
    args = ap.parse_args()
    if args.bench:
        asyncio.run(bench_run(args))
    elif args.set:
        asyncio.run(run(args))
    else:
        ap.error("--set 或 --bench 必填其一")
