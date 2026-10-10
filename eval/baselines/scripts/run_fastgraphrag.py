#!/usr/bin/env python3
"""fast-graphrag 基线（快入库图方法代表，S 切片）。

目的：检验"快图 RAG"声称（增量抽取/PPR）在我们的共享 qwen 端点与核学科语料上的
实际入库速度与检索质量——规模墙发现是否 LightRAG 特有。
chunk→doc 映射：响应 chunk 文本前缀查语料文本表（fast-graphrag 的 chunk id 是
内容哈希，无外部 id 通道）。
用法：venvs/fgrag/bin/python run_fastgraphrag.py --slice S --set nea_S [--rebuild]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import INDEXES, load_docs, load_questions, load_services_env, load_slice_ids, write_predictions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="S")
    ap.add_argument("--set", default="nea_S")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--start-group", type=int, default=0,
                    help="断点续跑：跳过前 N 组入库（组=10 篇；索引复用不重建）")
    args = ap.parse_args()
    env = load_services_env()

    from fast_graphrag import GraphRAG
    from fast_graphrag._llm import OpenAIEmbeddingService, OpenAILLMService

    class QwenEmbeddingService(OpenAIEmbeddingService):
        """去掉 dimensions 参数——Qwen3-Embedding 不支持 matryoshka 降维，传 4096
        也被 serve 拒收（400）；嵌入维度由模型自身决定（=4096）。"""

        async def _embedding_request(self, input, model):
            from openai import AsyncOpenAI
            client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=300)
            return await client.embeddings.create(
                model=model, input=input, encoding_format="float")

    workdir = os.path.join(INDEXES, f"fgrag_{args.slice}")
    os.makedirs(workdir, exist_ok=True)

    llm = OpenAILLMService(model=env["LLM_MODEL"], base_url=env["LLM_API_BASE"],
                           api_key=env["LLM_API_KEY"])
    emb = QwenEmbeddingService(model=env["EMBEDDING_MODEL"], base_url=env["EMBEDDING_API_BASE"],
                               api_key=env["EMBEDDING_API_KEY"], embedding_dim=4096)

    grag = GraphRAG(
        working_dir=workdir,
        domain="核电子学与核探测技术（nuclear electronics and instrumentation）",
        example_queries="有哪些研究同时涉及硅光电倍增管与中子伽马甄别？|SiPM 的暗噪声如何测量？",
        entity_types=["设备", "材料", "方法", "物理量", "机构", "人物"],
        config=GraphRAG.Config(llm_service=llm, embedding_service=emb),
    )

    slice_ids = set(load_slice_ids(args.slice))
    docs = [d for d in load_docs() if d["doc_id"] in slice_ids]
    marker = os.path.join(workdir, ".ingested")
    wall = 0.0
    if args.rebuild or not os.path.exists(marker):
        # 分组列表 insert：并行单位=单次 insert 的文档数（每文档一个抽取任务，
        # 任务内再按块 gather）——传 list[str] 而非拼串（拼串=1 任务串行）；
        # 组=10 → 峰值 ~10×7 并发 LLM 调用，贴端点甜点区。
        # CONCURRENT_TASK_LIMIT（fast_graphrag LLM/嵌入 service 共用）必须设小：
        # 不限时默认 1024，客户端会同时开 400+ 连接把 .32 router.py 压死
        # （2026-09-27 事故：414 连接风暴 → router 死锁，重启后 10 分钟复现）。
        GROUP = 10
        start = args.start_group * GROUP
        t0 = time.time() - (args.start_group * 29 * 60 if args.start_group else 0)  # 估时不重置
        for i in range(start, len(docs), GROUP):
            part = docs[i:i + GROUP]
            asyncio.run(grag.async_insert([d["title"] + "\n" + d["text"] for d in part]))
            print(f"  group {i // GROUP + 1}/{(len(docs) + GROUP - 1) // GROUP}"
                  f"（{i + len(part)}/{len(docs)} 篇，{time.time() - t0:.0f}s）", flush=True)
        wall = time.time() - t0
        open(marker, "w").write(str(round(wall)))
        print(f"fgrag 入库：{len(docs) - start if start else len(docs)} 篇分组 insert"
              f"（组={GROUP}，start_group={args.start_group}）{wall:.0f}s", flush=True)
    else:
        wall = float(open(marker).read())
        print(f"fgrag 已入库（{wall:.0f}s，复用）")

    # chunk 文本 → doc 归属：fast-graphrag 内部重切块，chunk 不以文档前缀开头
    # （且剥掉标题 "# "），前缀精确匹配恒空；且 chunk 常以 ".|表头|公式" 等
    # 标点开头使单探针子串漏配（2026-09-27 实测 8 条中 2 条 None）——多探针：
    # 首段/去标点首段/中段各取一窗，任一命中即归属，带 memo。
    doc_texts = [(d["text"], d["doc_id"]) for d in docs]
    _doc_memo: dict[str, str | None] = {}

    def doc_of(chunk) -> str | None:
        content = getattr(chunk, "content", None) or getattr(chunk, "text", "") or ""
        if not content.strip():
            return None
        if content[:80] in _doc_memo:
            return _doc_memo[content[:80]]
        probes = {content[:60].lstrip(),
                  content.lstrip(" .,;:|#$\\%~()[]{}\"'\n\t")[:60]}
        if len(content) > 160:
            probes.add(content[100:160].lstrip())
        hit = next((did for t, did in doc_texts
                    for p in probes if p and p in t), None)
        _doc_memo[content[:80]] = hit
        return hit

    questions = load_questions(args.set)
    per_query = []

    # 查询生命周期包裹：本版本 fast_graphrag 的 async_query() 不做
    # query_start/done（同步 query() 才有）——纯查询进程不包则 HNSW 存储
    # _index 恒 None（'NoneType' get_current_count，2026-09-27 S 补跑实测）。
    async def _run_queries() -> None:
        await grag.state_manager.query_start()
        try:
            for q in questions:
                t0 = time.time()
                try:
                    resp = await grag.async_query(q["query"])
                except Exception as e:  # noqa: BLE001
                    print(f"[query fail] {q['query'][:30]}: {e}", flush=True)
                    continue
                dt = time.time() - t0
                # chunks 在 resp.context.chunks 的 (chunk, score) 元组里——
                # 直接取 resp.chunks 恒空（2026-09-27 S 全零的根因）
                pairs = getattr(getattr(resp, "context", None), "chunks", None) or []
                hits, rank = [], 0
                seen = set()
                for c, _sc in pairs:
                    did = doc_of(c)
                    if not did or did in seen:
                        continue
                    seen.add(did)
                    rank += 1
                    hits.append({"rank": rank, "doc_id": did,
                                 "chunk_id": (lambda v: int(v) if isinstance(v, (int, float)) else str(v))(getattr(c, "id", None)),
                                 "score": float(_sc) if _sc is not None else None})
                per_query.append({"query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
                                  "hits": hits, "latency_s": round(dt, 3)})
                print(f"  [q {len(per_query)}/{len(questions)}] {dt:.1f}s", flush=True)
        finally:
            await grag.state_manager.query_done()

    asyncio.run(_run_queries())
    write_predictions("fastgraphrag", args.slice, args.set,
                      {"wall_s": round(wall), "note": f"llm={env['LLM_MODEL']}, "
                       f"emb={env['EMBEDDING_MODEL']}；图抽取+PPR 查询；doc 级去重命中"},
                      per_query, len(docs))


if __name__ == "__main__":
    main()
