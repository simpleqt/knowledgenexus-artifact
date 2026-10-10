#!/usr/bin/env python3
"""通用 LLM-RAG 基线臂（论文一主表对照：能力派通用管线，不享 KNC 领域资产）。

设计 = "一个胜任的通用派会怎么搭"：
  LLM 查询改写（无意图门控）→ BM25(jieba) + dense(Qwen3-Embedding-8B@本地:8012) 混合
  → RRF 融合 → LLM 列表重排 top-60 → top-20

与外部基线组互补：bm25/llamaindex=单通道，hipporag=图构建派（建库 4.1h），
本臂=查询时 LLM 派（零建库、纯查询时算力）。LLM 一律走 CDUT 网关
（qwen3.8-flash-next 主力；qwen3.5-35b 禁用——与入库富化共容量），
并发 8、逐请求超时 120s、重试 ×2。嵌入走本地 :8012（与生产同端点、
MRL 1024 客户端截断+L2——2026-10-06 起 bge-m3@网关换本地 Qwen3-8B，
与 KNC dense 通道同嵌入面，隔离"嵌入模型差异"变量到融合/重排策略）。

用法：.venv/bin/python scripts/run_genrag.py --set nea [--slice L] [--limit 5]
依赖：仅 common.py 契约 + numpy（块向量 28k×1024 进程内暴力余弦——切片级，
     全库禁止，见 README 铁律）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from common import BASE, load_chunks, load_questions, load_slice_ids, write_predictions

GATEWAY = "https://aigateway.cdut.edu.cn/v1"
GATEWAY_KEY = os.environ.get("CDUT_GATEWAY_KEY", "")
LLM_MODEL = os.environ.get("GENRAG_LLM", "qwen3.8-flash-next")
# 嵌入=本地生产端点（用户 2026-10-06 指示：与生产同嵌入服务；GPU4 空闲压力不大）。
# key 从 deploy/.env.services 读（不入库），env GENRAG_EMB_KEY 可覆盖。
EMB_BASE = os.environ.get("GENRAG_EMB_BASE", "http://202.115.138.31:8012/v1")
EMB_MODEL = "Qwen3-Embedding-8B"
EMB_DIM = 1024  # MRL 客户端截断（端点拒 dimensions 参数，见 AGENTS EMBEDDING_MRL_DIM）


def _emb_key() -> str:
    k = os.environ.get("GENRAG_EMB_KEY")
    if k:
        return k
    for line in open(os.path.join(BASE, "..", "..", "deploy", ".env.services"),
                     encoding="utf-8"):
        if line.startswith("EMBEDDING_API_KEY="):
            return line.split("=", 1)[1].strip()
    return ""


CONC = 8
EMB_BATCH = 32
TOPK_FUSE = 60


def _post(path: str, payload: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        f"{GATEWAY}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {GATEWAY_KEY}"}, method="POST")
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as exc:  # noqa: BLE001 网关瞬态重试
            last = exc
            time.sleep(3)
    raise RuntimeError(f"gateway {path} failed: {last}")


def llm_chat(prompt: str, system: str = "", max_tokens: int = 3072) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    r = _post("/chat/completions", {"model": LLM_MODEL, "messages": msgs,
                                    "max_tokens": max_tokens, "temperature": 0.0})
    msg = r["choices"][0]["message"]
    return (msg.get("content") or "").strip()


_EMB_KEY = ""


def _post_emb(payload: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        f"{EMB_BASE}/embeddings", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {_EMB_KEY}"}, method="POST")
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as exc:  # noqa: BLE001 端点瞬态重试
            last = exc
            time.sleep(3)
    raise RuntimeError(f"embed endpoint failed: {last}")


def embed(texts: list[str]) -> list[list[float]]:
    import math
    import threading
    batches = [[t[:8000] for t in texts[i:i + EMB_BATCH]]
               for i in range(0, len(texts), EMB_BATCH)]
    out: list = [None] * len(batches)
    done = [0]
    lock = threading.Lock()

    def one(bi: int) -> None:
        r = _post_emb({"model": EMB_MODEL, "input": batches[bi]})
        vs = []
        for d in r["data"]:
            v = d["embedding"][:EMB_DIM]  # MRL 截断 + L2（与生产 search.py 同口径）
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            vs.append([x / n for x in v])
        out[bi] = vs
        with lock:
            done[0] += 1
            if done[0] % 20 == 0 or done[0] == len(batches):
                print(f"  embed {min(done[0] * EMB_BATCH, len(texts))}/{len(texts)}", flush=True)

    with ThreadPoolExecutor(CONC) as ex:
        list(ex.map(one, range(len(batches))))
    return [v for b in out for v in b]


def rewrite_query(q: str) -> str:
    prompt = ("改写以下学术检索查询为更适合混合检索的查询：保留全部术语与语种，"
              "补全常见缩写全称，不添加查询中未出现的概念。只输出改写后的查询。\n" + q)
    r = llm_chat(prompt, max_tokens=2048)
    return r if 2 < len(r) < 400 else q


def rerank_list(query: str, docs: list[str]) -> list[int]:
    """LLM 列表重排：返回按相关性降序的索引。截断每段到 320 字控制上下文。"""
    listing = "\n".join(f"[{i}] {d[:320]}" for i, d in enumerate(docs))
    prompt = (f"查询：{query}\n\n候选文档段（编号[0]-[{len(docs)-1}]）：\n{listing}\n\n"
              f"按与查询的语义相关性从高到低输出编号，格式：逗号分隔的数字列表，"
              f"如 3,0,17。只输出这一行。")
    r = llm_chat(prompt, max_tokens=3072)
    order = []
    for tok in r.replace(" ", "").replace("\n", "").split(","):
        tok = tok.strip("[]")
        if tok.isdigit() and int(tok) < len(docs) and int(tok) not in order:
            order.append(int(tok))
        if len(order) >= len(docs):
            break
    order.extend(i for i in range(len(docs)) if i not in order)
    return order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="L", choices=["L", "S"])
    ap.add_argument("--set", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if not GATEWAY_KEY:
        raise SystemExit("需要 CDUT_GATEWAY_KEY env（网关 api key）")
    global _EMB_KEY
    _EMB_KEY = _emb_key()

    chunks = load_chunks()
    keep = set(load_slice_ids("L" if args.slice == "L" else "S"))
    chunks = [c for c in chunks if c["doc_id"] in keep]
    print(f"切片 {args.slice}: {len(chunks)} 块")
    questions = load_questions(args.set)
    if args.limit:
        questions = questions[:args.limit]

    # ---- 建库：BM25 词项 + bge-m3 块向量（缓存 indexes/genrag_*.npy）----
    import jieba
    from rank_bm25 import BM25Okapi
    t0 = time.time()
    toks = [[t for t in jieba.lcut((c.get("title_text") or "") + "\n" + c["content_text"].lower()) if t.strip()]
            for c in chunks]
    bm25 = BM25Okapi(toks)

    import numpy as np
    idx_dir = os.path.join(BASE, "indexes")
    os.makedirs(idx_dir, exist_ok=True)
    vec_path = os.path.join(idx_dir, f"genrag_qwen8b_{args.slice}.npy")
    texts = [(c.get("title_text") or "") + "\n" + c["content_text"] for c in chunks]
    if os.path.exists(vec_path):
        mat = np.load(vec_path)
        print(f"向量缓存命中 {mat.shape}")
    else:
        print(f"嵌入 {len(texts)} 块（Qwen3-8B@:8012，并发 {CONC}）...", flush=True)
        vecs = embed(texts)
        mat = np.array(vecs, dtype=np.float32)
        mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        np.save(vec_path, mat)
    build_s = round(time.time() - t0, 1)

    # ---- 查询循环 ----
    def one(qd: dict) -> dict:
        t1 = time.time()
        q = qd["query"]
        rq = rewrite_query(q)
        qv = np.array(embed([rq])[0], dtype=np.float32)
        qv /= (np.linalg.norm(qv) + 1e-9)
        dense = mat @ qv
        sparse = bm25.get_scores([t for t in jieba.lcut(rq.lower()) if t.strip()])
        d_idx = np.argsort(-dense)[:200].tolist()
        s_idx = sorted(range(len(chunks)), key=lambda i: sparse[i], reverse=True)[:200]
        # RRF
        rrf: dict[int, float] = {}
        for rank, i in enumerate(d_idx):
            rrf[i] = rrf.get(i, 0) + 1.0 / (60 + rank + 1)
        for rank, i in enumerate(s_idx):
            rrf[i] = rrf.get(i, 0) + 1.0 / (60 + rank + 1)
        fused = sorted(rrf, key=rrf.get, reverse=True)[:TOPK_FUSE]
        order = rerank_list(q, [texts[i] for i in fused])
        hits = [{"rank": n + 1, "doc_id": chunks[fused[i]]["doc_id"],
                 "chunk_id": chunks[fused[i]].get("chunk_id", ""),
                 "score": round(float(rrf[fused[i]]), 4)}
                for n, i in enumerate(order[:20])]
        return {"query": q, "rewrite": rq, "gold_doc_ids": qd["gold_doc_ids"],
                "hits": hits, "latency_s": round(time.time() - t1, 2)}

    per_query = []
    with ThreadPoolExecutor(CONC) as ex:
        for r in ex.map(one, questions):
            per_query.append(r)
            print(f"  [{len(per_query)}/{len(questions)}] {r['latency_s']}s {r['query'][:24]}", flush=True)

    llm_calls = 2 * len(questions)
    write_predictions("genrag", args.slice, args.set,
                      {"wall_s": round(time.time() - t0, 1),
                       "engine": f"LLM改写({LLM_MODEL}@CDUT)+BM25/Qwen3-8B RRF+LLM重排",
                       "llm_calls": llm_calls,
                       "build_s_if_cold": build_s},
                      per_query, len({c['doc_id'] for c in chunks}))


if __name__ == "__main__":
    main()
