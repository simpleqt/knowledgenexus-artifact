#!/usr/bin/env python3
"""LlamaIndex 全库直算旁路（2026-09-29）：流式解析 18GB 持久化向量 JSON，
numpy 精确余弦 top-20——与 SimpleVectorStore 检索语义等价（同向量、同块、
精确 cosine 排序），绕开 llama_index 对 19.6 万节点的小时级对象重建。

产物：runs/llamaindex_hard37_full.json（与官方 runner 同 schema）。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, INDEXES, REPO, load_questions, load_services_env, write_predictions  # noqa: E402

VEC_JSON = os.path.join(INDEXES, "llamaindex_full", "default__vector_store.json")
DIM = 4096


def stream_vectors(path: str):
    """流式抽 (chunk_id, float32[DIM])。格式：\"<hex64>\": [f, f, ...]"""
    pat = re.compile(rb'"([0-9a-f]{64})": \[([^\]]*)\]')
    ids: list[str] = []
    vecs = np.empty((196_618 + 4096, DIM), dtype=np.float32)
    n = 0
    carry = b""
    t0 = time.time()
    with open(path, "rb") as f:
        while True:
            block = f.read(64 << 20)
            if not block:
                break
            data = carry + block
            last_bracket = data.rfind(b"]")
            if last_bracket == -1:
                carry = data
                continue
            consumable = data[:last_bracket + 1]
            carry = data[last_bracket + 1:]
            for m in pat.finditer(consumable):
                cid = m.group(1).decode()
                nums = np.fromstring(m.group(2), dtype=np.float32, sep=",") \
                    if False else np.array(m.group(2).split(b","), dtype=np.float32)
                if len(nums) != DIM:
                    continue
                ids.append(cid)
                vecs[n] = nums
                n += 1
    return ids, vecs[:n]


def embed(env: dict, texts: list[str]) -> np.ndarray:
    import urllib.request
    out = []
    for i in range(0, len(texts), 16):
        body = json.dumps({"model": env["EMBEDDING_MODEL"],
                           "input": texts[i:i + 16]}).encode()
        req = urllib.request.Request(env["EMBEDDING_API_BASE"].rstrip("/") + "/embeddings",
                                     data=body, headers={"Content-Type": "application/json",
                                                         "Authorization": f"Bearer {env['EMBEDDING_API_KEY']}"})
        with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310
            data = json.loads(r.read())["data"]
        out.extend(d["embedding"] for d in sorted(data, key=lambda x: x["index"]))
    return np.asarray(out, dtype=np.float32)


def main() -> None:
    env = load_services_env()
    questions = load_questions(os.path.join(DATA, "hard37.jsonl"))

    t0 = time.time()
    ids, vecs = stream_vectors(VEC_JSON)
    print(f"向量装载：{len(ids)}×{DIM}，{time.time()-t0:.0f}s", flush=True)
    row = {cid: i for i, cid in enumerate(ids)}

    # chunk_id → doc_id（full_chunks）
    c2d = {}
    for l in open(os.path.join(DATA, "full_chunks.jsonl"), encoding="utf-8"):
        if l.strip():
            c = json.loads(l)
            c2d[c["chunk_id"]] = c["doc_id"]

    # 归一化矩阵（cosine）
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vecs / norms

    qtexts = [q["query"] for q in questions]
    qv = embed(env, qtexts)
    qn = np.linalg.norm(qv, axis=1, keepdims=True)
    qn[qn == 0] = 1.0
    qu = qv / qn

    per_query = []
    for q, qvec in zip(questions, qu):
        t1 = time.time()
        sims = unit @ qvec           # (N,)
        top = np.argpartition(-sims, 20)[:20]
        top = top[np.argsort(-sims[top])]
        hits = [{"rank": i + 1, "doc_id": c2d.get(ids[j], "?"),
                 "chunk_id": ids[j], "score": round(float(sims[j]), 4)}
                for i, j in enumerate(top)]
        per_query.append({"query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
                          "hits": hits, "latency_s": round(time.time() - t1, 3)})
    write_predictions("llamaindex", "full", "hard37",
                      {"wall_s": round(time.time() - t0),
                       "note": "复用 2026-09-24 持久化向量（同嵌入模型），"
                               "numpy 精确 cosine=SimpleVectorStore 语义等价"},
                      per_query, len({c2d.get(cid) for cid in ids}))


if __name__ == "__main__":
    main()
