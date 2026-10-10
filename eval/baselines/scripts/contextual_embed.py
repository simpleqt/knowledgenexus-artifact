#!/usr/bin/env python3
"""O13 上下文嵌入（contextual retrieval 零 LLM 变体）：块向量 = 标题前拼。

Anthropic contextual retrieval（失败率 -35~49%）的零 LLM 变体：嵌入文本从裸
content_text 改为 "{文档标题}\\n{content_text}"——块语义获得文档语境，直击
漏检归因的"票弱"层（单通道弱票因块语义缺语境）。新向量写独立场
chunk_vector_ctx（非破坏），服务侧 KNC_DENSE_FIELD=chunk_vector_ctx 切换。
用法（测试栈）：见本目录 README；幂等（_ctx_done 标记）。
"""
from __future__ import annotations

import json
import os
import ssl
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, REPO  # noqa: E402

KB = os.environ.get("CTX_KB", "0cf04665c3bd427281976c3502eb4d99")
OS_URL = os.environ.get("OS_URL", "https://127.0.0.1:29200")
OS_USER = os.environ.get("OS_USER", "admin")
OS_PASSWORD = os.environ.get("OS_PASSWORD", "KncTest#2026")
FIELD = "chunk_vector_ctx"
BATCH = 32


def os_req(path: str, body: dict | None = None, method: str = "GET") -> dict:
    import base64
    req = urllib.request.Request(f"{OS_URL}{path}",
                                 data=json.dumps(body).encode() if body else None,
                                 method=method,
                                 headers={"Content-Type": "application/json"})
    req.add_header("Authorization", "Basic " + base64.b64encode(
        f"{OS_USER}:{OS_PASSWORD}".encode()).decode())
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=300, context=ctx) as r:  # noqa: S310
        return json.loads(r.read())


def embed(env: dict, texts: list[str]) -> list[list[float]]:
    body = json.dumps({"model": env["EMBEDDING_MODEL"], "input": texts}).encode()
    req = urllib.request.Request(env["EMBEDDING_API_BASE"].rstrip("/") + "/embeddings",
                                 data=body, headers={
                                     "Content-Type": "application/json",
                                     "Authorization": f"Bearer {env['EMBEDDING_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310
        data = json.loads(r.read())["data"]
    return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]


def main() -> None:
    env = {}
    for line in open(os.path.join(REPO, "deploy", ".env.services"), encoding="utf-8"):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            env[k] = v
    # 测试栈 doc_id → 语料标题
    idmap = json.load(open(os.path.join(DATA, "teststack_ne", "idmap.json")))
    test2stable = {v: k for k, v in idmap.items()}
    titles = {}
    for line in open(os.path.join(DATA, "corpus_docs.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        titles[d["doc_id"]] = d["title"]

    # scroll 拉块
    n_done = 0
    t0 = time.time()
    scroll_id = None
    while True:
        if scroll_id is None:
            resp = os_req("/knc-chunk-v1/_search?scroll=5m", {
                "size": 500,
                "query": {"bool": {"filter": [{"term": {"kb_id": KB}}],
                                    "must_not": [{"exists": {"field": FIELD}}]}},
                "_source": ["chunk_id", "doc_id", "content_text"]})
        else:
            resp = os_req("/_search/scroll", {"scroll": "5m", "scroll_id": scroll_id})
        scroll_id = resp.get("_scroll_id", scroll_id)
        hits = resp["hits"]["hits"]
        if not hits:
            break
        for i in range(0, len(hits), BATCH):
            part = hits[i:i + BATCH]
            texts = []
            for h in part:
                s = h["_source"]
                title = titles.get(test2stable.get(s["doc_id"], ""), "")
                texts.append((title + "\n" + s["content_text"])[:12000])
            vecs = embed(env, texts)
            lines = []
            for h, v in zip(part, vecs):
                lines.append(json.dumps({"update": {"_index": "knc-chunk-v1",
                                                    "_id": h["_id"]}}))
                lines.append(json.dumps({"doc": {FIELD: v}}))
            os_req("/_bulk", None, method="POST") if False else None
            # bulk 走原始 ndjson
            import base64
            req = urllib.request.Request(f"{OS_URL}/_bulk",
                                         data=("\n".join(lines) + "\n").encode(),
                                         method="POST",
                                         headers={"Content-Type": "application/x-ndjson"})
            req.add_header("Authorization", "Basic " + base64.b64encode(
                f"{OS_USER}:{OS_PASSWORD}".encode()).decode())
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(req, timeout=300, context=ctx) as r:  # noqa: S310
                bulk_resp = json.loads(r.read())
            if bulk_resp.get("errors"):
                errs = [it for it in bulk_resp.get("items", []) if "error" in it.get("update", {})]
                raise RuntimeError(f"bulk 写入 {len(errs)} 条失败: {errs[0]}")  # 首错即熔断
            n_done += len(part)
        if n_done % 5000 < 500:
            print(f"  {n_done} 块（{time.time()-t0:.0f}s）", flush=True)
    if scroll_id:
        os_req("/_search/scroll", {"scroll": "1m", "scroll_id": scroll_id}, method="DELETE")
    print(f"O13 上下文嵌入完成：{n_done} 块 → {FIELD}（{time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()
