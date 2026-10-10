#!/usr/bin/env python3
"""holdout 题集草案生成器（LLM 起草，**非人工标注**——正式 holdout 须双人标注+manifest 哈希）。

从库内文档按块数加权抽题：每文档给 LLM 章节+内容摘要 → 生成 n 道可由该文档回答的
检索问题（gold_doc_ids=[该文档]）。输出 jsonl 供 retrieval_eval.py 使用。
用法：
  LLM_API_BASE=... LLM_API_KEY=... LLM_MODEL=... \
  python gen_holdout_draft.py --kb $KB --total 36 --out hydro_holdout_draft36.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

_SYSTEM = ("你是检索评测题集设计器。为给定文档生成用户真实会问的检索问题（中文），"
           "问题必须能且仅能由该文档内容回答，覆盖不同章节/主题，不要出宽泛到任何文档"
           "都能答的题。只输出 JSON。")


def llm(prompt: str) -> str:
    base = os.environ["LLM_API_BASE"].rstrip("/")
    payload = {"model": os.environ["LLM_MODEL"], "max_tokens": 4096, "temperature": 0.4,
               "messages": [{"role": "system", "content": _SYSTEM},
                            {"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ['LLM_API_KEY']}"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def load_docs(api_doc: str, kb: str) -> list[dict]:
    """经检索服务题录面拉文档清单（复用既有服务，不直连库）。"""
    req = urllib.request.Request(api_doc, data=json.dumps({"kb_id": kb, "size": 100}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return json.loads(resp.read())["documents"]


def doc_chunks(api_retrieve: str, kb: str, doc_id: str, title: str) -> list[dict]:
    """按标题词法召回后客户端过滤 doc_id（API 无 doc 精确过滤面，标题足够区分）。"""
    body = {"kb_id": kb, "query": title, "top_k": 50, "channels": ["lexical"]}
    req = urllib.request.Request(api_retrieve, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        return [c for c in json.loads(resp.read())["chunks"] if c["doc_id"] == doc_id]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--total", type=int, default=36)
    ap.add_argument("--out", required=True)
    ap.add_argument("--retrieve-api", default="http://127.0.0.1:18121/api/retrieve")
    ap.add_argument("--doc-api", default="http://127.0.0.1:18121/api/documents")
    args = ap.parse_args()

    docs = load_docs(args.doc_api, args.kb)
    sizes = {}
    for d in docs:
        sizes[d["doc_id"]] = len(doc_chunks(args.retrieve_api, args.kb,
                                            d["doc_id"], d["title"]))
    total_chunks = sum(sizes.values()) or 1
    quota = {d["doc_id"]: max(1, round(args.total * sizes[d["doc_id"]] / total_chunks)) for d in docs}
    # 截到 total：从配额最大的文档里减
    over = sum(quota.values()) - args.total
    for did in sorted(quota, key=lambda k: -quota[k]):
        while over > 0 and quota[did] > 1:
            quota[did] -= 1
            over -= 1

    rows = []
    for d in docs:
        did, n = d["doc_id"], quota[d["doc_id"]]
        chunks = doc_chunks(args.retrieve_api, args.kb, did, d["title"])
        if not chunks:
            print(f"WARN: no chunks for {d['title'][:30]}")
            continue
        text = "\n".join((c.get("title_text") or "") + "：" + c["content_text"][:400]
                         for c in chunks[:40])[:8000]
        raw = llm(f"文档标题：{d['title']}\n以下为文档章节与内容摘录：\n{text}\n\n"
                  f"生成 {n} 道针对该文档的检索问题。"
                  '只输出 JSON 数组，元素为字符串。')
        try:
            qs = json.loads(raw[raw.index("["):raw.rindex("]") + 1])
        except (ValueError, json.JSONDecodeError):
            print(f"WARN: parse failed for {d['title'][:30]}")
            continue
        for q in qs[:n]:
            if isinstance(q, str) and 6 <= len(q.strip()) <= 200:
                rows.append({"query": q.strip(), "gold_doc_ids": [did],
                             "source_doc_title": d["title"]})
        print(f"{d['title'][:24]}: {len([r for r in rows if r['gold_doc_ids'] == [did]])}/{n}")
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"written {len(rows)} questions -> {args.out}")


if __name__ == "__main__":
    main()
