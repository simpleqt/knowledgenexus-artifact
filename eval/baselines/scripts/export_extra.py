#!/usr/bin/env python3
"""导出 ne-journal 全库剩余文档（排除 300 篇既有切片）→ 测试栈扩容语料。

产物：data/corpus_docs_extra.jsonl（doc_id/title/text，运行时资产不入 Git）
用途：测试栈 ne-slice-test KB 扩容为生产全库镜像（靶题扩产的出题池与干扰面）。
用法：.venv/bin/python export_extra.py   （读 deploy/.env 的 OPENSEARCH_ADMIN_PASSWORD）
"""
from __future__ import annotations

import base64
import json
import os
import ssl
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(REPO, "eval", "baselines", "data")
KB = "21bcbed8ea6b4a5bbf55f3e3e9988c1f"


def env_password() -> str:
    for line in open(os.path.join(REPO, "deploy", ".env"), encoding="utf-8"):
        if line.startswith("OPENSEARCH_ADMIN_PASSWORD="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no password")


CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
AUTH = "Basic " + base64.b64encode(f"admin:{env_password()}".encode()).decode()


def os_post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"https://127.0.0.1:19200{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": AUTH})
    with urllib.request.urlopen(req, context=CTX, timeout=180) as r:  # noqa: S310
        return json.loads(r.read())


def main() -> None:
    slice_ids = {l.strip() for l in open(os.path.join(DATA, "slice_doc_ids.txt"))
                 if l.strip()}
    r = os_post("/knc-chunk/_search", {"size": 0, "track_total_hits": True,
                                       "query": {"term": {"kb_id": KB}},
                                       "aggs": {"bydoc": {"terms": {"field": "doc_id",
                                                                    "size": 5000}}}})
    all_docs = [b["key"] for b in r["aggregations"]["bydoc"]["buckets"]]
    extra = sorted(set(all_docs) - slice_ids)
    print(f"全库 {len(all_docs)} 篇 - 切片 {len(slice_ids)} = 补差 {len(extra)} 篇")

    out = open(os.path.join(DATA, "corpus_docs_extra.jsonl"), "w", encoding="utf-8")
    total = 0
    for i, did in enumerate(extra):
        hits: list[dict] = []
        after = None
        while True:
            body: dict = {"size": 1000,
                          "query": {"bool": {"filter": [{"term": {"kb_id": KB}},
                                                        {"term": {"doc_id": did}}]}},
                          "sort": [{"local_position": "asc"}],
                          "_source": ["local_position", "content_text", "title_text"]}
            if after:
                body["search_after"] = after
            rr = os_post("/knc-chunk/_search", body)
            batch = rr["hits"]["hits"]
            hits.extend(batch)
            if len(batch) < 1000:
                break
            after = batch[-1]["sort"]
        cs = [h["_source"] for h in hits]
        cs.sort(key=lambda x: x["local_position"])
        text = "\n\n".join(c["content_text"] for c in cs)
        total += len(text)
        out.write(json.dumps({"doc_id": did,
                              "title": cs[0].get("title_text") or "",
                              "text": text}, ensure_ascii=False) + "\n")
        if (i + 1) % 200 == 0:
            out.flush()
            print(f"  {i+1}/{len(extra)}  chars={total/1e6:.0f}M", flush=True)
    out.close()
    print(f"完成：{len(extra)} 篇 / {total/1e6:.0f}M 字符 → corpus_docs_extra.jsonl")


if __name__ == "__main__":
    main()
