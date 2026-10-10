#!/usr/bin/env python3
"""从 OpenSearch 导出基线对比语料切片。

切片构成：ner(ne_journal_realq) + nea(ne_journal_assoc_q48) 的 gold 文档并集
+ 固定随机种子抽取的补充文档，合计 N_DOCS 篇（ne-journal 库）。

产物（eval/baselines/data/）：
  slice_doc_ids.txt     每行一个 doc_id（切片成员清单，排序稳定）
  corpus_chunks.jsonl   每行一 chunk：chunk_id/doc_id/local_position/content_text/title_text/block_type
  corpus_docs.jsonl     每行一 doc：doc_id/title/text（按 local_position 拼接全文）
用法：.venv/bin/python export_slice.py   （读 deploy/.env 的 OPENSEARCH_ADMIN_PASSWORD）
"""
from __future__ import annotations

import json
import os
import random
import ssl
import sys
import urllib.error
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(REPO, "eval", "baselines", "data")
KB = "21bcbed8ea6b4a5bbf55f3e3e9988c1f"          # ne-journal
N_DOCS = 300
SEED = 20260923
SETS = ["ne_journal_realq.jsonl", "ne_journal_assoc_q48.jsonl"]


def env_password() -> str:
    for line in open(os.path.join(REPO, "deploy", ".env"), encoding="utf-8"):
        line = line.strip()
        if line.startswith("OPENSEARCH_ADMIN_PASSWORD="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("deploy/.env 中未找到 OPENSEARCH_ADMIN_PASSWORD")


CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
PW = env_password()
import base64
AUTH = "Basic " + base64.b64encode(f"admin:{PW}".encode()).decode()


def os_post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"https://127.0.0.1:19200{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": AUTH})
    with urllib.request.urlopen(req, context=CTX, timeout=120) as r:  # noqa: S310
        return json.loads(r.read())


def main() -> None:
    os.makedirs(DATA, exist_ok=True)
    gold: set[str] = set()
    for s in SETS:
        for line in open(os.path.join(REPO, "eval", "harness", s), encoding="utf-8"):
            if line.strip():
                gold.update(json.loads(line)["gold_doc_ids"])
    print(f"gold 并集: {len(gold)} 篇")

    # 补充文档抽样池：只取实际有 chunk 的文档（knc-doc 登记里有一部分零块文档）
    r = os_post("/knc-chunk/_search", {"size": 0, "track_total_hits": True,
                                       "query": {"term": {"kb_id": KB}},
                                       "aggs": {"bydoc": {"terms": {"field": "doc_id",
                                                                    "size": 5000}}}})
    docs = [b["key"] for b in r["aggregations"]["bydoc"]["buckets"]]
    print(f"库内有 chunk 文档: {len(docs)} 篇")
    missing = gold - set(docs)
    if missing:
        sys.exit(f"gold 文档缺失 {len(missing)} 篇: {sorted(missing)[:3]}")

    rng = random.Random(SEED)
    pool = sorted(set(docs) - gold)
    n_fill = N_DOCS - len(gold)
    fillers = rng.sample(pool, n_fill)
    slice_ids = sorted(gold | set(fillers))

    # 分页拉全量 chunks（terms 300 个 + search_after）
    hits: list[dict] = []
    after = None
    while True:
        body: dict = {"size": 1000,
                      "query": {"bool": {"filter": [{"term": {"kb_id": KB}},
                                                    {"terms": {"doc_id": slice_ids}}]}},
                      "sort": [{"doc_id": "asc"}, {"local_position": "asc"}],
                      "_source": ["chunk_id", "doc_id", "local_position", "content_text",
                                  "title_text", "block_type"]}
        if after:
            body["search_after"] = after
        r = os_post("/knc-chunk/_search", body)
        batch = r["hits"]["hits"]
        hits.extend(batch)
        if len(batch) < 1000:
            break
        after = [batch[-1]["sort"][0], batch[-1]["sort"][1]]

    chunks = [h["_source"] for h in hits]
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)
    print(f"切片 {len(by_doc)} 篇 / {len(chunks)} 块 / 平均 {len(chunks)/len(by_doc):.1f} 块每篇")
    assert set(by_doc) == set(slice_ids), "切片成员与拉取结果不一致"

    with open(os.path.join(DATA, "slice_doc_ids.txt"), "w") as f:
        f.write("\n".join(slice_ids) + "\n")
    with open(os.path.join(DATA, "corpus_chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    total_chars = 0
    with open(os.path.join(DATA, "corpus_docs.jsonl"), "w", encoding="utf-8") as f:
        for d in slice_ids:
            cs = sorted(by_doc[d], key=lambda x: x["local_position"])
            text = "\n\n".join(c["content_text"] for c in cs)
            total_chars += len(text)
            f.write(json.dumps({"doc_id": d, "title": cs[0].get("title_text") or "",
                                "text": text}, ensure_ascii=False) + "\n")
    print(f"全文总字符: {total_chars/1e6:.1f}M")


if __name__ == "__main__":
    main()
