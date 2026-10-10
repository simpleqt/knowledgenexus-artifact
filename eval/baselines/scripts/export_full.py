#!/usr/bin/env python3
"""导出 ne-journal 全库 chunks（规模退化曲线实验用）：data/full_chunks.jsonl。"""
from __future__ import annotations

import json
import os
import ssl
import sys
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
KB = "21bcbed8ea6b4a5bbf55f3e3e9988c1f"


def env_password() -> str:
    for line in open(os.path.join(REPO, "deploy", ".env"), encoding="utf-8"):
        if line.startswith("OPENSEARCH_ADMIN_PASSWORD="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no password")


CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
import base64
AUTH = "Basic " + base64.b64encode(f"admin:{env_password()}".encode()).decode()


def os_post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"https://127.0.0.1:19200{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": AUTH})
    with urllib.request.urlopen(req, context=CTX, timeout=120) as r:  # noqa: S310
        return json.loads(r.read())


def main() -> None:
    out = os.path.join(REPO, "eval", "baselines", "data", "full_chunks.jsonl")
    n, after = 0, None
    with open(out, "w", encoding="utf-8") as f:
        while True:
            body: dict = {"size": 2000,
                          "query": {"term": {"kb_id": KB}},
                          "sort": [{"doc_id": "asc"}, {"local_position": "asc"}],
                          "_source": ["chunk_id", "doc_id", "local_position",
                                      "content_text", "title_text"]}
            if after:
                body["search_after"] = after
            r = os_post("/knc-chunk/_search", body)
            batch = r["hits"]["hits"]
            for h in batch:
                s = h["_source"]
                f.write(json.dumps({k: s.get(k) for k in
                                    ("chunk_id", "doc_id", "content_text", "title_text")},
                                   ensure_ascii=False) + "\n")
            n += len(batch)
            if n % 20000 < 2000:
                print(f"  {n} 块...", flush=True)
            if len(batch) < 2000:
                break
            after = [batch[-1]["sort"][0], batch[-1]["sort"][1]]
    print(f"完成：{n} 块 → {out}")


if __name__ == "__main__":
    main()
