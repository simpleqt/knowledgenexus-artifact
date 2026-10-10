#!/usr/bin/env python3
"""O8 消融前置：ne 300 篇测试片的 publish_year 回填（2026-09-25）。

来源=生产库 doc_bibliography（同一批 NIM-A 文档的题录富化产物，按标题匹配——
211/300 命中、169 篇有年份），写入目标=隔离测试栈 knc-doc 索引（29200）。
只 UPDATE 测试栈；生产库仅只读 SELECT。幂等可重跑。

用法：
  python3 backfill_docvec_prior.py            # 干跑：只报匹配统计
  python3 backfill_docvec_prior.py --apply    # 真写入测试栈 knc-doc
依赖 env：生产 MySQL（MYSQL_PORT=13306 root）+ 测试栈 OS（OS_URL/OS_USER/OS_PASSWORD）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, REPO  # noqa: E402

PROD_KB = "21bcbed8ea6b4a5bbf55f3e3e9988c1f"   # 核电子学·NIM-A（题录富化源）
OS_TEST = os.environ.get("OS_URL", "https://127.0.0.1:29200")


def prod_year_by_title() -> dict[str, int]:
    import knc_engine.db as db
    out = {}
    for r in db.exec(
            "SELECT b.title_orig t, b.publish_year y FROM doc_bibliography b"
            " WHERE b.kb_id=UNHEX(%s) AND b.publish_year IS NOT NULL", (PROD_KB,), fetch=True):
        if r["t"]:
            out[r["t"].strip()] = int(r["y"])
    return out


def os_bulk(updates: list[tuple[str, int]], user: str, pw: str) -> int:
    lines = []
    for doc_id, year in updates:
        lines.append(json.dumps({"update": {"_index": "knc-doc-v1", "_id": doc_id}}))
        lines.append(json.dumps({"doc": {"publish_year": year}}))
    body = ("\n".join(lines) + "\n").encode()
    req = urllib.request.Request(f"{OS_TEST}/_bulk?refresh=true", data=body,
                                 headers={"Content-Type": "application/x-ndjson"})
    import base64
    tok = base64.b64encode(f"{user}:{pw}".encode()).decode()
    req.add_header("Authorization", f"Basic {tok}")
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=120, context=ctx) as r:  # noqa: S310
        resp = json.loads(r.read())
    return sum(1 for it in resp.get("items", []) if not it.get("update", {}).get("error"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    docs = [json.loads(l) for l in
            open(os.path.join(DATA, "corpus_docs.jsonl"), encoding="utf-8") if l.strip()]
    idmap = json.load(open(os.path.join(DATA, "teststack_ne", "idmap.json")))
    years = prod_year_by_title()
    matched, updates = 0, []
    for d in docs:
        y = years.get(d["title"].strip())
        if y is None:
            continue
        matched += 1
        test_did = idmap.get(d["doc_id"])
        if test_did:
            updates.append((test_did, y))
    ys = sorted({y for _, y in updates})
    print(f"生产题录年份匹配：{matched}/300；可回填（在测试片内）：{len(updates)}"
          f"；年份跨度 {ys[0]}–{ys[-1]}" if updates else f"匹配 {matched}/300（无可回填）")
    if args.apply and updates:
        ok = os_bulk(updates, os.environ.get("OS_USER", "admin"),
                     os.environ.get("OS_PASSWORD", ""))
        print(f"测试栈 knc-doc 更新成功 {ok}/{len(updates)}")


if __name__ == "__main__":
    main()
