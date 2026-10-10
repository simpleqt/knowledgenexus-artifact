#!/usr/bin/env python3
"""KNC 在基准库上的评测 runner（含通道消融臂）。

- 库 = bench_ingest.py 建的基准库（语料=池，与基线同语料，无需 scoped）
- 返回 doc_id 经 idmap 译回 stable_id（与基线/gold 同一 id 空间）
- --channels 可选（如 lexical / dense / lexical,dense）→ 消融臂；缺省=生产 profile
用法：.venv/bin/python bench_run_knc.py --dataset hotpotqa [--channels lexical]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import BASE, write_predictions

API = "http://127.0.0.1:18121/api/retrieve"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--channels", default=None)
    ap.add_argument("--api", default=API, help="检索基址（测试栈 http://127.0.0.1:28121）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全量；抽样对比用）")
    ap.add_argument("--tag", default="", help="输出 system 名后缀（避免覆盖存档 run）")
    args = ap.parse_args()
    if not args.api.rstrip("/").endswith("/api/retrieve"):  # 基址→完整端点（两种传法兼容）
        args.api = args.api.rstrip("/") + "/api/retrieve"
    d = os.path.join(BASE, "data", "bench", args.dataset)
    kb_hex = json.load(open(os.path.join(d, "kb.json")))["kb_id"]
    idmap: dict[str, str] = json.load(open(os.path.join(d, "idmap.json")))
    rev = {v: k for k, v in idmap.items()}
    questions = [json.loads(l) for l in open(os.path.join(d, "questions.jsonl"), encoding="utf-8") if l.strip()]
    if args.limit > 0:
        questions = questions[:args.limit]

    per_query = []
    for q in questions:
        body = {"kb_id": kb_hex, "query": q["query"], "top_k": 20}
        if args.channels:
            body["channels"] = args.channels.split(",")
        t0 = time.time()
        req = urllib.request.Request(args.api, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:  # noqa: S310
            chunks = json.loads(r.read())["chunks"]
        dt = time.time() - t0
        hits, rank = [], 0
        for c in chunks:
            sid = rev.get(c["doc_id"])
            if sid is None:
                continue
            rank += 1
            hits.append({"rank": rank, "doc_id": sid, "chunk_id": c.get("chunk_id"),
                         "score": c.get("_score")})
        per_query.append({"query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
                          "hits": hits, "latency_s": round(dt, 3)})
    arm = f"-{args.channels.replace(',', '+')}" if args.channels else ""
    write_predictions(f"knc{arm}{args.tag}", "pool", args.dataset,
                      {"wall_s": 0, "note": f"基准库全检索（语料=池）；"
                       f"channels={args.channels or '默认profile'}；top_k=20"
                       + (f"；limit={args.limit}" if args.limit else "")},
                      per_query, len(idmap))


if __name__ == "__main__":
    main()
