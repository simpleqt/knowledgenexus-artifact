#!/usr/bin/env python3
"""路由层评测（§7 路由层指标 / M3.3）：KB Router 的 Routing Oracle Gap。

口径：
- route_recall@K：gold 库进入 Router 路由集（top-K）的题比例——gold 库在路由层
  **不得被系统性裁掉**（方案 §7 门槛语义）；
- strong_single_precision：强单库判定且恰为 gold 库的比例（在判定为强单库的题上）；
- oracle_gap：联邦全库检索（oracle，不路由）与路由后检索的 recall@5 差——
  路由裁掉的召回损失（≥0 为底线，负值=路由反而更准的噪声）。

题集（jsonl）：{"query": str, "gold_kb": hex32}
用法：python eval/harness/routing_eval.py --dataset routing_cross_kb.jsonl [--k 3]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--api", default="http://127.0.0.1:18121")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    auth = ""
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", "deploy", ".env")
    try:
        u = p = ""
        for line in open(env_path):
            if line.startswith("KNC_AUTH_USER="):
                u = line.strip().split("=", 1)[1]
            if line.startswith("KNC_AUTH_PASSWORD="):
                p = line.strip().split("=", 1)[1]
        if u and p:
            auth = "Basic " + base64.b64encode(f"{u}:{p}".encode()).decode()
    except OSError:
        pass

    def post(path: str, body: dict) -> dict:
        req = urllib.request.Request(args.api + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": auth} if auth else {})})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    rows = [json.loads(l) for l in open(args.dataset) if l.strip()]
    route_hits, strong_total, strong_correct = [], 0, 0
    gap_recall = []
    for row in rows:
        gold = row["gold_kb"].lower()
        rt = post("/api/route", {"query": row["query"], "top_k": args.k})
        routed = [r["kb_id"] for r in rt["routed"]]
        hit = gold in routed
        route_hits.append(int(hit))
        if rt.get("strong_single"):
            strong_total += 1
            strong_correct += int(routed[:1] == [gold])
        # oracle（全库联邦）vs 路由后检索 的 recall@5 差
        all_kbs = ["f7327a9e090644d9b67e5b2bca7bd6a6",
                   "d84f82ab21874c088673fd34c8fb3130",
                   "ea8d5699b9db4b359b4e78a1e692f4ad"]
        oracle = post("/api/retrieve", {"kb_ids": all_kbs, "query": row["query"], "top_k": 5})
        oracle_docs = {c["doc_id"] for c in oracle["chunks"]}
        scoped = post("/api/retrieve", {"kb_ids": routed or [gold], "query": row["query"],
                                        "top_k": 5})
        scoped_docs = {c["doc_id"] for c in scoped["chunks"]}
        # oracle gap 以「路由后命中文档占 oracle 命中的比例」度量（同查询可比）
        gap = len(scoped_docs & oracle_docs) / len(oracle_docs) if oracle_docs else 1.0
        gap_recall.append(gap)
        print(f"  {'✓' if hit else '✗'} gold={gold[:8]} routed={[k[:8] for k in routed]}"
              f" strong={rt.get('strong_single')} overlap={gap:.2f} | {row['query'][:40]}")

    n = len(rows)
    result = {
        "queries": n,
        "route_recall@k": round(sum(route_hits) / n, 4) if n else 0.0,
        "strong_single_precision": round(strong_correct / strong_total, 4) if strong_total else None,
        "strong_single_coverage": round(strong_total / n, 4) if n else 0.0,
        "scoped_vs_oracle_overlap": round(sum(gap_recall) / n, 4) if n else 0.0,
        "k": args.k,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
