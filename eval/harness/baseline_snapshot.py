#!/usr/bin/env python3
"""基线快照（B13 配套）：144 题审校底稿过当前检索，出总览+按题型/难度断口。

用途：①审校人参考（哪些题当前系统已能答/答不了——难例保留价值判断依据）；
②基线冻结时的「冻结时刻系统状态」存档（同题集可跨版本对比）。
只读检索 API，不耗 LLM。输出 json 入 eval/harness/results/。
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import urllib.request


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eval/harness/annotation/baseline_draft_v1.jsonl")
    ap.add_argument("--review-csv", default="eval/harness/annotation/baseline_review_v1.csv")
    ap.add_argument("--kb", required=True)
    ap.add_argument("--api", default="http://127.0.0.1:18121/api/retrieve")
    ap.add_argument("--channels", default="lexical,dense")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env_path = os.path.join(os.path.dirname(__file__), "..", "..", "deploy", ".env")
    auth = ""
    try:
        u = p = ""
        for line in open(env_path):
            if line.startswith("KNC_AUTH_USER="):
                u = line.strip().split("=", 1)[1]
            if line.startswith("KNC_AUTH_PASSWORD="):
                p = line.strip().split("=", 1)[1]
        auth = "Basic " + base64.b64encode(f"{u}:{p}".encode()).decode()
    except OSError:
        pass

    meta = {}
    if os.path.exists(args.review_csv):
        for r in csv.DictReader(open(args.review_csv, encoding="utf-8-sig")):
            meta[r["query"]] = {"type": r["type"], "hard": r["hard"] == "1"}

    rows = [json.loads(l) for l in open(args.dataset) if l.strip()]
    per, ranks = [], []
    for row in rows:
        gold = set(row["gold_doc_ids"])
        body = json.dumps({"kb_id": args.kb, "query": row["query"], "top_k": 20,
                           "channels": args.channels.split(",")}).encode()
        req = urllib.request.Request(args.api, data=body,
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": auth} if auth else {})})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                chunks = json.loads(r.read())["chunks"]
        except Exception:
            chunks = []
        docs = []
        for c in chunks:
            if c["doc_id"] not in docs:
                docs.append(c["doc_id"])
        rank = next((i + 1 for i, d in enumerate(docs) if d in gold), None)
        ranks.append(rank)
        m = meta.get(row["query"], {})
        per.append({"query": row["query"][:60], "type": m.get("type", "?"),
                    "hard": m.get("hard", False), "rank": rank})

    def _agg(items):
        n = len(items) or 1
        return {"n": len(items),
                "recall@5": round(sum(1 for x in items if x["rank"] and x["rank"] <= 5) / n, 4),
                "recall@10": round(sum(1 for x in items if x["rank"] and x["rank"] <= 10) / n, 4),
                "recall@20": round(sum(1 for x in items if x["rank"]) / n, 4),
                "mrr@10": round(sum(1 / x["rank"] if x["rank"] and x["rank"] <= 10 else 0
                                    for x in items) / n, 4)}

    result = {"kb": args.kb, "channels": args.channels, "queries": len(per),
              "overall": _agg(per),
              "by_type": {t: _agg([x for x in per if x["type"] == t])
                          for t in {x["type"] for x in per}},
              "hard(预检难例)": _agg([x for x in per if x["hard"]]),
              "easy(预检正常)": _agg([x for x in per if not x["hard"]])}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("→", args.out)


if __name__ == "__main__":
    main()
