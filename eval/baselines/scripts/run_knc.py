#!/usr/bin/env python3
"""KNC 生产系统基线（POST /api/retrieve，全库 ne-journal，默认通道 profile）。

注意口径不对称：KNC 在全库（4850 篇）检索，外部基线在切片上检索——
该不对称对 KNC 不利（全库干扰池更大），详见 eval/baselines/README.md。
用法：.venv/bin/python run_knc.py --set nea   （题集名 = data/ 下或 eval/harness/ 下）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import NE_JOURNAL_KB, REPO, load_questions, write_predictions

API_DEFAULT = "http://127.0.0.1:18121/api/retrieve"   # 回环免认证（authn.py 白名单）
TOP_K = 10   # 与 B77-80 存档 harness 原命令 --ks 10 对齐（RRF 候选池大小影响 R@10）


def retrieve(query: str, api: str) -> list[dict]:
    body = json.dumps({"kb_id": NE_JOURNAL_KB, "query": query, "top_k": TOP_K}).encode()
    req = urllib.request.Request(api, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310
        return json.loads(resp.read())["chunks"]


SHORT = {"nea": "ne_journal_assoc_q48.jsonl", "ner": "ne_journal_realq.jsonl"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True,
                    help="题集短码 nea/ner/nea_S，或 data/、eval/harness/ 下文件名")
    ap.add_argument("--base-url", default=API_DEFAULT,
                    help="检索端点（20261003_knn_efsearch：只读副本臂注入 env 用）")
    ap.add_argument("--tag", default="",
                    help="系统名后缀（knc<tag>），区分消融臂产物文件")
    args = ap.parse_args()
    name = SHORT.get(args.set, args.set if args.set.endswith(".jsonl") else args.set + ".jsonl")
    cand = [os.path.join(REPO, "eval", "baselines", "data", name),
            os.path.join(REPO, "eval", "harness", name)]
    p = next((c for c in cand if os.path.exists(c)), None)
    if p is None:
        raise SystemExit(f"题集未找到: {name}（找过 data/ 与 eval/harness/）")
    questions = load_questions(p)

    per_query = []
    for q in questions:
        t0 = time.time()
        hits_raw = retrieve(q["query"], args.base_url)
        dt = time.time() - t0
        hits = [{"rank": i + 1, "doc_id": h["doc_id"], "chunk_id": h.get("chunk_id"),
                 "score": h.get("_score")}
                for i, h in enumerate(hits_raw)]
        per_query.append({"query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
                          "hits": hits, "latency_s": round(dt, 3)})
    write_predictions(f"knc{args.tag}", "full-kb", args.set,
                      {"wall_s": 0, "note": f"存量索引；全库检索（非切片）；api={args.base_url}"},
                      per_query, 4850)


if __name__ == "__main__":
    main()
