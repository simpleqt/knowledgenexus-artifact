#!/usr/bin/env python3
"""KNC 同语料（切片作用域）口径：进程内直调 retrieval-api 的 _retrieve 规划器。

- 显式 strategy=A + doc_ids=切片清单（model_construct 绕过 API 层 max_length=50），
  通道/融合/门控与生产路径同码——仅 stub Redis 准入/缓存旁路（不影响排序，
  也因此不读结果缓存，无脏缓存问题）
- top_k=60 取前 20 个切片内命中：cooc 第三票候选集不带 doc_ids、库外命中占坑
  由过滤吸收（oos 计数如实上报）
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  .venv/bin/python eval/baselines/scripts/run_knc_scoped.py --slice L --set nea'
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import NE_JOURNAL_KB, REPO, load_questions, load_slice_ids, write_predictions

sys.path.insert(0, os.path.join(REPO, "services", "retrieval-api"))

import app.main as M  # noqa: E402  (导入即建 FastAPI app，不启动 server)

# Redis 旁路 stub（准入/单飞/结果缓存/指标——均与排序无关）
M.cache.admit = lambda *a, **k: (True, "")
M.cache.session_admit = lambda *a, **k: (True, 0)
M.cache.singleflight_lock = lambda *a, **k: True
M.cache.singleflight_release = lambda *a, **k: None
M.cache.result_get = lambda *a, **k: None
M.cache.record_latency_ms = lambda *a, **k: None
M.cache.bump_metric = lambda *a, **k: None
M.cache.acl_version = lambda *a, **k: 0


def scoped_retrieve(query: str, doc_ids: list[str], top_k: int = 20):
    req = M.RetrieveRequest.model_construct(
        kb_id=NE_JOURNAL_KB, kb_ids=None, query=query, top_k=60, channels=None,
        doc_ids=list(doc_ids), strategy="A", session_id=None,
        tenant_id="public", acl_scopes=["public"])
    deadline = time.monotonic() + 120.0
    resp = M._retrieve(req, [NE_JOURNAL_KB], deadline,
                       lambda: time.monotonic() < deadline, "knc-scoped-eval")
    sliceset = set(doc_ids)
    all_hits = resp.chunks
    in_slice = [c for c in all_hits if c.get("doc_id") in sliceset][:top_k]
    oos = sum(1 for c in all_hits if c.get("doc_id") not in sliceset)
    return in_slice, oos, resp.channels, resp.strategy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="L", choices=["L", "S"])
    ap.add_argument("--set", required=True)
    args = ap.parse_args()
    slice_ids = load_slice_ids(args.slice)
    questions = load_questions(args.set)

    per_query, oos_total = [], 0
    for q in questions:
        t0 = time.time()
        hits, oos, channels, strategy = scoped_retrieve(q["query"], slice_ids)
        dt = time.time() - t0
        oos_total += oos
        per_query.append({
            "query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
            "hits": [{"rank": i + 1, "doc_id": h["doc_id"], "chunk_id": h.get("chunk_id"),
                      "score": h.get("_score")} for i, h in enumerate(hits)],
            "latency_s": round(dt, 3), "channels": channels,
            "out_of_slice_hits": oos, "strategy": strategy})
    ch_mode = max({tuple(p["channels"]) for p in per_query},
                  key=lambda c: sum(1 for p in per_query if tuple(p["channels"]) == c))
    write_predictions("knc-scoped", args.slice,
                      os.path.basename(args.set).replace(".jsonl", ""),
                      {"wall_s": 0, "note": f"strategy=A + doc_ids={len(slice_ids)} 篇；"
                       f"top60→切片内前20；oos命中合计 {oos_total}；"
                       f"常见通道 {list(ch_mode)}"},
                      per_query, len(slice_ids))


if __name__ == "__main__":
    main()
