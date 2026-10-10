#!/usr/bin/env python3
"""6.6 深度端点配对重测（2026-09-29，B88 修复后系统）：生产存储 scoped 双臂。

口径：L 切片 300 篇（与论文 6.3～6.6 同语料纪律），进程内直调 _retrieve
（生产只读），base=正常通道 top20；deep=超采 50 + rerank + ppr（修复后
组件全）→ 切片内前 20。命中 doc_id 即生产主键=金标空间（无翻译）。
产物：runs/deepfix-{base,deep}_{set}_L.json
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  .venv/bin/python scoped_prod_deep.py --sets nea,ner'
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO, load_questions, load_slice_ids, write_predictions  # noqa: E402

KB = "21bcbed8ea6b4a5bbf55f3e3e9988c1f"
PROFILE = ["lexical", "dense", "graph", "synq"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="nea,ner")
    args = ap.parse_args()
    slice_ids = load_slice_ids("L")

    sys.path.insert(0, os.path.join(REPO, "services", "retrieval-api"))
    import app.main as M  # noqa: E402

    M.cache.admit = lambda *a, **k: (True, "")
    M.cache.session_admit = lambda *a, **k: (True, 0)
    M.cache.singleflight_lock = lambda *a, **k: True
    M.cache.singleflight_release = lambda *a, **k: None
    M.cache.result_get = lambda *a, **k: None
    M.cache.record_latency_ms = lambda *a, **k: None
    M.cache.bump_metric = lambda *a, **k: None
    M.cache.acl_version = lambda *a, **k: 0
    os.environ.pop("RERANK_DISABLED", None)   # deep 语义

    ARMS = {
        "base": {"top_k": 20, "channels": PROFILE + ["docvec", "cooc"]},
        "deep": {"top_k": 50, "channels": PROFILE + ["docvec", "cooc",
                                                     "rerank", "ppr"]},
    }
    for set_name in args.sets.split(","):
        if not set_name:
            continue
        questions = load_questions(set_name)
        for arm, cfg in ARMS.items():
            per_query, oos_total = [], 0
            for qi, q in enumerate(questions):
                req = M.RetrieveRequest.model_construct(
                    kb_id=KB, kb_ids=None, query=q["query"], top_k=cfg["top_k"],
                    channels=cfg["channels"], doc_ids=list(slice_ids),
                    strategy="A", session_id=None,
                    tenant_id="public", acl_scopes=["public"])
                deadline = time.monotonic() + 150.0
                t0 = time.time()
                resp = M._retrieve(req, [KB], deadline,
                                   lambda: time.monotonic() < deadline,
                                   f"deepfix-{arm}")
                dt = time.time() - t0
                sliceset = set(slice_ids)
                in_slice = [c for c in resp.chunks if c.get("doc_id") in sliceset][:20]
                oos_total += sum(1 for c in resp.chunks
                                 if c.get("doc_id") not in sliceset)
                per_query.append({
                    "query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
                    "hits": [{"rank": i + 1, "doc_id": c.get("doc_id"),
                              "chunk_id": c.get("chunk_id"),
                              "score": c.get("_score")}
                             for i, c in enumerate(in_slice)],
                    "latency_s": round(dt, 3)})
                if (qi + 1) % 15 == 0:
                    print(f"[{arm}/{set_name} {qi+1}/{len(questions)}]", flush=True)
            write_predictions(f"deepfix-{arm}", "L", set_name,
                              {"wall_s": 0,
                               "note": f"生产存储 scoped L 切片（进程内只读）；"
                                       f"{arm}；oos={oos_total}"},
                              per_query, len(slice_ids))


if __name__ == "__main__":
    main()
