#!/usr/bin/env python3
"""实验七（⑦）：微调终版 a03 上生产 scoped 深轨——接 +10.8 证据链（R2-M5）。

exp2（zs OneJev vs 融合 +10.8，20261004）只证了零样本臂；审稿指出微调配方
从未上过生产 scoped 轨。本脚本进程内只读（生产 API 纪律，scoped_prod_deep.py
/exp2_prod_gctx.py 同款），四臂同日同环境：base（融合）/ zsjev / a03 / a03+gctx。
重排模型切换在壳层（adapter arm 文件 + :18217 换 serve），本脚本按 --arm 打标。
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  RERANK_API_BASE=http://127.0.0.1:18214/v1 RERANK_API_KEY=local \
  RERANK_MODEL=onejev-4b .venv/bin/python eval/baselines/scripts/exp7_prod_a03.py --arm a03'
产物：runs/p7-<arm>_<set>_L.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO, load_questions, load_slice_ids, write_predictions  # noqa: E402

KB = "21bcbed8ea6b4a5bbf55f3e3e9988c1f"
PROFILE = ["lexical", "dense", "graph", "synq"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    help="base|zs|a03|a03g（重排臂已在壳层就位，仅打标）")
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
    os.environ.pop("RERANK_DISABLED", None)

    deep = ["rerank", "ppr"] if args.arm != "base" else []
    channels = PROFILE + ["docvec", "cooc"] + deep
    os.environ["KNC_RERANK_GRAPH_CTX"] = "1" if args.arm == "a03g" else ""
    top_k = 50 if deep else 20

    for set_name in args.sets.split(","):
        if not set_name:
            continue
        questions = load_questions(set_name)
        per_query, oos_total = [], 0
        for qi, q in enumerate(questions):
            req = M.RetrieveRequest.model_construct(
                kb_id=KB, kb_ids=None, query=q["query"], top_k=top_k,
                channels=channels, doc_ids=list(slice_ids),
                strategy="A", session_id=None,
                tenant_id="public", acl_scopes=["public"])
            deadline = time.monotonic() + 600.0   # 富化舰队期放宽（同日四臂一致）
            t0 = time.time()
            resp = M._retrieve(req, [KB], deadline,
                               lambda: time.monotonic() < deadline, f"exp7-{args.arm}")
            dt = time.time() - t0
            sliceset = set(slice_ids)
            in_slice = [c for c in resp.chunks if c.get("doc_id") in sliceset][:20]
            oos_total += sum(1 for c in resp.chunks if c.get("doc_id") not in sliceset)
            per_query.append({
                "query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
                "hits": [{"rank": i + 1, "doc_id": c.get("doc_id"),
                          "chunk_id": c.get("chunk_id"),
                          "score": c.get("_score"), "rerank": c.get("_rerank_score")}
                         for i, c in enumerate(in_slice)],
                "latency_s": round(dt, 3)})
            if (qi + 1) % 15 == 0:
                print(f"[{args.arm}/{set_name} {qi+1}/{len(questions)}] p50~"
                      f"{sorted(p['latency_s'] for p in per_query)[len(per_query)//2]:.1f}s",
                      flush=True)
        write_predictions(f"p7-{args.arm}", "L", set_name,
                          {"wall_s": 0,
                           "note": f"生产 scoped L（进程内只读，2026-10-08 富化舰队期）；"
                                   f"arm={args.arm}；oos={oos_total}"},
                          per_query, len(slice_ids))
    print(f"EXP7_{args.arm.upper()}_DONE")


if __name__ == "__main__":
    main()
