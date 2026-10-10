#!/usr/bin/env python3
"""实体层单变量消融（2026-10-09 审查补实验）：同配置下 实体依赖通道 开/关。

审查缺口（两篇论文共指）：C线#11/论文一#1——检索前后对照混入配置演进，
缺"实体通道 on/off"单变量归因。本脚本在 run_knc_scoped.py 同口径下
（进程内直调 _retrieve、strategy=A+doc_ids、stub Redis）跑两臂：
  arm=all      channels=None（全配置通道，与生产 scoped 相同）
  arm=noentity channels=["lexical","dense","docvec","synq"] + KNC_COOC_OFF=1
               （entity/ppr/graph 通道受 want=channels 门控即关；cooc 不看
               channels、面形扫描自触发——首版消融因此泄漏（nea 双臂逐题
               全零差），须用其专用 env 开关击杀，逐请求读 env 可臂间切换；
               synq 是 doc2query 资产非实体层依赖，保留防二次混杂）
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  .venv/bin/python eval/baselines/scripts/run_knc_scoped_ablate_channels.py --set nea'
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import NE_JOURNAL_KB, REPO, load_questions, load_slice_ids, write_predictions

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

NO_ENTITY_CHANNELS = ["lexical", "dense", "docvec", "synq"]


def scoped_retrieve(query, doc_ids, channels, top_k=20):
    req = M.RetrieveRequest.model_construct(
        kb_id=NE_JOURNAL_KB, kb_ids=None, query=query, top_k=60,
        channels=channels, doc_ids=list(doc_ids), strategy="A", session_id=None,
        tenant_id="public", acl_scopes=["public"])
    deadline = time.monotonic() + 120.0
    resp = M._retrieve(req, [NE_JOURNAL_KB], deadline,
                       lambda: time.monotonic() < deadline, "knc-chab-eval")
    sliceset = set(doc_ids)
    in_slice = [c for c in resp.chunks if c.get("doc_id") in sliceset][:top_k]
    oos = sum(1 for c in resp.chunks if c.get("doc_id") not in sliceset)
    return in_slice, oos, resp.channels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="L", choices=["L", "S"])
    ap.add_argument("--set", required=True)
    args = ap.parse_args()
    slice_ids = load_slice_ids(args.slice)
    questions = load_questions(args.set)

    for arm, channels in (("chab-all", None), ("chab-noentity", NO_ENTITY_CHANNELS)):
        if arm == "chab-noentity":
            os.environ["KNC_COOC_OFF"] = "1"  # 逐请求读 env；先跑 all 臂防污染
        per_query, oos_total = [], 0
        for q in questions:
            t0 = time.time()
            hits, oos, got = scoped_retrieve(q["query"], slice_ids, channels)
            dt = time.time() - t0
            oos_total += oos
            per_query.append({
                "query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
                "hits": [{"rank": i + 1, "doc_id": h["doc_id"],
                          "chunk_id": h.get("chunk_id")} for i, h in enumerate(hits)],
                "latency_s": round(dt, 3), "channels": got,
                "out_of_slice_hits": oos})
        write_predictions(arm, args.slice,
                          os.path.basename(args.set).replace(".jsonl", ""),
                          {"wall_s": 0, "note": f"channels={channels}; "
                           f"KNC_COOC_OFF={os.environ.get('KNC_COOC_OFF', '')}; oos {oos_total}"},
                          per_query, len(slice_ids))
        print(f"[done] {arm} {args.set} n={len(per_query)}")


if __name__ == "__main__":
    main()
