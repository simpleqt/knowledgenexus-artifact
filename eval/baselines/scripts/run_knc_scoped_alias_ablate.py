#!/usr/bin/env python3
"""双语别名扩展单变量消融（2026-10-10 大修 P0-4）：

KNC_LEX_ALIAS_EXPAND 逐请求读 env（main.py lexical 通道），进程内可臂间切换。
臂：kaa-alias-on（env 继承 knc_env.sh）vs kaa-alias-off（unset）。
同 run_knc_scoped.py 口径（in-process、strategy=A+doc_ids、stub Redis）。
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


def scoped_retrieve(query, doc_ids, top_k=20):
    req = M.RetrieveRequest.model_construct(
        kb_id=NE_JOURNAL_KB, kb_ids=None, query=query, top_k=60, channels=None,
        doc_ids=list(doc_ids), strategy="A", session_id=None,
        tenant_id="public", acl_scopes=["public"])
    deadline = time.monotonic() + 120.0
    resp = M._retrieve(req, [NE_JOURNAL_KB], deadline,
                       lambda: time.monotonic() < deadline, "knc-kaa-eval")
    s = set(doc_ids)
    hits = [c for c in resp.chunks if c.get("doc_id") in s][:top_k]
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="L", choices=["L", "S"])
    ap.add_argument("--set", required=True)
    args = ap.parse_args()
    slice_ids = load_slice_ids(args.slice)
    questions = load_questions(args.set)
    assert os.environ.get("KNC_LEX_ALIAS_EXPAND"), "on 臂须由 knc_env.sh 带入该 env"

    for arm in ("kaa-alias-on", "kaa-alias-off"):
        if arm == "kaa-alias-off":
            del os.environ["KNC_LEX_ALIAS_EXPAND"]
        per_query = []
        for q in questions:
            t0 = time.time()
            hits = scoped_retrieve(q["query"], slice_ids)
            per_query.append({
                "query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
                "hits": [{"rank": i + 1, "doc_id": h["doc_id"],
                          "chunk_id": h.get("chunk_id")} for i, h in enumerate(hits)],
                "latency_s": round(time.time() - t0, 3)})
        write_predictions(arm, args.slice,
                          os.path.basename(args.set).replace(".jsonl", ""),
                          {"wall_s": 0, "note": f"alias={'on' if arm.endswith('on') else 'off'}"},
                          per_query, len(slice_ids))
        print(f"[done] {arm} {args.set} n={len(per_query)}")


if __name__ == "__main__":
    main()
