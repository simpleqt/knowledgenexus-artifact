#!/usr/bin/env python3
"""自然措辞关联题集评测（2026-09-30）：词典解锁后 cooc 在 natural 面的价值。

两臂（进程内生产只读，部署深轨语义=optimal env）：
  dep     部署配置全通道（含 cooc）
  nocooc  同 env 去 cooc 通道（归因 cooc 边际）
题集：ne_journal_natural_assoc.jsonl（99 题，禁模板词构造，金标=概念对共现文档）。
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  .venv/bin/python /abs/path/natural_assoc_eval.py'
产物：results/20260930_naturalassoc/natural_assoc_eval.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO  # noqa: E402

KB = "21bcbed8ea6b4a5bbf55f3e3e9988c1f"
BASE = ["lexical", "dense", "graph", "synq", "docvec", "cooc"]
ARMS = {"dep": BASE + ["rerank", "ppr"],
        "nocooc": [c for c in BASE if c != "cooc"] + ["rerank", "ppr"]}
SETFILE = Path(REPO) / "eval/harness/ne_journal_natural_assoc.jsonl"
OUT = Path(REPO) / "eval/baselines/results/20260930_naturalassoc/natural_assoc_eval.json"


def main() -> None:
    rows = [json.loads(l) for l in open(SETFILE, encoding="utf-8") if l.strip()]
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
    os.environ["RERANK_DISABLED"] = ""
    # 部署深轨语义：KNC_DEEP_ENV_UNSET 同款撤除（gctx 已由 knc_env.sh 提供）
    for k in ("KNC_DOCVEC_ON", "RRF_SYNQ_BOOST_ASSOC", "RRF_DOCVEC_ASSOC_W"):
        os.environ.pop(k, None)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results: dict = json.load(open(OUT)) if OUT.exists() else {}
    for arm, ch in ARMS.items():
        if arm in results:
            print(f"[skip] {arm}")
            continue
        # cooc 是融合后自适应注入（非 want 门控通道）——真正的开关是
        # KNC_COOC_OFF；channels 里去掉仅是元数据一致性
        if arm == "nocooc":
            os.environ["KNC_COOC_OFF"] = "1"
        else:
            os.environ.pop("KNC_COOC_OFF", None)
        out_rows = []
        for qi, r in enumerate(rows):
            gold = set(r["gold_doc_ids"])
            req = M.RetrieveRequest.model_construct(
                kb_id=KB, kb_ids=None, query=r["query"], top_k=50,
                channels=ch, doc_ids=None, strategy="A", session_id=None,
                tenant_id="public", acl_scopes=["public"])
            dl = time.monotonic() + 150.0
            t0 = time.time()
            resp = M._retrieve(req, [KB], dl, lambda: time.monotonic() < dl, f"na-{arm}")
            docs = [c.get("doc_id") for c in resp.chunks]
            fr = next((i for i, d in enumerate(docs, 1) if d in gold), None)
            out_rows.append({"query": r["query"], "rank": fr,
                             "latency_s": round(time.time() - t0, 2),
                             "channels": list(resp.channels)})
            if (qi + 1) % 10 == 0:
                print(f"[{arm} {qi+1}/{len(rows)}]", flush=True)
        n = len(out_rows)
        results[arm] = {
            "hit@10": round(sum(1 for x in out_rows if (x["rank"] or 99) <= 10) / n, 4),
            "hit@20": round(sum(1 for x in out_rows if (x["rank"] or 99) <= 20) / n, 4),
            "mrr@10": round(sum(1.0 / x["rank"] for x in out_rows
                                if x["rank"] and x["rank"] <= 10) / n, 4),
            "cooc_fired": sum(1 for x in out_rows if "cooc" in x["channels"]),
            "n": n, "rows": out_rows,
        }
        print(f"[{arm}] hit@10={results[arm]['hit@10']} hit@20={results[arm]['hit@20']} "
              f"mrr={results[arm]['mrr@10']} cooc={results[arm]['cooc_fired']}", flush=True)
        json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"[written] {OUT}")


if __name__ == "__main__":
    main()
