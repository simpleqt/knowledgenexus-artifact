#!/usr/bin/env python3
"""⑥ 票权调优包生产迁移差归因（2026-09-30）：单 env 组剥离臂。

背景（20260930_deepq 附带发现）：完整生产 env 相对裸配置（20260930_graphgov
prod 臂，同日同栈同题）nea48 hit@10 −4.2pt、hard93 hit@20 +4.3pt——切片上
采纳的票权/doc_fuse 调优在生产深轨上题型分裂。本脚本在完整生产 env 基线上
逐组剥离，定位伤 nea 的 env（基线=deepq prod 臂，同配置同日可配对比较）：

  nosynq   剥 RRF_SYNQ_BOOST_ASSOC + RRF_DOCVEC_ASSOC_W（意图门控票权对）
  nodfuse  剥 RRF_DOC_FUSE + RRF_DOC_FUSE_EMIT（文档整合重排）
  nodocvec 剥 KNC_DOCVEC_ON（题录向量通道）
  noalias  剥 KNC_LEX_ALIAS_EXPAND（词法双语别名扩展）

用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  .venv/bin/python /abs/path/env_attribution.py [--only nosynq,nodfuse] [--sets nea48,hard93]'
产物：results/20260930_envattr/env_attribution.json（分 (arm,set) 增量落盘）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402

KB = "21bcbed8ea6b4a5bbf55f3e3e9988c1f"
CH = ["lexical", "dense", "graph", "synq", "docvec", "cooc", "rerank", "ppr"]
ARMS: dict[str, list[str]] = {
    "nosynq": ["RRF_SYNQ_BOOST_ASSOC", "RRF_DOCVEC_ASSOC_W"],
    "nodfuse": ["RRF_DOC_FUSE", "RRF_DOC_FUSE_EMIT"],
    "nodocvec": ["KNC_DOCVEC_ON"],
    "noalias": ["KNC_LEX_ALIAS_EXPAND"],
    # 最优候选配置：撤 docvec 通道（深轨双集毒药）+ synq 提权对（hard 小幅
    # 负），保留 doc_fuse（hard hit@20 唯一增益源）与 alias（nea 强正）
    "optimal": ["KNC_DOCVEC_ON", "RRF_SYNQ_BOOST_ASSOC", "RRF_DOCVEC_ASSOC_W"],
    # ⑦ graph-aware 重排输入（KNC_RERANK_GRAPH_CTX，实现后追加跑）
    "gctx": [],
    # ⑦ 叠在最优候选配置上
    "optimal_gctx": ["KNC_DOCVEC_ON", "RRF_SYNQ_BOOST_ASSOC", "RRF_DOCVEC_ASSOC_W"],
}
GCTX_ENV = {"KNC_RERANK_GRAPH_CTX": "1"}
ALL_KEYS = sorted({k for v in ARMS.values() for k in v} | set(GCTX_ENV))
OUT = Path(REPO) / "eval/baselines/results/20260930_envattr/env_attribution.json"


def load_sets() -> dict[str, list[dict]]:
    hard = [json.loads(l) for l in
            open(Path(REPO) / "eval/baselines/results/20260929_hardtargets/"
                 "hard_targets_prod.jsonl", encoding="utf-8") if l.strip()]
    nea = [json.loads(l) for l in
           open(Path(REPO) / "eval/harness/ne_journal_assoc_q48.jsonl",
                encoding="utf-8") if l.strip()]
    ner = [json.loads(l) for l in
           open(Path(REPO) / "eval/harness/ne_journal_realq.jsonl",
                encoding="utf-8") if l.strip()]
    return {"nea48": nea, "hard93": hard, "realq39": ner}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="nea48,hard93")
    ap.add_argument("--only", default="nosynq,nodfuse,nodocvec,noalias")
    args = ap.parse_args()
    sets = {k: v for k, v in load_sets().items() if k in args.sets.split(",")}
    arms = [a for a in ARMS if a in args.only.split(",")]

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

    OUT.parent.mkdir(parents=True, exist_ok=True)
    results: dict = json.load(open(OUT)) if OUT.exists() else {}
    saved = {k: os.environ.get(k) for k in ALL_KEYS}
    try:
        for arm in arms:
            for k in ALL_KEYS:  # 每臂先恢复完整生产 env
                if saved[k] is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = saved[k]
            for k in ARMS[arm]:  # 再剥本臂组
                os.environ.pop(k, None)
            if arm in ("gctx", "optimal_gctx"):
                os.environ.update(GCTX_ENV)
            for set_name, rows in sets.items():
                key = f"{arm}|{set_name}"
                if key in results:
                    print(f"[skip] {key}", flush=True)
                    continue
                out_rows = []
                t_all = time.time()
                for qi, r in enumerate(rows):
                    gold = set(r["gold_doc_ids"])
                    req = M.RetrieveRequest.model_construct(
                        kb_id=KB, kb_ids=None, query=r["query"], top_k=50,
                        channels=CH, doc_ids=None, strategy="A", session_id=None,
                        tenant_id="public", acl_scopes=["public"])
                    deadline = time.monotonic() + 150.0
                    t0 = time.time()
                    resp = M._retrieve(req, [KB], deadline,
                                       lambda: time.monotonic() < deadline, f"enva-{arm}")
                    dt = time.time() - t0
                    docs = [c.get("doc_id") for c in resp.chunks]
                    fr = next((i for i, d in enumerate(docs, 1) if d in gold), None)
                    out_rows.append({"query": r["query"], "rank": fr,
                                     "latency_s": round(dt, 2),
                                     "channels": list(resp.channels)})
                    if (qi + 1) % 10 == 0:
                        print(f"[{key} {qi+1}/{len(rows)}]", flush=True)
                n = len(out_rows)
                lat = sorted(x["latency_s"] for x in out_rows)
                results[key] = {
                    "hit@10": round(sum(1 for x in out_rows if (x["rank"] or 99) <= 10) / n, 4),
                    "hit@20": round(sum(1 for x in out_rows if (x["rank"] or 99) <= 20) / n, 4),
                    "mrr@10": round(sum(1.0 / x["rank"] for x in out_rows
                                        if x["rank"] and x["rank"] <= 10) / n, 4),
                    "cooc_fired": sum(1 for x in out_rows if "cooc" in x["channels"]),
                    "rerank_fired": sum(1 for x in out_rows if "rerank" in x["channels"]),
                    "n": n, "p50_s": lat[n // 2],
                    "wall_s": round(time.time() - t_all, 1),
                    "rows": out_rows,
                }
                print(f"[{key}] hit@10={results[key]['hit@10']} hit@20={results[key]['hit@20']} "
                      f"mrr={results[key]['mrr@10']} rr={results[key]['rerank_fired']} "
                      f"p50={lat[n//2]}s", flush=True)
                json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)
    finally:
        for k, v in saved.items():  # 收尾恢复
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print(f"[written] {OUT}")


if __name__ == "__main__":
    main()
