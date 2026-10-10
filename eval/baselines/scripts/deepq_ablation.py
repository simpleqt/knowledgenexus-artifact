#!/usr/bin/env python3
"""O-4 深轨查询理解消融（2026-09-30）：LLM 意图/实体 vs 模板词门。

背景（B77 定界）：关联意图模板词门自然覆盖率 4/70，自然措辞关联题是
cooc/synq 提权/取深门控唯一未触达作用面。O-4 在 deep 端点加一次 LLM
查询理解（意图分类+实体面形），结果挂 req.deep_llm：
  _assoc_intent 覆盖（synq 提权/B85/取深门） + cooc 面形并入。

两臂（进程内生产只读，top_k=50 深轨语义，判定离线冻结注入保确定性）：
  prod  deep_llm=None（现网行为）
  llm   deep_llm=verdicts.json[query]
题集：nea48（模板关联回归——LLM 应判 assoc 不回退）、realq70（自然措辞主
战场）、hard93（构造难题回归——直查/元数据不应误触发）。

用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  .venv/bin/python /abs/path/deepq_ablation.py [--only prod,llm] [--sets ...]'
产物：results/20260930_deepq/deepq_ablation.json（分 (arm,set) 增量落盘）
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
BASE = ["lexical", "dense", "graph", "synq", "docvec", "cooc"]
CH = BASE + ["rerank", "ppr"]
VERDICTS = Path(REPO) / "eval/baselines/results/20260930_deepq/verdicts.json"
OUT = Path(REPO) / "eval/baselines/results/20260930_deepq/deepq_ablation.json"
ARMS = ["prod", "llm"]


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
    return {"hard93": hard, "nea48": nea, "realq70": ner}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="nea48,realq70,hard93")
    ap.add_argument("--only", default="prod,llm")
    args = ap.parse_args()
    sets = {k: v for k, v in load_sets().items() if k in args.sets.split(",")}
    arms = [a for a in ARMS if a in args.only.split(",")]
    verdicts = json.load(open(VERDICTS, encoding="utf-8"))

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
    for arm in arms:
        for set_name, rows in sets.items():
            key = f"{arm}|{set_name}"
            if key in results:
                print(f"[skip] {key}", flush=True)
                continue
            out_rows = []
            t_all = time.time()
            for qi, r in enumerate(rows):
                gold = set(r["gold_doc_ids"])
                kw = dict(kb_id=KB, kb_ids=None, query=r["query"], top_k=50,
                          channels=CH, doc_ids=None, strategy="A",
                          session_id=None, tenant_id="public", acl_scopes=["public"])
                if arm == "llm":
                    kw["deep_llm"] = verdicts.get(r["query"])
                req = M.RetrieveRequest.model_construct(**kw)
                deadline = time.monotonic() + 150.0
                t0 = time.time()
                resp = M._retrieve(req, [KB], deadline,
                                   lambda: time.monotonic() < deadline, f"dq-{arm}")
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
                "n": n, "p50_s": lat[n // 2],
                "wall_s": round(time.time() - t_all, 1),
                "rows": out_rows,
            }
            print(f"[{key}] hit@10={results[key]['hit@10']} hit@20={results[key]['hit@20']} "
                  f"mrr={results[key]['mrr@10']} cooc={results[key]['cooc_fired']} "
                  f"p50={lat[n//2]}s", flush=True)
            json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"[written] {OUT}")


if __name__ == "__main__":
    main()
