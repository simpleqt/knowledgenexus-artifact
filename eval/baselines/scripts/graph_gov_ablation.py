#!/usr/bin/env python3
"""图谱治理消融（2026-09-30，O-1/O-2）：PPR 关系类型治理 + 种子稠密兜底。

背景（生产库实测，只读）：relation_registry 87% 是 authored_by/affiliated_with
书目星型边，graphppr 边加载无类型过滤、PPR 种子无书目类型守卫（graph 通道 L1
有）；词典扫描不足时 PPR 整通道哑火。本脚本五臂进程内直调生产 _retrieve
（只读），量化治理与兜底的边际：

  prod        现网 deep 行为（profile+rerank+ppr，env 全空）
  excl        KNC_PPR_REL_DOWN=authored_by:0,affiliated_with:0 + 种子排除书目类型
  dw          书目边降权 0.25 替代排除（防语义子图过薄的对照臂）
  excl_dense  excl + KNC_SEED_DENSE_FALLBACK=1（<2 种子时实体向量面补种）
  excl_entity excl + channels 追加 entity（稠密实体→块直连 vs PPR walk 对照）

题集：hard93（构造性难题，base 按构造不可见）+ nea48（模板关联题）。
两集分布不同，分集报告；逐题行落盘供配对 bootstrap。

用法：
  bash -c 'source eval/baselines/scripts/knc_env.sh && \
    .venv/bin/python /abs/path/graph_gov_ablation.py [--smoke]'
产物：results/20260930_graphgov/graph_gov_ablation.json（分 (arm,set) 增量落盘）
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
BASE = ["lexical", "dense", "graph", "synq", "docvec", "cooc"]  # 生产 profile + 门控通道
GOV_ENV = {
    "KNC_PPR_SEED_EXCLUDE_TYPES": "paper,person,organization",
}
ARMS: dict[str, dict] = {
    "prod":       {"env": {},                       "channels": BASE + ["rerank", "ppr"]},
    "excl":       {"env": {"KNC_PPR_REL_DOWN": "authored_by:0,affiliated_with:0", **GOV_ENV},
                   "channels": BASE + ["rerank", "ppr"]},
    "dw":         {"env": {"KNC_PPR_REL_DOWN": "authored_by:0.25,affiliated_with:0.25", **GOV_ENV},
                   "channels": BASE + ["rerank", "ppr"]},
    "excl_dense": {"env": {"KNC_PPR_REL_DOWN": "authored_by:0,affiliated_with:0",
                           "KNC_SEED_DENSE_FALLBACK": "1", **GOV_ENV},
                   "channels": BASE + ["rerank", "ppr"]},
    "excl_entity": {"env": {"KNC_PPR_REL_DOWN": "authored_by:0,affiliated_with:0", **GOV_ENV},
                    "channels": BASE + ["rerank", "ppr", "entity"]},
    # O-3 追加臂（--bppr 才跑）：治理栈 + 文档级二部 PPR 通道
    "excl_bppr":  {"env": {"KNC_PPR_REL_DOWN": "authored_by:0,affiliated_with:0", **GOV_ENV},
                   "channels": BASE + ["rerank", "ppr", "bppr"]},
}
ENV_KEYS = ["KNC_PPR_REL_DOWN", "KNC_PPR_SEED_EXCLUDE_TYPES", "KNC_SEED_DENSE_FALLBACK"]
OUT = Path(REPO) / "eval/baselines/results/20260930_graphgov/graph_gov_ablation.json"


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
    return {"hard93": hard, "nea48": nea, "ner70": ner}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="每集只跑前 3 题 × 3 臂")
    ap.add_argument("--bppr", action="store_true", help="追加 excl_bppr 臂（O-3）")
    ap.add_argument("--sets", default="hard93,nea48",
                    help="逗号分隔题集：hard93,nea48,ner70")
    ap.add_argument("--only", default="", help="只跑指定臂（逗号分隔，缺省全部未完成的）")
    args = ap.parse_args()
    sets = load_sets()
    sets = {k: v for k, v in sets.items() if k in args.sets.split(",")}
    if args.smoke:
        sets = {k: v[:3] for k, v in sets.items()}
        for a in list(ARMS):
            if a not in ("prod", "excl", "excl_dense"):
                del ARMS[a]
    if not args.bppr:
        ARMS.pop("excl_bppr", None)
    if args.only:
        for a in list(ARMS):
            if a not in args.only.split(","):
                del ARMS[a]

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
    os.environ["RERANK_DISABLED"] = ""  # deep 语义：强制重排可用

    OUT.parent.mkdir(parents=True, exist_ok=True)
    results: dict = json.load(open(OUT)) if OUT.exists() else {}
    for arm, cfg in ARMS.items():
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(cfg["env"])
        for set_name, rows in sets.items():
            key = f"{arm}|{set_name}"
            if key in results:  # 断点续跑
                print(f"[skip] {key}", flush=True)
                continue
            out_rows = []
            t_all = time.time()
            for qi, r in enumerate(rows):
                gold = set(r["gold_doc_ids"])
                req = M.RetrieveRequest.model_construct(
                    kb_id=KB, kb_ids=None, query=r["query"], top_k=50,
                    channels=cfg["channels"], doc_ids=None, strategy="A", session_id=None,
                    tenant_id="public", acl_scopes=["public"])
                deadline = time.monotonic() + 150.0
                t0 = time.time()
                resp = M._retrieve(req, [KB], deadline,
                                   lambda: time.monotonic() < deadline, f"ggov-{arm}")
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
                "ppr_fired": sum(1 for x in out_rows if "ppr" in x["channels"]),
                "entity_fired": sum(1 for x in out_rows if "entity" in x["channels"]),
                "n": n, "p50_s": lat[n // 2],
                "wall_s": round(time.time() - t_all, 1),
                "rows": out_rows,
            }
            print(f"[{key}] hit@10={results[key]['hit@10']} hit@20={results[key]['hit@20']} "
                  f"mrr={results[key]['mrr@10']} ppr={results[key]['ppr_fired']} "
                  f"ent={results[key]['entity_fired']} p50={lat[n//2]}s", flush=True)
            json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    print(f"[written] {OUT}")


if __name__ == "__main__":
    main()
